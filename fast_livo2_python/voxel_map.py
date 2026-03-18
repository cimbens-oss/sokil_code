"""Hierarchical voxel octree map for LiDAR-inertial odometry."""
import numpy as np
from so3_math import skew_sym, skew_sym_batch, exp_so3_dt
from states import StatesGroup, PointWithVar, PointToPlane, DIM_STATE

VOXELMAP_HASH_P = 116101
VOXELMAP_MAX_N = 10000000000

_voxel_plane_id = 0


def voxel_location_hash(x, y, z):
    return ((((z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (x)


def calc_body_cov(pb, range_inc, degree_inc):
    """Calculate measurement covariance in body frame (single point)."""
    pb = pb.copy()
    if pb[2] == 0:
        pb[2] = 0.0001
    rng = np.linalg.norm(pb)
    range_var = range_inc ** 2
    direction = pb / rng
    deg_var = np.sin(np.radians(degree_inc)) ** 2
    direction_var = np.diag([deg_var, deg_var])

    if abs(direction[2]) > 1e-6:
        base1 = np.array([1.0, 1.0, -(direction[0] + direction[1]) / direction[2]])
    elif abs(direction[1]) > 1e-6:
        base1 = np.array([1.0, -(direction[0] + direction[2]) / direction[1], 1.0])
    else:
        base1 = np.array([-(direction[1] + direction[2]) / direction[0], 1.0, 1.0])
    base1 = base1 / np.linalg.norm(base1)
    base2 = np.cross(base1, direction)
    base2 = base2 / np.linalg.norm(base2)

    direction_hat = skew_sym(direction)
    N = np.column_stack([base1, base2])
    A = rng * direction_hat @ N
    cov = np.outer(direction, direction) * range_var + A @ direction_var @ A.T
    return cov


def calc_body_cov_batch(points, range_inc, degree_inc):
    """Calculate measurement covariance for all points at once. Returns Nx3x3."""
    n = len(points)
    pts = points.copy()
    mask_z = pts[:, 2] == 0
    pts[mask_z, 2] = 0.0001
    ranges = np.linalg.norm(pts, axis=1)  # N
    range_var = range_inc ** 2
    d = pts / ranges[:, None]  # Nx3 directions
    deg_var = np.sin(np.radians(degree_inc)) ** 2

    abs_d = np.abs(d)

    # Build orthonormal basis vectors (vectorized)
    base1 = np.zeros((n, 3))
    m1 = abs_d[:, 2] > 1e-6
    m2 = (~m1) & (abs_d[:, 1] > 1e-6)
    m3 = ~m1 & ~m2

    if np.any(m1):
        base1[m1, 0] = 1.0
        base1[m1, 1] = 1.0
        base1[m1, 2] = -(d[m1, 0] + d[m1, 1]) / d[m1, 2]
    if np.any(m2):
        base1[m2, 0] = 1.0
        base1[m2, 1] = -(d[m2, 0] + d[m2, 2]) / d[m2, 1]
        base1[m2, 2] = 1.0
    if np.any(m3):
        base1[m3, 0] = -(d[m3, 1] + d[m3, 2]) / d[m3, 0]
        base1[m3, 1] = 1.0
        base1[m3, 2] = 1.0

    base1 /= np.linalg.norm(base1, axis=1, keepdims=True)
    base2 = np.cross(base1, d)
    base2 /= np.linalg.norm(base2, axis=1, keepdims=True)

    # Batch skew-symmetric of d: Nx3x3
    d_hat = skew_sym_batch(d)

    # N_mat = [base1, base2]: Nx3x2
    N_mat = np.stack([base1, base2], axis=-1)

    # A = range * d_hat @ N_mat: Nx3x2
    A = ranges[:, None, None] * np.einsum('nij,njk->nik', d_hat, N_mat)

    # cov = outer(d,d)*range_var + deg_var * A @ A.T
    dd = d[:, :, None] * d[:, None, :]  # Nx3x3
    AAt = np.einsum('nij,nkj->nik', A, A)  # Nx3x3
    covs = dd * range_var + AAt * deg_var
    return covs


class VoxelPlane:
    def __init__(self):
        self.center = np.zeros(3)
        self.normal = np.zeros(3)
        self.y_normal = np.zeros(3)
        self.x_normal = np.zeros(3)
        self.covariance = np.zeros((3, 3))
        self.plane_var = np.zeros((6, 6))
        self.radius = 0.0
        self.min_eigen_value = 1.0
        self.mid_eigen_value = 1.0
        self.max_eigen_value = 1.0
        self.d = 0.0
        self.points_size = 0
        self.is_plane = False
        self.is_init = False
        self.is_update = False
        self.id = 0


class VoxelOctoTree:
    def __init__(self, max_layer, layer, points_size_threshold, max_points_num, planer_threshold):
        self.max_layer = max_layer
        self.layer = layer
        self.points_size_threshold = points_size_threshold
        self.max_points_num = max_points_num
        self.planer_threshold = planer_threshold

        self.temp_points = []  # list of PointWithVar
        self.octo_state = 0
        self.new_points = 0
        self.update_size_threshold = 5
        self.init_octo = False
        self.update_enable = True
        self.leaves = [None] * 8
        self.plane_ptr = VoxelPlane()
        self.voxel_center = np.zeros(3)
        self.quater_length = 0.0
        self.layer_init_num = []

    def init_plane(self, points, plane):
        """Initialize plane from points using PCA."""
        global _voxel_plane_id
        plane.plane_var = np.zeros((6, 6))
        plane.covariance = np.zeros((3, 3))
        plane.center = np.zeros(3)
        n = len(points)
        plane.points_size = n
        plane.radius = 0.0

        # Vectorized covariance computation
        pts_w = np.array([pv.point_w for pv in points])  # Nx3
        plane.center = pts_w.mean(axis=0)
        centered = pts_w - plane.center
        plane.covariance = (centered.T @ centered) / n

        eigenvalues, eigenvectors = np.linalg.eigh(plane.covariance)
        # eigh returns sorted ascending
        idx_min = 0
        idx_mid = 1
        idx_max = 2

        if eigenvalues[idx_min] < self.planer_threshold:
            # Vectorized plane variance computation
            J_Q = np.eye(3) / n
            e_min = eigenvectors[:, idx_min]  # 3
            plane_var = np.zeros((6, 6))

            # Precompute the symmetric matrix pairs for m != idx_min
            # M_m = e_m @ e_min.T + e_min @ e_m.T, scaled by 1/(n*(lam_min - lam_m))
            diffs = centered  # Nx3 (already computed)
            pt_vars = np.array([pv.var for pv in points])  # Nx3x3

            for pt_idx in range(n):
                diff = diffs[pt_idx]
                F = np.zeros((3, 3))
                for m in range(3):
                    if m != idx_min:
                        scale = 1.0 / (n * (eigenvalues[idx_min] - eigenvalues[m]))
                        e_m = eigenvectors[:, m]
                        sym_mat = np.outer(e_m, e_min) + np.outer(e_min, e_m)
                        F[m, :] = diff @ sym_mat * scale
                J = np.zeros((6, 3))
                J[:3, :] = eigenvectors @ F
                J[3:6, :] = J_Q
                plane_var += J @ pt_vars[pt_idx] @ J.T

            plane.plane_var = plane_var

            plane.normal = eigenvectors[:, idx_min]
            plane.y_normal = eigenvectors[:, idx_mid]
            plane.x_normal = eigenvectors[:, idx_max]
            plane.min_eigen_value = eigenvalues[idx_min]
            plane.mid_eigen_value = eigenvalues[idx_mid]
            plane.max_eigen_value = eigenvalues[idx_max]
            plane.radius = np.sqrt(eigenvalues[idx_max])
            plane.d = -(plane.normal @ plane.center)
            plane.is_plane = True
            plane.is_update = True
            if not plane.is_init:
                plane.id = _voxel_plane_id
                _voxel_plane_id += 1
                plane.is_init = True
        else:
            plane.is_update = True
            plane.is_plane = False

    def init_octo_tree(self):
        if len(self.temp_points) > self.points_size_threshold:
            self.init_plane(self.temp_points, self.plane_ptr)
            if self.plane_ptr.is_plane:
                self.octo_state = 0
                if len(self.temp_points) > self.max_points_num:
                    self.update_enable = False
                    self.temp_points = []
                    self.new_points = 0
            else:
                self.octo_state = 1
                self.cut_octo_tree()
            self.init_octo = True
            self.new_points = 0

    def cut_octo_tree(self):
        if self.layer >= self.max_layer:
            self.octo_state = 0
            return

        vc = self.voxel_center
        for pt in self.temp_points:
            pw = pt.point_w
            leafnum = (4 if pw[0] > vc[0] else 0) + (2 if pw[1] > vc[1] else 0) + (1 if pw[2] > vc[2] else 0)
            if self.leaves[leafnum] is None:
                child = VoxelOctoTree(self.max_layer, self.layer + 1,
                                      self.layer_init_num[self.layer + 1] if self.layer + 1 < len(self.layer_init_num) else 5,
                                      self.max_points_num, self.planer_threshold)
                child.layer_init_num = self.layer_init_num
                ql = self.quater_length
                child.voxel_center[0] = vc[0] + (1 if leafnum & 4 else -1) * ql
                child.voxel_center[1] = vc[1] + (1 if leafnum & 2 else -1) * ql
                child.voxel_center[2] = vc[2] + (1 if leafnum & 1 else -1) * ql
                child.quater_length = ql / 2
                self.leaves[leafnum] = child
            self.leaves[leafnum].temp_points.append(pt)
            self.leaves[leafnum].new_points += 1

        for i in range(8):
            if self.leaves[i] is not None:
                if len(self.leaves[i].temp_points) > self.leaves[i].points_size_threshold:
                    self.leaves[i].init_plane(self.leaves[i].temp_points, self.leaves[i].plane_ptr)
                    if self.leaves[i].plane_ptr.is_plane:
                        self.leaves[i].octo_state = 0
                        if len(self.leaves[i].temp_points) > self.leaves[i].max_points_num:
                            self.leaves[i].update_enable = False
                            self.leaves[i].temp_points = []
                            self.leaves[i].new_points = 0
                    else:
                        self.leaves[i].octo_state = 1
                        self.leaves[i].cut_octo_tree()
                    self.leaves[i].init_octo = True
                    self.leaves[i].new_points = 0

    def UpdateOctoTree(self, pv):
        if not self.init_octo:
            self.new_points += 1
            self.temp_points.append(pv)
            if len(self.temp_points) > self.points_size_threshold:
                self.init_octo_tree()
        else:
            if self.plane_ptr.is_plane:
                if self.update_enable:
                    self.new_points += 1
                    self.temp_points.append(pv)
                    if self.new_points > self.update_size_threshold:
                        self.init_plane(self.temp_points, self.plane_ptr)
                        self.new_points = 0
                    if len(self.temp_points) >= self.max_points_num:
                        self.update_enable = False
                        self.temp_points = []
                        self.new_points = 0
            else:
                if self.layer < self.max_layer:
                    pw = pv.point_w
                    vc = self.voxel_center
                    leafnum = (4 if pw[0] > vc[0] else 0) + (2 if pw[1] > vc[1] else 0) + (1 if pw[2] > vc[2] else 0)
                    if self.leaves[leafnum] is not None:
                        self.leaves[leafnum].UpdateOctoTree(pv)
                    else:
                        child = VoxelOctoTree(self.max_layer, self.layer + 1,
                                              self.layer_init_num[self.layer + 1] if self.layer + 1 < len(self.layer_init_num) else 5,
                                              self.max_points_num, self.planer_threshold)
                        child.layer_init_num = self.layer_init_num
                        ql = self.quater_length
                        child.voxel_center[0] = vc[0] + (1 if leafnum & 4 else -1) * ql
                        child.voxel_center[1] = vc[1] + (1 if leafnum & 2 else -1) * ql
                        child.voxel_center[2] = vc[2] + (1 if leafnum & 1 else -1) * ql
                        child.quater_length = ql / 2
                        self.leaves[leafnum] = child
                        child.UpdateOctoTree(pv)
                else:
                    if self.update_enable:
                        self.new_points += 1
                        self.temp_points.append(pv)
                        if self.new_points > self.update_size_threshold:
                            self.init_plane(self.temp_points, self.plane_ptr)
                            self.new_points = 0
                        if len(self.temp_points) > self.max_points_num:
                            self.update_enable = False
                            self.temp_points = []
                            self.new_points = 0


class VoxelMapManager:
    def __init__(self, config):
        self.config = config
        self.voxel_map = {}  # dict of (x,y,z) -> VoxelOctoTree
        self.extR = np.eye(3)
        self.extT = np.zeros(3)
        self.state = StatesGroup()
        self.position_last = np.zeros(3)
        self.last_slide_position = np.zeros(3)

        self.feats_undistort = None
        self.feats_down_body = None
        self.feats_down_world = None
        self.feats_down_size = 0

        self.cross_mat_list = []
        self.body_cov_list = []
        self.pv_list = []
        self.ptpl_list = []

    def TransformLidar(self, rot, t, input_cloud):
        """Transform point cloud from body to world frame. Returns Nx3 (vectorized)."""
        RE = rot @ self.extR
        Te = rot @ self.extT + t
        return input_cloud @ RE.T + Te[None, :]

    def BuildVoxelMap(self):
        """Build initial voxel map from first scan (vectorized covariance)."""
        voxel_size = self.config.voxel_size
        planer_threshold = self.config.min_eigen_value
        max_layer = self.config.max_layer
        max_points_num = self.config.max_points_num
        layer_init_num = self.config.layer_init_num
        n = len(self.feats_down_world)

        # Batch compute body covariances
        body_covs = calc_body_cov_batch(self.feats_down_body, self.config.dept_err, self.config.beam_err)

        # Batch transform covariances to world frame
        RE = self.state.rot_end @ self.extR
        RcovR = np.einsum('ij,njk,lk->nil', RE, body_covs, RE)  # Nx3x3
        pts_lidar = self.feats_down_body @ self.extR.T + self.extT[None, :]
        cross_mats = skew_sym_batch(pts_lidar)  # Nx3x3
        rot_var = self.state.cov[0:3, 0:3]
        t_var = self.state.cov[3:6, 3:6]
        CrC = np.einsum('nij,jk,nlk->nil', cross_mats, rot_var, cross_mats)
        covs_world = RcovR + CrC + t_var[None, :, :]

        # Batch compute voxel keys
        loc = self.feats_down_world / voxel_size
        loc = np.where(loc < 0, loc - 1.0, loc)
        keys_int = loc.astype(np.int64)

        for i in range(n):
            pv = PointWithVar()
            pv.point_w = self.feats_down_world[i]
            pv.var = covs_world[i]
            key = (keys_int[i, 0], keys_int[i, 1], keys_int[i, 2])

            if key in self.voxel_map:
                self.voxel_map[key].temp_points.append(pv)
                self.voxel_map[key].new_points += 1
            else:
                octo = VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold)
                octo.layer_init_num = layer_init_num
                octo.quater_length = voxel_size / 4
                octo.voxel_center[0] = (0.5 + key[0]) * voxel_size
                octo.voxel_center[1] = (0.5 + key[1]) * voxel_size
                octo.voxel_center[2] = (0.5 + key[2]) * voxel_size
                octo.temp_points.append(pv)
                octo.new_points += 1
                self.voxel_map[key] = octo

        for octo in self.voxel_map.values():
            octo.init_octo_tree()

    def UpdateVoxelMap(self, input_points):
        """Incrementally update voxel map."""
        voxel_size = self.config.voxel_size
        planer_threshold = self.config.min_eigen_value
        max_layer = self.config.max_layer
        max_points_num = self.config.max_points_num
        layer_init_num = self.config.layer_init_num

        # Vectorized key computation (same as BuildVoxelMap)
        pts_w = np.array([pv.point_w for pv in input_points])
        loc_all = pts_w / voxel_size
        loc_all = np.where(loc_all < 0, loc_all - 1.0, loc_all)
        keys_int = loc_all.astype(np.int64)

        for i, pv in enumerate(input_points):
            key = (keys_int[i, 0], keys_int[i, 1], keys_int[i, 2])

            if key in self.voxel_map:
                self.voxel_map[key].UpdateOctoTree(pv)
            else:
                octo = VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold)
                octo.layer_init_num = layer_init_num
                octo.quater_length = voxel_size / 4
                octo.voxel_center[0] = (0.5 + key[0]) * voxel_size
                octo.voxel_center[1] = (0.5 + key[1]) * voxel_size
                octo.voxel_center[2] = (0.5 + key[2]) * voxel_size
                octo.UpdateOctoTree(pv)
                self.voxel_map[key] = octo

    def build_single_residual(self, pv, current_octo, current_layer):
        """Try to match a point to a plane in the octree. Returns (success, prob, ptpl)."""
        sigma_num = 3.0
        radius_k = 3.0
        p_w = pv.point_w

        if current_octo.plane_ptr.is_plane:
            plane = current_octo.plane_ptr
            dis_to_plane = abs(plane.normal @ p_w + plane.d)
            diff = plane.center - p_w
            dis_to_center = np.dot(diff, diff)
            range_dis = np.sqrt(max(dis_to_center - dis_to_plane * dis_to_plane, 0))

            if range_dis <= radius_k * plane.radius:
                J_nq = np.zeros(6)
                J_nq[:3] = p_w - plane.center
                J_nq[3:] = -plane.normal
                sigma_l = J_nq @ plane.plane_var @ J_nq
                sigma_l += plane.normal @ pv.var @ plane.normal

                if dis_to_plane < sigma_num * np.sqrt(max(sigma_l, 1e-12)):
                    prob = 1.0 / np.sqrt(max(sigma_l, 1e-12)) * np.exp(-0.5 * dis_to_plane * dis_to_plane / max(sigma_l, 1e-12))
                    ptpl = PointToPlane()
                    ptpl.body_cov = pv.body_var
                    ptpl.point_b = pv.point_b
                    ptpl.point_w = pv.point_w
                    ptpl.plane_var = plane.plane_var
                    ptpl.normal = plane.normal
                    ptpl.center = plane.center
                    ptpl.d = plane.d
                    ptpl.layer = current_layer
                    ptpl.dis_to_plane = plane.normal @ p_w + plane.d
                    pv.normal = plane.normal.copy()
                    return True, prob, ptpl
            return False, 0.0, None
        else:
            if current_layer < self.config.max_layer:
                best_prob = 0.0
                best_ptpl = None
                success = False
                for leaf in current_octo.leaves:
                    if leaf is not None:
                        ok, prob, ptpl = self.build_single_residual(pv, leaf, current_layer + 1)
                        if ok and prob > best_prob:
                            best_prob = prob
                            best_ptpl = ptpl
                            success = True
                return success, best_prob, best_ptpl
            return False, 0.0, None

    def BuildResidualList(self, pv_list, world_pts=None):
        """Build residual list by matching points to voxel map planes."""
        voxel_size = self.config.voxel_size
        ptpl_list = []

        # Pre-compute all voxel keys at once
        n = len(pv_list)
        if world_pts is None:
            world_pts = np.array([pv.point_w for pv in pv_list])  # Nx3
        loc = world_pts / voxel_size
        loc = np.where(loc < 0, loc - 1.0, loc)
        keys_int = loc.astype(np.int64)

        for idx in range(n):
            pv = pv_list[idx]
            key = (keys_int[idx, 0], keys_int[idx, 1], keys_int[idx, 2])

            if key in self.voxel_map:
                current_octo = self.voxel_map[key]
                ok, prob, ptpl = self.build_single_residual(pv, current_octo, 0)
                if not ok:
                    near = list(key)
                    vc = current_octo.voxel_center
                    ql = current_octo.quater_length
                    li = loc[idx]
                    if li[0] > vc[0] + ql: near[0] += 1
                    elif li[0] < vc[0] - ql: near[0] -= 1
                    if li[1] > vc[1] + ql: near[1] += 1
                    elif li[1] < vc[1] - ql: near[1] -= 1
                    if li[2] > vc[2] + ql: near[2] += 1
                    elif li[2] < vc[2] - ql: near[2] -= 1
                    near_key = tuple(near)
                    if near_key in self.voxel_map:
                        ok, prob, ptpl = self.build_single_residual(pv, self.voxel_map[near_key], 0)
                if ok:
                    ptpl_list.append(ptpl)

        return ptpl_list

    def StateEstimation(self, state_propagat):
        """Iterated EKF state estimation using point-to-plane residuals (vectorized)."""
        # Batch compute body covariances and cross matrices
        body_covs = calc_body_cov_batch(self.feats_down_body, self.config.dept_err, self.config.beam_err)
        # Transform body points: pts_lidar = extR @ p + extT
        pts_lidar = self.feats_down_body @ self.extR.T + self.extT[None, :]  # Nx3
        cross_mats = skew_sym_batch(pts_lidar)  # Nx3x3

        self.body_cov_list = body_covs
        self.cross_mat_list = cross_mats
        self.pv_list = [PointWithVar() for _ in range(self.feats_down_size)]

        rematch_num = 0
        I_STATE = np.eye(DIM_STATE)

        for iterCount in range(self.config.lio_max_iterations):
            # Vectorized world transform
            world_pts = self.TransformLidar(self.state.rot_end, self.state.pos_end, self.feats_down_body)
            rot_var = self.state.cov[0:3, 0:3]
            t_var = self.state.cov[3:6, 3:6]
            R = self.state.rot_end

            # Batch covariance in world frame:
            # cov_w = R @ body_cov @ R.T + cross @ rot_var @ cross.T + t_var
            RcovR = np.einsum('ij,njk,lk->nil', R, body_covs, R)  # Nx3x3
            CrC = np.einsum('nij,jk,nlk->nil', cross_mats, rot_var, cross_mats)  # Nx3x3
            covs_world = RcovR + CrC + t_var[None, :, :]

            for i in range(self.feats_down_size):
                pv = self.pv_list[i]
                pv.point_b = self.feats_down_body[i]
                pv.point_w = world_pts[i]
                pv.var = covs_world[i]
                pv.body_var = body_covs[i]

            self.ptpl_list = self.BuildResidualList(self.pv_list, world_pts)
            effct_feat_num = len(self.ptpl_list)

            if effct_feat_num < 3:
                print(f"[LIO] Too few effective features: {effct_feat_num}")
                break

            total_residual = sum(abs(p.dis_to_plane) for p in self.ptpl_list)
            _last_iter_msg = (f"[LIO] Iter {iterCount}: effective features: {effct_feat_num}, "
                              f"avg residual: {total_residual / effct_feat_num:.6f}")

            # Extract arrays from ptpl_list in a single pass
            n_eff = effct_feat_num
            ptpl_points_b = np.empty((n_eff, 3))
            ptpl_normals = np.empty((n_eff, 3))
            ptpl_centers = np.empty((n_eff, 3))
            ptpl_body_covs = np.empty((n_eff, 3, 3))
            ptpl_plane_vars = np.empty((n_eff, 6, 6))
            ptpl_dis = np.empty(n_eff)
            for _i, p in enumerate(self.ptpl_list):
                ptpl_points_b[_i] = p.point_b
                ptpl_normals[_i] = p.normal
                ptpl_centers[_i] = p.center
                ptpl_body_covs[_i] = p.body_cov
                ptpl_plane_vars[_i] = p.plane_var
                ptpl_dis[_i] = p.dis_to_plane

            # point_this = extR @ point_b + extT
            pts_this = ptpl_points_b @ self.extR.T + self.extT[None, :]  # Mx3
            pts_cross = skew_sym_batch(pts_this)  # Mx3x3
            # point_world = state_propagat.rot_end @ pts_this + pos
            R_prop = state_propagat.rot_end
            pts_world = pts_this @ R_prop.T + state_propagat.pos_end[None, :]  # Mx3

            # J_nq: Mx6
            J_nq = np.zeros((n_eff, 6))
            J_nq[:, :3] = pts_world - ptpl_centers
            J_nq[:, 3:] = -ptpl_normals

            # sigma_l = J_nq @ plane_var @ J_nq (per-point)
            sigma_l = np.einsum('ni,nij,nj->n', J_nq, ptpl_plane_vars, J_nq)  # M

            # var = R_prop @ extR @ body_cov @ (R_prop @ extR).T
            RE_prop = R_prop @ self.extR
            var_world = np.einsum('ij,njk,lk->nil', RE_prop, ptpl_body_covs, RE_prop)  # Mx3x3
            # normal @ var @ normal per point
            nVn = np.einsum('ni,nij,nj->n', ptpl_normals, var_world, ptpl_normals)  # M
            R_inv = 1.0 / (0.001 + sigma_l + nVn)

            # A = cross @ R.T @ normal -> Mx3
            RtN = ptpl_normals @ self.state.rot_end  # Mx3 (normal @ R = (R.T @ normal).T)
            A = np.einsum('nij,nj->ni', pts_cross, RtN)  # Mx3

            Hsub = np.zeros((n_eff, 6))
            Hsub[:, :3] = A
            Hsub[:, 3:6] = ptpl_normals
            meas_vec = -ptpl_dis

            Hsub_T_R_inv = Hsub.T * R_inv[None, :]  # 6xM

            HTz = Hsub_T_R_inv @ meas_vec
            H_T_H = np.zeros((DIM_STATE, DIM_STATE))
            H_T_H[:6, :6] = Hsub_T_R_inv @ Hsub

            K_1 = np.linalg.inv(H_T_H + np.linalg.inv(self.state.cov))
            G = np.zeros((DIM_STATE, DIM_STATE))
            G[:, :6] = K_1[:, :6] @ H_T_H[:6, :6]

            vec = state_propagat.diff(self.state)
            solution = K_1[:, :6] @ HTz + vec - G[:, :6] @ vec[:6]

            self.state.iadd(solution)
            rot_add = solution[:3]
            t_add = solution[3:6]

            if np.linalg.norm(rot_add) * 57.3 < 0.01 and np.linalg.norm(t_add) * 100 < 0.015:
                flg_EKF_converged = True
            else:
                flg_EKF_converged = False

            if flg_EKF_converged or (rematch_num == 0 and iterCount == self.config.lio_max_iterations - 2):
                rematch_num += 1

            if rematch_num >= 2 or iterCount == self.config.lio_max_iterations - 1:
                self.state.cov = (I_STATE - G) @ self.state.cov
                self.position_last = self.state.pos_end.copy()
                print(_last_iter_msg)
                break

    def mapSliding(self):
        """Slide local map to keep memory bounded."""
        if np.linalg.norm(self.position_last - self.last_slide_position) < self.config.sliding_thresh:
            return

        self.last_slide_position = self.position_last.copy()
        voxel_size = self.config.voxel_size
        loc = self.position_last / voxel_size
        loc = np.where(loc < 0, loc - 1.0, loc)

        cx, cy, cz = int(loc[0]), int(loc[1]), int(loc[2])
        hs = self.config.half_map_size
        x_lo, x_hi = cx - hs, cx + hs
        y_lo, y_hi = cy - hs, cy + hs
        z_lo, z_hi = cz - hs, cz + hs

        keys_to_remove = [
            key for key in self.voxel_map
            if key[0] > x_hi or key[0] < x_lo or key[1] > y_hi or key[1] < y_lo or key[2] > z_hi or key[2] < z_lo
        ]

        for key in keys_to_remove:
            del self.voxel_map[key]
