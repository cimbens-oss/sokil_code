"""Visual-Inertial Odometry (VIO) manager - simplified Python implementation.

This implements the core direct photometric tracking from FAST-LIVO2:
- Grid-based feature management
- Affine warp tracking with multi-level pyramids
- EKF update with photometric residuals
"""
import numpy as np
import cv2
from so3_math import skew_sym, exp_so3_xyz, log_so3
from states import StatesGroup, PointWithVar, DIM_STATE


class PinholeCamera:
    """Simple pinhole camera model."""
    def __init__(self, width, height, fx, fy, cx, cy, distortion=None):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = distortion if distortion is not None else np.zeros(4)

    def world2cam(self, xyz):
        """Project 3D point (in camera frame) to pixel coordinates."""
        return np.array([self.fx * xyz[0] / xyz[2] + self.cx,
                         self.fy * xyz[1] / xyz[2] + self.cy])

    def cam2world(self, u, v):
        """Backproject pixel to unit ray in camera frame."""
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])

    def is_in_frame(self, px, border=0):
        return (border <= px[0] < self.width - border and
                border <= px[1] < self.height - border)


class VisualPoint:
    """A 3D map point with visual observations."""
    _id_counter = 0

    def __init__(self, pos, normal=None):
        self.pos = pos.copy()
        self.normal = normal.copy() if normal is not None else np.zeros(3)
        self.ref_patch = None  # reference image patch (flattened float array)
        self.ref_px = None     # reference pixel position
        self.ref_img = None    # reference image (grayscale)
        self.ref_T_f_w = None  # reference frame pose (4x4)
        self.ref_inv_expo = 1.0
        self.is_initialized = False
        self.id = VisualPoint._id_counter
        VisualPoint._id_counter += 1


class VIOManager:
    """Simplified VIO manager for direct photometric tracking."""

    def __init__(self, config):
        self.config = config
        self.cam = None
        self.state = None
        self.state_propagat = None

        # Extrinsics
        self.Rli = np.eye(3)
        self.Pli = np.zeros(3)
        self.Rcl = np.eye(3)
        self.Pcl = np.zeros(3)
        self.Rci = np.eye(3)
        self.Pci = np.zeros(3)

        # Grid parameters
        self.grid_size = config.patch_size
        self.patch_size = config.patch_size
        self.patch_size_half = config.patch_size // 2
        self.patch_size_total = config.patch_size ** 2
        self.patch_pyrimid_level = config.patch_pyrimid_level
        self.outlier_threshold = config.outlier_threshold
        self.max_iterations = config.vio_max_iterations
        self.img_point_cov = config.img_point_cov
        self.normal_en = config.normal_en
        self.exposure_estimate_en = config.exposure_estimate_en

        self.visual_points = []  # list of VisualPoint in the map
        self.feat_map = {}       # voxel -> list of VisualPoints
        self.frame_count = 0
        self.img_cp = None
        self.img_rgb = None

        # Jacobian matrices
        self.Jdphi_dR = np.eye(3)
        self.Jdp_dR = np.zeros((3, 3))
        self.G = np.zeros((DIM_STATE, DIM_STATE))

    def initialize(self, extT, extR, cameraextrinR, cameraextrinT):
        """Initialize VIO with extrinsics and camera model."""
        self.Rli = extR.T
        self.Pli = -extR.T @ extT

        self.Rcl = np.array(cameraextrinR).reshape(3, 3) if len(cameraextrinR) == 9 else cameraextrinR
        self.Pcl = np.array(cameraextrinT)

        self.Rci = self.Rcl @ self.Rli
        self.Pci = self.Rcl @ self.Pli + self.Pcl

        self.Jdphi_dR = self.Rci.copy()
        Pic = -self.Rci.T @ self.Pci
        self.Jdp_dR = -self.Rci @ skew_sym(Pic)

        if self.config.cam_fx > 0:
            self.cam = PinholeCamera(
                self.config.cam_width, self.config.cam_height,
                self.config.cam_fx, self.config.cam_fy,
                self.config.cam_cx, self.config.cam_cy,
                self.config.cam_distortion
            )
            print(f"[VIO] Camera: {self.cam.width}x{self.cam.height}, "
                  f"fx={self.cam.fx:.2f}, fy={self.cam.fy:.2f}")

        self.grid_n_width = max(1, self.cam.width // self.grid_size) if self.cam else 1
        self.grid_n_height = max(1, self.cam.height // self.grid_size) if self.cam else 1
        self.length = self.grid_n_width * self.grid_n_height
        self.border = (self.patch_size_half + 1) * (1 << self.patch_pyrimid_level)

    def world_to_camera(self, p_w):
        """Transform world point to camera frame."""
        R_wc = self.state.rot_end
        t_wc = self.state.pos_end
        p_imu = R_wc.T @ (p_w - t_wc)
        p_cam = self.Rci @ p_imu + self.Pci
        return p_cam

    def world_to_pixel(self, p_w):
        """Project world point to pixel."""
        p_cam = self.world_to_camera(p_w)
        if p_cam[2] <= 0:
            return None, p_cam
        px = self.cam.world2cam(p_cam)
        return px, p_cam

    def get_image_patch(self, img, px, level=0):
        """Extract bilinear-interpolated image patch."""
        scale = 1 << level
        u_ref_i = int(np.floor(px[0] / scale)) * scale
        v_ref_i = int(np.floor(px[1] / scale)) * scale
        subpix_u = (px[0] - u_ref_i) / scale
        subpix_v = (px[1] - v_ref_i) / scale
        w_tl = (1.0 - subpix_u) * (1.0 - subpix_v)
        w_tr = subpix_u * (1.0 - subpix_v)
        w_bl = (1.0 - subpix_u) * subpix_v
        w_br = subpix_u * subpix_v

        h, w = img.shape[:2]
        patch = np.zeros(self.patch_size_total, dtype=np.float32)
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                row = v_ref_i - self.patch_size_half * scale + x * scale
                col = u_ref_i - self.patch_size_half * scale + y * scale
                if 0 <= row < h - scale and 0 <= col < w - scale:
                    patch[x * self.patch_size + y] = (
                        w_tl * img[row, col] +
                        w_tr * img[row, col + scale] +
                        w_bl * img[row + scale, col] +
                        w_br * img[row + scale, col + scale]
                    )
        return patch

    def processFrame(self, img, pv_list, voxel_map, img_time):
        """Process a visual frame: retrieve map points, track, update EKF, add new points."""
        if self.cam is None:
            print("[VIO] No camera model, skipping VIO")
            return

        if len(img.shape) == 3:
            self.img_rgb = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            self.img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.img_cp = self.img_rgb.copy()

        self.frame_count += 1

        # Step 1: Retrieve visual map points visible in current frame
        matched_points = []
        matched_patches_ref = []
        matched_patches_cur = []
        matched_inv_expo_ref = []

        for vp in self.visual_points:
            if not vp.is_initialized or vp.ref_patch is None:
                continue
            px, p_cam = self.world_to_pixel(vp.pos)
            if px is None:
                continue
            if not self.cam.is_in_frame(px, self.border):
                continue

            cur_patch = self.get_image_patch(gray, px, 0)
            ref_patch = vp.ref_patch

            # Photometric error check
            error = np.sum((vp.ref_inv_expo * ref_patch - self.state.inv_expo_time * cur_patch) ** 2)
            if error > self.outlier_threshold * self.patch_size_total:
                continue

            matched_points.append(vp)
            matched_patches_ref.append(ref_patch)
            matched_patches_cur.append(cur_patch)
            matched_inv_expo_ref.append(vp.ref_inv_expo)

            # Draw tracking visualization
            cv2.circle(self.img_cp, (int(px[0]), int(px[1])), 3, (0, 255, 0), -1)

        total_tracked = len(matched_points)
        print(f"[VIO] Tracked {total_tracked} visual points")

        # Step 2: EKF update with photometric residuals
        if total_tracked > 0:
            self._update_ekf_photometric(gray, matched_points, matched_patches_ref,
                                         matched_patches_cur, matched_inv_expo_ref)

        # Step 3: Generate new visual map points from LiDAR point cloud
        self._generate_visual_map_points(gray, pv_list)

    def _update_ekf_photometric(self, img, matched_points, ref_patches, cur_patches, inv_expo_refs):
        """Update EKF state using photometric residuals."""
        n = len(matched_points)
        if n == 0:
            return

        I_STATE = np.eye(DIM_STATE)
        self.G = np.zeros((DIM_STATE, DIM_STATE))

        for iteration in range(self.max_iterations):
            H_list = []
            residual_list = []
            R_inv_list = []

            for i, vp in enumerate(matched_points):
                p_cam = self.world_to_camera(vp.pos)
                if p_cam[2] <= 0:
                    continue
                px = self.cam.world2cam(p_cam)
                if not self.cam.is_in_frame(px, self.border):
                    continue

                cur_patch = self.get_image_patch(img, px, 0)
                ref_patch = ref_patches[i]
                inv_expo_ref = inv_expo_refs[i]

                # Photometric residual: ref * tau_ref - cur * tau_cur
                for idx in range(self.patch_size_total):
                    r = inv_expo_ref * ref_patch[idx] - self.state.inv_expo_time * cur_patch[idx]
                    if abs(r) > 50:  # per-pixel outlier
                        continue

                    # Image gradient
                    x_idx = idx // self.patch_size
                    y_idx = idx % self.patch_size
                    row = int(px[1]) - self.patch_size_half + x_idx
                    col = int(px[0]) - self.patch_size_half + y_idx

                    h, w = img.shape[:2]
                    if row < 1 or row >= h - 1 or col < 1 or col >= w - 1:
                        continue

                    du = 0.5 * (float(img[row, col + 1]) - float(img[row, col - 1]))
                    dv = 0.5 * (float(img[row + 1, col]) - float(img[row - 1, col]))

                    # Jacobian: dr/dstate
                    # dr/dpx = -tau_cur * [du, dv]
                    # dpx/dp_cam = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
                    z_inv = 1.0 / p_cam[2]
                    J_proj = np.array([
                        [self.cam.fx * z_inv, 0, -self.cam.fx * p_cam[0] * z_inv * z_inv],
                        [0, self.cam.fy * z_inv, -self.cam.fy * p_cam[1] * z_inv * z_inv]
                    ])

                    J_img = -self.state.inv_expo_time * np.array([du, dv])
                    J_pixel = J_img @ J_proj  # 1x3

                    # dp_cam/dR = Rci * [-p_imu]x * R_w^T  (simplified)
                    p_imu = self.state.rot_end.T @ (vp.pos - self.state.pos_end)
                    J_rot = J_pixel @ self.Rci @ skew_sym(p_imu)  # 1x3
                    J_pos = J_pixel @ (-self.Rci @ self.state.rot_end.T)  # 1x3

                    H = np.zeros(DIM_STATE)
                    H[0:3] = J_rot
                    H[3:6] = J_pos
                    if self.exposure_estimate_en:
                        H[6] = -cur_patch[idx]  # dr/d(inv_expo_time)

                    H_list.append(H)
                    residual_list.append(r)
                    R_inv_list.append(1.0 / self.img_point_cov)

            m = len(residual_list)
            if m < 6:
                break

            H_mat = np.array(H_list)  # mx19
            z_vec = np.array(residual_list)  # m
            R_inv_vec = np.array(R_inv_list)  # m

            # Weighted least squares: H^T R^{-1} H, H^T R^{-1} z
            H_T_R_inv = H_mat.T * R_inv_vec[np.newaxis, :]  # 19xm
            HTz = H_T_R_inv @ z_vec
            H_T_H = H_T_R_inv @ H_mat

            K_1 = np.linalg.inv(H_T_H + np.linalg.inv(self.state.cov))
            self.G = K_1 @ H_T_H

            vec = self.state_propagat.diff(self.state)
            solution = K_1 @ HTz + vec - self.G @ vec
            self.state.iadd(solution)

            if np.linalg.norm(solution[:3]) * 57.3 < 0.01 and np.linalg.norm(solution[3:6]) * 100 < 0.015:
                break

        self.state.cov = (I_STATE - self.G) @ self.state.cov

    def _generate_visual_map_points(self, img, pv_list):
        """Generate new visual map points from LiDAR scan projected into image."""
        if self.cam is None or len(pv_list) < 10:
            return

        added = 0
        grid_occupied = set()

        # Mark existing visual point grid cells
        for vp in self.visual_points:
            if not vp.is_initialized:
                continue
            px, p_cam = self.world_to_pixel(vp.pos)
            if px is not None and self.cam.is_in_frame(px, self.border):
                gi = int(px[1] / self.grid_size) * self.grid_n_width + int(px[0] / self.grid_size)
                grid_occupied.add(gi)

        for pv in pv_list:
            if np.all(pv.normal == 0):
                continue
            px, p_cam = self.world_to_pixel(pv.point_w)
            if px is None or not self.cam.is_in_frame(px, self.border):
                continue

            gi = int(px[1] / self.grid_size) * self.grid_n_width + int(px[0] / self.grid_size)
            if gi in grid_occupied:
                continue
            grid_occupied.add(gi)

            patch = self.get_image_patch(img, px, 0)
            vp = VisualPoint(pv.point_w, pv.normal)
            vp.ref_patch = patch
            vp.ref_px = px.copy()
            vp.ref_inv_expo = self.state.inv_expo_time
            vp.is_initialized = True
            self.visual_points.append(vp)
            added += 1

        # Limit total visual points to prevent memory growth
        max_points = 50000
        if len(self.visual_points) > max_points:
            self.visual_points = self.visual_points[-max_points:]

        print(f"[VIO] Added {added} new visual map points (total: {len(self.visual_points)})")
