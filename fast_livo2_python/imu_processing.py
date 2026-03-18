"""IMU processing: initialization, forward propagation, and point cloud undistortion."""
import numpy as np
from so3_math import exp_so3, exp_so3_dt, skew_sym
from states import (StatesGroup, IMUData, Pose6D, MeasureGroup,
                    DIM_STATE, G_m_s2, LIO_FLAG, VIO_FLAG, LO_FLAG, L515)


class ImuProcess:
    def __init__(self):
        self.imu_en = True
        self.imu_need_init = True
        self.b_first_frame = True
        self.init_iter_num = 1
        self.MAX_INI_COUNT = 30

        self.cov_acc = np.array([0.1, 0.1, 0.1])
        self.cov_gyr = np.array([0.1, 0.1, 0.1])
        self.cov_bias_gyr = np.array([0.1, 0.1, 0.1])
        self.cov_bias_acc = np.array([0.1, 0.1, 0.1])
        self.cov_inv_expo = 0.2

        self.mean_acc = np.array([0.0, 0.0, -1.0])
        self.mean_gyr = np.zeros(3)
        self.IMU_mean_acc_norm = 1.0

        self.angvel_last = np.zeros(3)
        self.acc_s_last = np.zeros(3)

        self.Lid_offset_to_IMU = np.zeros(3)
        self.Lid_rot_to_IMU = np.eye(3)

        self.last_imu = IMUData()
        self.IMUpose = []
        self.pcl_wait_proc = None  # Nx4 array (x,y,z,curvature)
        self.last_prop_end_time = 0.0
        self.time_last_scan = 0.0
        self.first_lidar_time = 0.0

        self.gravity_est_en = True
        self.ba_bg_est_en = True
        self.exposure_estimate_en = True
        self.imu_time_init = False
        self.lidar_type = 1

    def set_extrinsic(self, t, R):
        self.Lid_offset_to_IMU = t.copy()
        self.Lid_rot_to_IMU = R.copy()

    def set_gyr_cov_scale(self, s):
        self.cov_gyr = s.copy()

    def set_acc_cov_scale(self, s):
        self.cov_acc = s.copy()

    def set_gyr_bias_cov(self, b):
        self.cov_bias_gyr = b.copy()

    def set_acc_bias_cov(self, b):
        self.cov_bias_acc = b.copy()

    def set_inv_expo_cov(self, c):
        self.cov_inv_expo = c

    def set_imu_init_frame_num(self, n):
        self.MAX_INI_COUNT = n

    def disable_imu(self):
        self.imu_en = False
        self.imu_need_init = False

    def disable_gravity_est(self):
        self.gravity_est_en = False

    def disable_bias_est(self):
        self.ba_bg_est_en = False

    def disable_exposure_est(self):
        self.exposure_estimate_en = False

    def IMU_init(self, meas, state, N):
        """Initialize IMU: estimate gravity direction from static measurements."""
        if self.b_first_frame:
            N = 1
            self.b_first_frame = False
            if len(meas.imu) > 0:
                self.mean_acc = meas.imu[0].acc.copy()
                self.mean_gyr = meas.imu[0].gyr.copy()

        for imu in meas.imu:
            self.mean_acc += (imu.acc - self.mean_acc) / N
            self.mean_gyr += (imu.gyr - self.mean_gyr) / N
            N += 1

        self.IMU_mean_acc_norm = np.linalg.norm(self.mean_acc)
        state.gravity = -self.mean_acc / self.IMU_mean_acc_norm * G_m_s2
        state.rot_end = np.eye(3)
        state.bias_g = np.zeros(3)
        self.last_imu = meas.imu[-1] if meas.imu else self.last_imu
        return N

    def Forward_without_imu(self, pcl_points, pcl_times, state, pcl_beg_time):
        """Forward propagation without IMU (constant velocity model).
        pcl_points: Nx3 array, pcl_times: N array (offset times in seconds).
        Returns undistorted points Nx3."""
        # Sort by time
        idx = np.argsort(pcl_times)
        pcl_points = pcl_points[idx]
        pcl_times = pcl_times[idx]
        pcl_end_offset = pcl_times[-1]

        if self.b_first_frame:
            dt = 0.1
            self.b_first_frame = False
        else:
            dt = pcl_beg_time - self.time_last_scan
        self.time_last_scan = pcl_beg_time

        Exp_f = exp_so3_dt(state.bias_g, dt)

        # Covariance propagation
        F_x = np.eye(DIM_STATE)
        cov_w = np.zeros((DIM_STATE, DIM_STATE))
        F_x[0:3, 0:3] = exp_so3_dt(state.bias_g, -dt)
        F_x[0:3, 10:13] = np.eye(3) * dt
        F_x[3:6, 7:10] = np.eye(3) * dt
        cov_w[10:13, 10:13] = np.diag(self.cov_gyr) * dt * dt
        cov_w[7:10, 7:10] = np.diag(self.cov_acc) * dt * dt
        state.cov = F_x @ state.cov @ F_x.T + cov_w

        state.rot_end = state.rot_end @ Exp_f
        state.pos_end = state.pos_end + state.vel_end * dt

        # Undistort points
        if self.lidar_type != L515:
            for i in range(len(pcl_points) - 1, -1, -1):
                dt_j = pcl_end_offset - pcl_times[i]
                R_jk = exp_so3_dt(state.bias_g, -dt_j)
                p_jk = -state.rot_end.T @ state.vel_end * dt_j
                pcl_points[i] = R_jk @ pcl_points[i] + p_jk

        return pcl_points, pcl_beg_time + pcl_end_offset

    def UndistortPcl(self, meas, lio_vio_flg, pcl_proc_cur, pcl_proc_times,
                     state, last_lio_update_time):
        """Undistort point cloud using IMU forward propagation and backward compensation.

        Args:
            meas: MeasureGroup with IMU data
            lio_vio_flg: LIO_FLAG or VIO_FLAG
            pcl_proc_cur: Nx3 points array
            pcl_proc_times: N array of offset times (seconds)
            state: StatesGroup (modified in place)
            last_lio_update_time: previous update time

        Returns:
            undistorted_pcl (Nx3), new_lio_update_time
        """
        # Build IMU sequence with last frame's tail IMU prepended
        v_imu = [self.last_imu] + list(meas.imu)
        if len(v_imu) < 2:
            return pcl_proc_cur, last_lio_update_time

        prop_beg_time = self.last_prop_end_time
        prop_end_time = meas.lio_time if lio_vio_flg == LIO_FLAG else meas.vio_time

        # Initialize tau
        if not self.imu_time_init:
            tau = 1.0
            self.imu_time_init = True
        else:
            tau = state.inv_expo_time

        # Prepare pcl for undistortion
        if lio_vio_flg == LIO_FLAG:
            self.pcl_wait_proc = pcl_proc_cur.copy()
            self.pcl_wait_proc_times = pcl_proc_times.copy()
            self.IMUpose = []
            self.IMUpose.append(Pose6D(
                offset_time=0.0,
                acc=self.acc_s_last.copy(),
                gyr=self.angvel_last.copy(),
                vel=state.vel_end.copy(),
                pos=state.pos_end.copy(),
                rot=state.rot_end.copy()
            ))

        # Forward propagation through IMU measurements
        acc_imu = self.acc_s_last.copy()
        angvel_avr = self.angvel_last.copy()
        vel_imu = state.vel_end.copy()
        pos_imu = state.pos_end.copy()
        R_imu = state.rot_end.copy()
        dt_all = 0.0

        if lio_vio_flg in (LIO_FLAG, VIO_FLAG):
            for i in range(len(v_imu) - 1):
                head = v_imu[i]
                tail = v_imu[i + 1]

                if tail.timestamp < prop_beg_time:
                    continue

                angvel_avr = 0.5 * (head.gyr + tail.gyr)
                acc_avr = 0.5 * (head.acc + tail.acc)

                angvel_avr = angvel_avr - state.bias_g
                acc_avr = acc_avr * G_m_s2 / np.linalg.norm(self.mean_acc) - state.bias_a

                if head.timestamp < prop_beg_time:
                    dt = tail.timestamp - self.last_prop_end_time
                    offs_t = tail.timestamp - prop_beg_time
                elif i != len(v_imu) - 2:
                    dt = tail.timestamp - head.timestamp
                    offs_t = tail.timestamp - prop_beg_time
                else:
                    dt = prop_end_time - head.timestamp
                    offs_t = prop_end_time - prop_beg_time

                if dt <= 0:
                    continue

                dt_all += dt

                # Covariance propagation
                acc_avr_skew = skew_sym(acc_avr)
                Exp_f = exp_so3_dt(angvel_avr, dt)

                F_x = np.eye(DIM_STATE)
                cov_w = np.zeros((DIM_STATE, DIM_STATE))

                F_x[0:3, 0:3] = exp_so3_dt(angvel_avr, -dt)
                if self.ba_bg_est_en:
                    F_x[0:3, 10:13] = -np.eye(3) * dt
                F_x[3:6, 7:10] = np.eye(3) * dt
                F_x[7:10, 0:3] = -R_imu @ acc_avr_skew * dt
                if self.ba_bg_est_en:
                    F_x[7:10, 13:16] = -R_imu * dt
                if self.gravity_est_en:
                    F_x[7:10, 16:19] = np.eye(3) * dt

                if self.exposure_estimate_en:
                    cov_w[6, 6] = self.cov_inv_expo * dt * dt
                cov_w[0:3, 0:3] = np.diag(self.cov_gyr) * dt * dt
                cov_w[7:10, 7:10] = R_imu @ np.diag(self.cov_acc) @ R_imu.T * dt * dt
                cov_w[10:13, 10:13] = np.diag(self.cov_bias_gyr) * dt * dt
                cov_w[13:16, 13:16] = np.diag(self.cov_bias_acc) * dt * dt

                state.cov = F_x @ state.cov @ F_x.T + cov_w

                # Propagate IMU attitude
                R_imu = R_imu @ Exp_f

                # Specific acceleration (global frame)
                acc_imu = R_imu @ acc_avr + state.gravity

                # Propagate IMU position
                pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt

                # Propagate velocity
                vel_imu = vel_imu + acc_imu * dt

                self.angvel_last = angvel_avr.copy()
                self.acc_s_last = acc_imu.copy()

                self.IMUpose.append(Pose6D(
                    offset_time=offs_t,
                    acc=acc_imu.copy(),
                    gyr=angvel_avr.copy(),
                    vel=vel_imu.copy(),
                    pos=pos_imu.copy(),
                    rot=R_imu.copy()
                ))

        # Update state
        state.vel_end = vel_imu
        state.rot_end = R_imu
        state.pos_end = pos_imu
        state.inv_expo_time = tau

        new_lio_update_time = prop_end_time
        self.last_imu = v_imu[-1]
        self.last_prop_end_time = prop_end_time

        # Backward undistortion for LIO
        if self.pcl_wait_proc is not None and len(self.pcl_wait_proc) > 0 and lio_vio_flg == LIO_FLAG:
            pcl_out = self.pcl_wait_proc.copy()
            times_out = self.pcl_wait_proc_times.copy()
            n_pts = len(pcl_out)

            extR_Ri = self.Lid_rot_to_IMU.T @ state.rot_end.T
            exrR_extT = self.Lid_rot_to_IMU.T @ self.Lid_offset_to_IMU
            n_poses = len(self.IMUpose)

            if n_poses > 1:
                # Vectorized backward undistortion:
                # 1. Sort points by time to enable efficient bracket finding
                sort_idx = np.argsort(times_out)
                times_sorted = times_out[sort_idx]
                pts_sorted = pcl_out[sort_idx]

                # 2. Get IMU pose times
                pose_times = np.array([p.offset_time for p in self.IMUpose])

                # 3. Find bracket index for each point
                # searchsorted gives index where time would be inserted
                bracket_idx = np.searchsorted(pose_times, times_sorted, side='right') - 1
                bracket_idx = np.clip(bracket_idx, 1, n_poses - 1)  # at least index 1

                # 4. Group points by bracket and process each group
                # Transform to LiDAR frame first: L @ P + offset
                pts_imu = pts_sorted @ self.Lid_rot_to_IMU.T + self.Lid_offset_to_IMU[None, :]

                unique_brackets = np.unique(bracket_idx)
                result = np.empty_like(pts_sorted)

                for bi in unique_brackets:
                    mask = bracket_idx == bi
                    head = self.IMUpose[bi - 1]
                    dt_pts = times_sorted[mask] - head.offset_time  # group of dt values

                    # For each unique dt, we need exp_so3_dt(head.gyr, dt)
                    # Group by similar dt to batch the Rodrigues formula
                    group_pts = pts_imu[mask]
                    group_dt = dt_pts
                    n_group = len(group_dt)

                    # Compute R_i and T_ei for each point in group
                    ang = head.gyr  # 3-vector
                    ang_norm = np.linalg.norm(ang)

                    if ang_norm > 1e-7:
                        r = ang / ang_norm
                        K = skew_sym(r)
                        KK = K @ K
                        thetas = ang_norm * group_dt  # n_group
                        sin_t = np.sin(thetas)
                        cos_t = np.cos(thetas)
                        # R_i = head.rot @ (I + sin(theta)*K + (1-cos(theta))*K@K) for each theta
                        # Batch: exp_mats[j] = I + sin_t[j]*K + (1-cos_t[j])*KK
                        # Then R_i[j] = head.rot @ exp_mats[j]
                        # Applied to pts: R_i @ p = head.rot @ (p + sin_t*K@p + (1-cos_t)*KK@p)
                        Kp = group_pts @ K.T       # n_group x 3
                        KKp = group_pts @ KK.T     # n_group x 3
                        rotated = group_pts + sin_t[:, None] * Kp + (1.0 - cos_t)[:, None] * KKp
                        Ri_pts = rotated @ head.rot.T  # n_group x 3
                    else:
                        Ri_pts = group_pts @ head.rot.T

                    # T_ei = head.pos + head.vel * dt + 0.5 * head.acc * dt^2 - state.pos_end
                    T_ei = (head.pos[None, :] + head.vel[None, :] * group_dt[:, None] +
                            0.5 * head.acc[None, :] * (group_dt * group_dt)[:, None] -
                            state.pos_end[None, :])

                    # P_compensate = extR_Ri @ (Ri_pts + T_ei) - exrR_extT
                    combined = Ri_pts + T_ei
                    compensated = combined @ extR_Ri.T - exrR_extT[None, :]
                    result[mask] = compensated

                # Unsort back to original order
                pcl_out[sort_idx] = result
            else:
                # Only 1 pose, no undistortion needed
                pass

            self.pcl_wait_proc = None
            self.IMUpose = []
            return pcl_out, new_lio_update_time

        return pcl_proc_cur, new_lio_update_time

    def Process2(self, meas, lio_vio_flg, pcl_proc_cur, pcl_proc_times,
                 state, last_lio_update_time):
        """Main IMU processing entry point.
        Returns: (undistorted_pcl, new_lio_update_time)"""
        if not self.imu_en:
            pcl_beg_time = last_lio_update_time if last_lio_update_time > 0 else meas.lio_time
            return self.Forward_without_imu(pcl_proc_cur, pcl_proc_times, state, pcl_beg_time)

        if self.imu_need_init:
            if not meas.imu:
                return pcl_proc_cur, last_lio_update_time

            self.init_iter_num = self.IMU_init(meas, state, self.init_iter_num)
            self.imu_need_init = True
            self.last_imu = meas.imu[-1]

            if self.init_iter_num > self.MAX_INI_COUNT:
                self.imu_need_init = False
                print(f"[IMU] Init done. Gravity: {state.gravity}, mean_acc_norm: {self.IMU_mean_acc_norm:.4f}")

            return pcl_proc_cur, last_lio_update_time

        return self.UndistortPcl(meas, lio_vio_flg, pcl_proc_cur, pcl_proc_times,
                                 state, last_lio_update_time)
