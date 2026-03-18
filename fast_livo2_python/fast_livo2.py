"""FAST-LIVO2 Python Implementation - Main Pipeline.

Processes a rosbag with LiDAR, IMU, and camera data through a tightly-coupled
LiDAR-Inertial-Visual SLAM pipeline. Outputs point cloud maps, odometry, and trajectory.

Usage:
    python fast_livo2.py --bag /path/to/data.bag --config /path/to/config.yaml
"""
import os
import sys
import time
import argparse
import numpy as np
from collections import deque

from config import load_config
from states import (StatesGroup, MeasureGroup, PointWithVar, IMUData,
                    DIM_STATE, G_m_s2, LIO_FLAG, VIO_FLAG, WAIT,
                    LIVO, ONLY_LIO, ONLY_LO)
from so3_math import rot_to_euler, rot_to_quat, log_so3, skew_sym, skew_sym_batch
from imu_processing import ImuProcess
from voxel_map import VoxelMapManager, calc_body_cov, calc_body_cov_batch
from vio import VIOManager
from preprocess import preprocess_pointcloud, voxel_downsample
from rosbag_reader import read_rosbag


class FastLIVO2:
    """Main FAST-LIVO2 pipeline."""

    def __init__(self, config):
        self.config = config
        self.state = StatesGroup()
        self.state_propagat = StatesGroup()
        self.p_imu = ImuProcess()
        self.voxelmap_mgr = VoxelMapManager(config)
        self.vio_mgr = VIOManager(config)

        # Extrinsics
        self.extT = config.extrinsic_T.copy()
        self.extR = config.extrinsic_R.copy()

        # Initialize IMU processor
        self.p_imu.set_extrinsic(self.extT, self.extR)
        self.p_imu.set_gyr_cov_scale(np.array([config.gyr_cov] * 3))
        self.p_imu.set_acc_cov_scale(np.array([config.acc_cov] * 3))
        self.p_imu.set_inv_expo_cov(config.inv_expo_cov)
        self.p_imu.set_gyr_bias_cov(np.array([config.b_gyr_cov] * 3))
        self.p_imu.set_acc_bias_cov(np.array([config.b_acc_cov] * 3))
        self.p_imu.set_imu_init_frame_num(config.imu_int_frame)
        self.p_imu.lidar_type = config.lidar_type

        if not config.imu_en:
            self.p_imu.disable_imu()
        if not getattr(config, 'gravity_est_en', True):
            self.p_imu.disable_gravity_est()

        # Initialize voxelmap
        self.voxelmap_mgr.extR = self.extR.copy()
        self.voxelmap_mgr.extT = self.extT.copy()

        # VIO
        self.vio_mgr.state = self.state
        self.vio_mgr.state_propagat = self.state_propagat
        self.vio_mgr.initialize(self.extT, self.extR,
                                config.Rcl.flatten().tolist(),
                                config.Pcl.tolist())

        # SLAM mode
        self.slam_mode = LIVO if (config.img_en and config.lidar_en) else (ONLY_LIO if config.imu_en else ONLY_LO)
        mode_names = {ONLY_LO: "LO", ONLY_LIO: "LIO", LIVO: "LIVO"}
        print(f"[SLAM] Mode: {mode_names.get(self.slam_mode, 'UNKNOWN')}")

        # State tracking
        self.is_first_frame = False
        self.first_lidar_time = 0.0
        self.last_lio_update_time = 0.0
        self.gravity_align_finished = False
        self.scan_count = 0
        self.frame_id = 0

        # Accumulators
        self.world_cloud = []       # accumulated map points (x,y,z,intensity)
        self.trajectory = []        # list of (timestamp, pos, quat)
        self.odometry_log = []      # detailed odometry log
        self.vio_images = []        # saved VIO tracking images

    def run(self, bag_path, camera_yaml_path=None):
        """Process a rosbag file end-to-end."""
        cfg = self.config

        # Load camera config if needed
        if camera_yaml_path and self.slam_mode == LIVO:
            from config import _load_camera_config
            _load_camera_config(camera_yaml_path, cfg)
            self.vio_mgr.initialize(self.extT, self.extR,
                                    cfg.Rcl.flatten().tolist(),
                                    cfg.Pcl.tolist())

        # Read rosbag
        lidar_msgs, imu_msgs, img_msgs = read_rosbag(
            bag_path,
            cfg.lid_topic, cfg.imu_topic,
            cfg.img_topic if cfg.img_en else None,
            lidar_time_offset=0.0,
            imu_time_offset=cfg.imu_time_offset,
            img_time_offset=cfg.img_time_offset
        )

        if not lidar_msgs:
            print("[ERROR] No LiDAR messages found in bag!")
            return

        # Build synchronized measurement groups
        imu_idx = 0
        img_idx = 0

        total_scans = len(lidar_msgs)
        print(f"\n[Pipeline] Processing {total_scans} LiDAR scans...")
        t_start = time.time()

        for scan_idx, (lid_time, points, times, *_rest) in enumerate(lidar_msgs):
            # Preprocess LiDAR
            points, times = preprocess_pointcloud(
                points, times,
                blind=cfg.blind,
                point_filter_num=cfg.point_filter_num
            )

            if len(points) == 0:
                continue

            # First frame initialization
            if not self.is_first_frame:
                self.first_lidar_time = lid_time
                self.p_imu.first_lidar_time = lid_time
                self.last_lio_update_time = lid_time
                self.is_first_frame = True
                print(f"[Pipeline] First LiDAR frame at t={lid_time:.6f}")

            # Collect IMU data up to this LiDAR scan
            meas = MeasureGroup()
            meas.lio_time = lid_time + times[-1] if len(times) > 0 else lid_time
            meas.vio_time = lid_time

            while imu_idx < len(imu_msgs) and imu_msgs[imu_idx].timestamp <= meas.lio_time:
                meas.imu.append(imu_msgs[imu_idx])
                imu_idx += 1

            # Find closest image if VIO is enabled
            cur_img = None
            if self.slam_mode == LIVO and img_msgs:
                while img_idx < len(img_msgs) and img_msgs[img_idx][0] < lid_time - 0.05:
                    img_idx += 1
                if img_idx < len(img_msgs) and abs(img_msgs[img_idx][0] - lid_time) < 0.1:
                    cur_img = img_msgs[img_idx][1]

            # --- IMU Processing ---
            feats_undistort, self.last_lio_update_time = self.p_imu.Process2(
                meas, LIO_FLAG, points, times,
                self.state, self.last_lio_update_time
            )

            # Gravity alignment (if enabled)
            if cfg.gravity_align_en and not self.p_imu.imu_need_init and not self.gravity_align_finished:
                self._gravity_alignment()

            self.state_propagat = self.state.copy()
            self.voxelmap_mgr.state = self.state.copy()

            # Skip if IMU still initializing
            if self.p_imu.imu_need_init:
                if scan_idx % 10 == 0:
                    print(f"[Pipeline] IMU initializing... scan {scan_idx}/{total_scans}")
                continue

            # --- LIO: LiDAR-Inertial Odometry ---
            self._handle_LIO(feats_undistort, meas, points)

            # --- VIO: Visual-Inertial Odometry ---
            if self.slam_mode == LIVO and cur_img is not None and self.vio_mgr.cam is not None:
                self._handle_VIO(cur_img, meas)

            # --- Save trajectory and accumulate map ---
            self._save_state(meas.lio_time)

            self.scan_count += 1
            if self.scan_count % 50 == 0:
                elapsed = time.time() - t_start
                rate = self.scan_count / elapsed if elapsed > 0 else 0
                print(f"[Pipeline] Scan {self.scan_count}/{total_scans}, "
                      f"{rate:.1f} scans/s, "
                      f"map points: {len(self.world_cloud)}, "
                      f"pos: [{self.state.pos_end[0]:.2f}, {self.state.pos_end[1]:.2f}, {self.state.pos_end[2]:.2f}]")

        elapsed = time.time() - t_start
        print(f"\n[Pipeline] Done! Processed {self.scan_count} scans in {elapsed:.1f}s "
              f"({self.scan_count/elapsed:.1f} scans/s)")
        print(f"[Pipeline] Total map points: {len(self.world_cloud)}")
        print(f"[Pipeline] Trajectory points: {len(self.trajectory)}")

    def _gravity_alignment(self):
        """Align world frame Z-axis with gravity."""
        ez = np.array([0, 0, -1.0])
        gz = self.state.gravity / np.linalg.norm(self.state.gravity)

        # Rotation from gravity direction to -Z
        v = np.cross(gz, ez)
        s = np.linalg.norm(v)
        c = np.dot(gz, ez)
        if s < 1e-8:
            G_R_I0 = np.eye(3) if c > 0 else -np.eye(3)
        else:
            vx = skew_sym(v)
            G_R_I0 = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

        self.state.pos_end = G_R_I0 @ self.state.pos_end
        self.state.rot_end = G_R_I0 @ self.state.rot_end
        self.state.vel_end = G_R_I0 @ self.state.vel_end
        self.state.gravity = G_R_I0 @ self.state.gravity
        self.gravity_align_finished = True
        print("[Pipeline] Gravity alignment done")

    def _handle_LIO(self, feats_undistort, meas, raw_points):
        """Handle LIO: downsample, state estimation, voxel map update (vectorized)."""
        if len(feats_undistort) == 0:
            return

        # Downsample
        feats_down_body = voxel_downsample(feats_undistort, self.config.filter_size_surf)
        if len(feats_down_body) == 0:
            return

        feats_down_size = len(feats_down_body)

        # Vectorized transform to world frame
        RE = self.state.rot_end @ self.extR
        Te = self.state.rot_end @ self.extT + self.state.pos_end
        feats_down_world = feats_down_body @ RE.T + Te[None, :]

        # Set data in voxelmap manager
        self.voxelmap_mgr.feats_down_body = feats_down_body
        self.voxelmap_mgr.feats_down_world = feats_down_world
        self.voxelmap_mgr.feats_down_size = feats_down_size
        self.voxelmap_mgr.state = self.state.copy()

        if self.scan_count == 0:
            # First scan: build voxel map
            self.voxelmap_mgr.BuildVoxelMap()
            print(f"[LIO] Initial voxel map built with {len(self.voxelmap_mgr.voxel_map)} voxels")
        else:
            # State estimation using iterated EKF
            self.voxelmap_mgr.StateEstimation(self.state_propagat)

            # Update state from voxelmap manager
            self.state = self.voxelmap_mgr.state.copy()
            self.vio_mgr.state = self.state

            # Vectorized re-transform to updated world frame
            RE = self.state.rot_end @ self.extR
            Te = self.state.rot_end @ self.extT + self.state.pos_end
            feats_down_world = feats_down_body @ RE.T + Te[None, :]

            # Vectorized variance computation for map update
            body_covs = calc_body_cov_batch(feats_down_body, self.config.dept_err, self.config.beam_err)
            RcovR = np.einsum('ij,njk,lk->nil', RE, body_covs, RE)
            pts_lidar = feats_down_body @ self.extR.T + self.extT[None, :]
            cross_mats = skew_sym_batch(pts_lidar)
            rot_var = self.state.cov[0:3, 0:3]
            t_var = self.state.cov[3:6, 3:6]
            CrC = np.einsum('nij,jk,nlk->nil', cross_mats, rot_var, cross_mats)
            covs_world = RcovR + CrC + t_var[None, :, :]

            input_points = []
            for i in range(feats_down_size):
                pv = PointWithVar()
                pv.point_w = feats_down_world[i]
                pv.point_b = feats_down_body[i]
                pv.var = covs_world[i]
                pv.body_var = body_covs[i]
                input_points.append(pv)

            self.voxelmap_mgr.UpdateVoxelMap(input_points)

            # Map sliding if enabled
            if self.config.map_sliding_en:
                self.voxelmap_mgr.mapSliding()

        # Accumulate world cloud (vectorized append)
        self.world_cloud.extend(feats_down_world.tolist())

    def _handle_VIO(self, img, meas):
        """Handle VIO: visual tracking and EKF update."""
        if self.vio_mgr.cam is None:
            return

        # Build pv_list from recent LIO points for VIO
        pv_list = []
        if self.voxelmap_mgr.feats_down_body is not None:
            for i in range(len(self.voxelmap_mgr.feats_down_body)):
                pv = PointWithVar()
                pv.point_w = self.voxelmap_mgr.feats_down_world[i].copy()
                pv.point_b = self.voxelmap_mgr.feats_down_body[i].copy()
                pv.normal = np.zeros(3)  # Will be filled from voxel map planes
                # Try to get normal from voxel map
                voxel_size = self.config.voxel_size
                loc = [0, 0, 0]
                for j in range(3):
                    loc[j] = pv.point_w[j] / voxel_size
                    if loc[j] < 0:
                        loc[j] -= 1.0
                key = (int(loc[0]), int(loc[1]), int(loc[2]))
                if key in self.voxelmap_mgr.voxel_map:
                    octo = self.voxelmap_mgr.voxel_map[key]
                    if octo.plane_ptr.is_plane:
                        pv.normal = octo.plane_ptr.normal.copy()
                pv.var = np.eye(3) * 0.01
                pv_list.append(pv)

        self.vio_mgr.state = self.state
        self.vio_mgr.state_propagat = self.state_propagat
        self.vio_mgr.processFrame(img, pv_list, self.voxelmap_mgr.voxel_map, meas.vio_time)
        self.state = self.vio_mgr.state

        # Save tracking image
        if self.vio_mgr.img_cp is not None:
            self.vio_images.append((meas.vio_time, self.vio_mgr.img_cp))

    def _save_state(self, timestamp):
        """Record current state for output."""
        pos = self.state.pos_end.copy()
        quat = rot_to_quat(self.state.rot_end)  # x,y,z,w
        euler = rot_to_euler(self.state.rot_end)
        vel = self.state.vel_end.copy()

        self.trajectory.append((timestamp, pos, quat))
        self.odometry_log.append({
            'time': timestamp,
            'pos': pos, 'quat': quat, 'euler': euler,
            'vel': vel,
            'bias_g': self.state.bias_g.copy(),
            'bias_a': self.state.bias_a.copy(),
            'gravity': self.state.gravity.copy(),
            'inv_expo': self.state.inv_expo_time
        })

    def save_results(self, output_dir):
        """Save all results to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save point cloud as PCD
        pcd_path = os.path.join(output_dir, "map.pcd")
        self._save_pcd(pcd_path, self.world_cloud)
        print(f"[Output] Saved point cloud: {pcd_path} ({len(self.world_cloud)} points)")

        # 2. Save point cloud as PLY (more universally supported)
        ply_path = os.path.join(output_dir, "map.ply")
        self._save_ply(ply_path, self.world_cloud)
        print(f"[Output] Saved PLY: {ply_path}")

        # 3. Save trajectory (TUM format: timestamp tx ty tz qx qy qz qw)
        traj_path = os.path.join(output_dir, "trajectory.txt")
        with open(traj_path, 'w') as f:
            for ts, pos, quat in self.trajectory:
                f.write(f"{ts:.9f} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f} "
                        f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}\n")
        print(f"[Output] Saved trajectory: {traj_path} ({len(self.trajectory)} poses)")

        # 4. Save detailed odometry
        odom_path = os.path.join(output_dir, "odometry.txt")
        with open(odom_path, 'w') as f:
            f.write("# timestamp tx ty tz vx vy vz roll pitch yaw bgx bgy bgz bax bay baz gx gy gz inv_expo\n")
            for o in self.odometry_log:
                e = o['euler']
                f.write(f"{o['time']:.9f} "
                        f"{o['pos'][0]:.9f} {o['pos'][1]:.9f} {o['pos'][2]:.9f} "
                        f"{o['vel'][0]:.6f} {o['vel'][1]:.6f} {o['vel'][2]:.6f} "
                        f"{e[0]:.9f} {e[1]:.9f} {e[2]:.9f} "
                        f"{o['bias_g'][0]:.9f} {o['bias_g'][1]:.9f} {o['bias_g'][2]:.9f} "
                        f"{o['bias_a'][0]:.9f} {o['bias_a'][1]:.9f} {o['bias_a'][2]:.9f} "
                        f"{o['gravity'][0]:.6f} {o['gravity'][1]:.6f} {o['gravity'][2]:.6f} "
                        f"{o['inv_expo']:.9f}\n")
        print(f"[Output] Saved odometry: {odom_path}")

        # 5. Save VIO tracking images
        if self.vio_images:
            import cv2
            img_dir = os.path.join(output_dir, "vio_tracking")
            os.makedirs(img_dir, exist_ok=True)
            for i, (ts, img) in enumerate(self.vio_images):
                cv2.imwrite(os.path.join(img_dir, f"{i:06d}_{ts:.6f}.png"), img)
            print(f"[Output] Saved {len(self.vio_images)} VIO tracking images")

        print(f"\n[Output] All results saved to: {output_dir}")

    def _save_pcd(self, path, points):
        """Save point cloud in ASCII PCD format."""
        pts = np.array(points)
        n = len(pts)
        with open(path, 'w') as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write(f"WIDTH {n}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {n}\n")
            f.write("DATA ascii\n")
            # Write all points at once using numpy
            np.savetxt(f, pts, fmt='%.6f')

    def _save_ply(self, path, points):
        """Save point cloud in PLY format."""
        pts = np.array(points)
        n = len(pts)
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            np.savetxt(f, pts, fmt='%.6f')


def main():
    parser = argparse.ArgumentParser(description="FAST-LIVO2 Python - LiDAR-Inertial-Visual SLAM")
    parser.add_argument("--bag", required=True, help="Path to rosbag file (.bag or .db3)")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g., avia.yaml)")
    parser.add_argument("--camera-config", default=None, help="Path to camera YAML config")
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, args.camera_config)

    print("=" * 60)
    print("FAST-LIVO2 Python Implementation")
    print("=" * 60)
    print(f"Bag:    {args.bag}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Topics: LiDAR={config.lid_topic}, IMU={config.imu_topic}, Img={config.img_topic}")
    print(f"IMU enabled: {config.imu_en}, Image enabled: {config.img_en}")
    print(f"Voxel size: {config.voxel_size}, Filter: {config.filter_size_surf}")
    print("=" * 60)

    # Create and run pipeline
    pipeline = FastLIVO2(config)
    pipeline.run(args.bag, args.camera_config)

    # Save results
    pipeline.save_results(args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
