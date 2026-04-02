"""Colorize a point cloud using camera images and trajectory poses.

Projects world-frame 3D points into each camera image to sample RGB colors.
Generates a self-contained HTML viewer with Three.js, slice tools, and
multiple color modes (RGB, height, time, intensity, distance).

Usage:
    python colorize_cloud.py --pcd outputs/output_scan/map.pcd \
        --bag Bags/scan.bag \
        --trajectory outputs/output_scan/trajectory.txt \
        --config config/avia.yaml \
        --camera config/camera_pinhole.yaml \
        --output outputs/output_scan/colored_viewer.html
"""
import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import time
import zlib

import cv2
import numpy as np
import yaml


# ─── Point cloud I/O ───────────────────────────────────────────────

def read_pcd_ascii(path):
    """Read ASCII PCD file -> Nx3 float32."""
    with open(path, 'r') as f:
        lines = f.readlines()
    data_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('DATA'):
            data_idx = i + 1
            break
    pts = []
    for line in lines[data_idx:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(pts, dtype=np.float32)


def _save_pcd_simple(path, pts):
    """Save Nx3 point cloud as ASCII PCD file."""
    pts = np.asarray(pts, dtype=np.float32)
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
        np.savetxt(f, pts, fmt='%.6f')


def voxel_downsample(pts, voxel_size, max_points=None, min_count=1):
    """Downsample points to one point per voxel cube.

    Returns indices into the original array.
    min_count: only keep voxels with at least this many points (filters noise).
    max_points: if result still exceeds this, randomly subsample.
    """
    keys = (pts / voxel_size).astype(np.int64)
    # Pack keys into single int64 for fast counting
    unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)

    if min_count > 1:
        # Find which voxels pass the threshold
        dense_mask = counts >= min_count
        # Map back to point indices: keep first point per qualifying voxel
        point_mask = dense_mask[inverse]
        # Get unique indices from qualifying points only
        qualifying_keys = keys[point_mask]
        _, first_idx = np.unique(qualifying_keys, axis=0, return_index=True)
        # Map back to original array indices
        qualifying_indices = np.where(point_mask)[0]
        unique_idx = qualifying_indices[first_idx]
    else:
        _, unique_idx = np.unique(keys, axis=0, return_index=True)

    unique_idx.sort()
    if max_points and len(unique_idx) > max_points:
        rng = np.random.default_rng(42)
        unique_idx = rng.choice(unique_idx, max_points, replace=False)
        unique_idx.sort()
    return unique_idx


# ─── Trajectory I/O ────────────────────────────────────────────────

def read_trajectory(path):
    """Read TUM-format trajectory -> list of (timestamp, R_3x3, t_3)."""
    poses = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = quat_to_rot(qx, qy, qz, qw)
            t = np.array([tx, ty, tz])
            poses.append((ts, R, t))
    return poses


def quat_to_rot(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])


# ─── Image extraction from rosbag ──────────────────────────────────

def extract_images(bag_path, img_topic, img_time_offset=0.0):
    """Extract all images from rosbag -> list of (timestamp, bgr_image)."""
    from rosbags.rosbag1 import Reader as Reader1
    import pathlib

    images = []
    with Reader1(pathlib.Path(str(bag_path))) as reader:
        conns = [c for c in reader.connections if c.topic == img_topic]
        if not conns:
            print(f"[Colorize] No image topic '{img_topic}' found")
            return images

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            data = bytes(rawdata)
            offset = 0
            offset += 4  # seq
            stamp_secs = struct.unpack_from('<I', data, offset)[0]; offset += 4
            stamp_nsecs = struct.unpack_from('<I', data, offset)[0]; offset += 4
            frame_id_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
            offset += frame_id_len

            msgtype = conn.msgtype
            if 'CompressedImage' in msgtype:
                fmt_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
                offset += fmt_len
                data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
                img_bytes = data[offset:offset+data_len]
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            else:
                h = struct.unpack_from('<I', data, offset)[0]; offset += 4
                w = struct.unpack_from('<I', data, offset)[0]; offset += 4
                enc_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
                encoding = data[offset:offset+enc_len].decode('utf-8', errors='ignore'); offset += enc_len
                offset += 1  # is_bigendian
                offset += 4  # step
                data_len_img = struct.unpack_from('<I', data, offset)[0]; offset += 4
                img_data = data[offset:offset+data_len_img]

                if encoding in ('mono8', '8UC1'):
                    img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif encoding in ('bgr8', '8UC3'):
                    img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w, 3)
                elif encoding == 'rgb8':
                    img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w, 3)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    bpp = data_len_img // (h * w)
                    if bpp == 1:
                        img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif bpp == 3:
                        img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w, 3)
                    else:
                        continue

            ts = stamp_secs + stamp_nsecs * 1e-9 + img_time_offset
            if img is not None:
                images.append((ts, img))

    images.sort(key=lambda x: x[0])
    print(f"[Colorize] Extracted {len(images)} images")
    return images


# ─── Projection and colorization ───────────────────────────────────

def colorize_points(pts_world, poses, images,
                    cam_K, cam_dist, cam_w, cam_h,
                    Rcl, Pcl, ext_R, ext_T):
    """Project world points into each camera image to assign colors.

    Transform chain (world -> camera):
        p_imu = R_wi^T * (p_world - t_wi)
        p_lidar = ext_R^T * (p_imu - ext_T)
        p_cam = Rcl * p_lidar + Pcl   (Rcl = R_{cam<-lid})
    """
    n_pts = len(pts_world)
    colors = np.full((n_pts, 3), 128, dtype=np.uint8)
    colored = np.zeros(n_pts, dtype=bool)
    best_dist = np.full(n_pts, np.inf)

    R_lid_from_imu = ext_R.T
    R_cam_from_lid = Rcl
    R_cam_from_imu = R_cam_from_lid @ R_lid_from_imu
    t_cam_from_imu = R_cam_from_lid @ (R_lid_from_imu @ (-ext_T)) + Pcl

    img_cx, img_cy = cam_w / 2.0, cam_h / 2.0
    pose_times = np.array([p[0] for p in poses])

    t0 = time.time()
    for i, (img_ts, img) in enumerate(images):
        idx = np.argmin(np.abs(pose_times - img_ts))
        _, R_wi, t_wi = poses[idx]

        img_undist = cv2.undistort(img, cam_K, cam_dist)

        R_cam_from_world = R_cam_from_imu @ R_wi.T
        t_cam_from_world = R_cam_from_imu @ (R_wi.T @ (-t_wi)) + t_cam_from_imu

        pts_cam = (pts_world @ R_cam_from_world.T) + t_cam_from_world[None, :]

        valid_z = pts_cam[:, 2] > 0.1

        z_inv = np.zeros(n_pts)
        z_inv[valid_z] = 1.0 / pts_cam[valid_z, 2]
        u = cam_K[0, 0] * pts_cam[:, 0] * z_inv + cam_K[0, 2]
        v = cam_K[1, 1] * pts_cam[:, 1] * z_inv + cam_K[1, 2]

        border = 5
        in_frame = (valid_z &
                    (u >= border) & (u < cam_w - border) &
                    (v >= border) & (v < cam_h - border))

        if not np.any(in_frame):
            continue

        dist_to_center = np.sqrt((u - img_cx)**2 + (v - img_cy)**2)
        update_mask = in_frame & ((~colored) | (dist_to_center < best_dist))

        if not np.any(update_mask):
            continue

        u_int = np.clip(u[update_mask].astype(np.int32), 0, cam_w - 1)
        v_int = np.clip(v[update_mask].astype(np.int32), 0, cam_h - 1)

        sampled = img_undist[v_int, u_int]
        if sampled.ndim == 1:
            colors[update_mask, 0] = sampled
            colors[update_mask, 1] = sampled
            colors[update_mask, 2] = sampled
        else:
            colors[update_mask, 0] = sampled[:, 2]  # BGR -> RGB
            colors[update_mask, 1] = sampled[:, 1]
            colors[update_mask, 2] = sampled[:, 0]

        colored[update_mask] = True
        best_dist[update_mask] = dist_to_center[update_mask]

        if (i + 1) % 50 == 0 or i == len(images) - 1:
            pct = np.sum(colored) / n_pts * 100
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(images)}] {pct:.1f}% colored ({elapsed:.1f}s)")

    total_colored = np.sum(colored)
    print(f"[Colorize] {total_colored:,}/{n_pts:,} points colored "
          f"({total_colored/n_pts*100:.1f}%)")
    return colors, colored


# ─── LiDAR extraction and odometry cloud ──────────────────────────

def extract_lidar_scans(bag_path, lid_topic, imu_topic):
    """Extract raw LiDAR scans from rosbag using rosbag_reader.

    Returns list of (timestamp, points_Nx3, times_N, intensities_N).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from rosbag_reader import read_rosbag
    print(f"[Odometry] Extracting LiDAR scans (topic={lid_topic})...")
    lidar_msgs, _, _ = read_rosbag(
        bag_path, lid_topic, imu_topic, img_topic=None,
        lidar_time_offset=0.0, imu_time_offset=0.0, img_time_offset=0.0
    )
    print(f"[Odometry] Extracted {len(lidar_msgs)} LiDAR scans")
    return lidar_msgs


def build_raw_lidar_cloud(lidar_scans, blind=0.8):
    """Stack all raw LiDAR scans in the sensor frame (no transforms).

    Returns (pts_lidar_Mx3 float32, intensities_M float32).
    The result should look like a cone / fan since all scans overlap at origin.
    """
    blind_sq = blind * blind
    all_pts = []
    all_intens = []
    for ts, pts, times, intens in lidar_scans:
        if len(pts) == 0:
            continue
        dist_sq = np.sum(pts * pts, axis=1)
        mask = dist_sq > blind_sq
        if mask.any():
            all_pts.append(pts[mask].astype(np.float32))
            all_intens.append(intens[mask].astype(np.float32))
    pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), dtype=np.float32)
    ints = np.concatenate(all_intens, axis=0) if all_intens else np.zeros(0, dtype=np.float32)
    print(f"[Raw LiDAR] {len(pts):,} points in sensor frame")
    return pts, ints


def build_odometry_cloud(lidar_scans, poses, ext_R, ext_T, blind=0.8):
    """Transform raw LiDAR scans to world frame using odometry poses.

    For each scan:
      1. Find nearest pose by scan-end timestamp
      2. Filter blind zone (range > blind)
      3. Transform: p_world = R_pose @ (ext_R @ p_lidar + ext_T) + t_pose

    This uses the odometry file output by the SLAM pipeline.
    The result differs from map.pcd because map.pcd is built internally
    with undistortion, scan matching, and voxel filtering, while this
    projects the raw scans using the nearest odometry pose.

    Returns (pts_world_Mx3 float32, intensities_M float32).
    """
    pose_times = np.array([p[0] for p in poses])
    blind_sq = blind * blind
    all_world = []
    all_intens = []
    t0 = time.time()

    for i, (ts, pts, times, intens) in enumerate(lidar_scans):
        if len(pts) == 0:
            continue
        # Use scan-end time for pose matching
        scan_end = ts + float(times.max()) if len(times) > 0 else ts
        idx = np.argmin(np.abs(pose_times - scan_end))

        # Blind zone filter
        dist_sq = np.sum(pts * pts, axis=1)
        mask = dist_sq > blind_sq
        pts_filt = pts[mask]
        intens_filt = intens[mask]
        if len(pts_filt) == 0:
            continue

        _, R_pose, t_pose = poses[idx]
        # LiDAR -> IMU -> World
        p_imu = pts_filt @ ext_R.T + ext_T[None, :]
        p_world = p_imu @ R_pose.T + t_pose[None, :]
        all_world.append(p_world.astype(np.float32))
        all_intens.append(intens_filt.astype(np.float32))

        if (i + 1) % 50 == 0 or i == len(lidar_scans) - 1:
            elapsed = time.time() - t0
            n_pts = sum(len(w) for w in all_world)
            print(f"  [{i+1}/{len(lidar_scans)}] {n_pts:,} world pts ({elapsed:.1f}s)")

    pts_world = np.concatenate(all_world, axis=0) if all_world else np.zeros((0, 3), dtype=np.float32)
    intensities = np.concatenate(all_intens, axis=0) if all_intens else np.zeros(0, dtype=np.float32)
    print(f"[Odometry] Built world cloud: {len(pts_world):,} points")
    return pts_world, intensities


def run_thickened_variants(bag_path, config_path, camera_path, output_base_dir,
                           num_variants=11, max_shift_s=0.001):
    """Re-run SLAM pipeline with shifted IMU timestamps.

    Creates num_variants runs with imu_time_offset shifted linearly from
    -max_shift_s to +max_shift_s. Returns list of (shift_value, points_Nx3).
    """
    shifts = np.linspace(-max_shift_s, max_shift_s, num_variants)
    os.makedirs(output_base_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_script = os.path.join(script_dir, 'fast_livo2.py')

    # Load base config to get original imu_time_offset
    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    base_imu_offset = base_cfg.get('time_offset', {}).get('imu_time_offset', 0.0)

    results = []
    t0 = time.time()

    for vi, shift_s in enumerate(shifts):
        print(f"\n[Thickened] Variant {vi+1}/{num_variants}: "
              f"IMU shift = {shift_s*1000:+.4f} ms")

        # Write modified config
        variant_cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy
        if 'time_offset' not in variant_cfg:
            variant_cfg['time_offset'] = {}
        variant_cfg['time_offset']['imu_time_offset'] = float(base_imu_offset + shift_s)

        variant_config_path = os.path.join(output_base_dir, f'config_variant_{vi:03d}.yaml')
        with open(variant_config_path, 'w') as f:
            yaml.dump(variant_cfg, f, default_flow_style=False)

        variant_output_dir = os.path.join(output_base_dir, f'variant_{vi:03d}')
        os.makedirs(variant_output_dir, exist_ok=True)

        # Run SLAM pipeline as subprocess
        cmd = [sys.executable, pipeline_script,
               '--bag', str(bag_path),
               '--config', variant_config_path,
               '--output', variant_output_dir]
        if camera_path:
            cmd.extend(['--camera-config', str(camera_path)])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Read map.pcd from variant
            map_path = os.path.join(variant_output_dir, 'map.pcd')
            if os.path.exists(map_path):
                pts = read_pcd_ascii(map_path)
                results.append((shift_s, pts))
                elapsed = time.time() - t0
                print(f"  -> {len(pts):,} points ({elapsed:.1f}s total)")
            else:
                print(f"  -> WARNING: No map.pcd produced, skipping")
        except subprocess.CalledProcessError as e:
            print(f"  -> ERROR: Pipeline failed (exit {e.returncode}), skipping")
            if e.stderr:
                for line in e.stderr.strip().split('\n')[-3:]:
                    print(f"     {line}")

    print(f"\n[Thickened] {len(results)}/{num_variants} variants completed "
          f"({time.time()-t0:.1f}s)")
    return results


def sort_colored_first(pts, rgb, rgb_mask, display_count, intensities=None):
    """Sort points so colored points come first (for setDrawRange display).

    Returns reordered (pts, rgb, rgb_mask, intensities).
    """
    colored_idx = np.where(rgb_mask)[0]
    uncolored_idx = np.where(~rgb_mask)[0]

    # If display_count fits all colored, put colored first then uncolored
    if len(colored_idx) <= display_count:
        order = np.concatenate([colored_idx, uncolored_idx])
    else:
        # More colored than display_count: random subset of colored first
        rng = np.random.default_rng(42)
        display_colored = rng.choice(colored_idx, display_count, replace=False)
        display_colored.sort()
        remaining_colored = np.setdiff1d(colored_idx, display_colored)
        order = np.concatenate([display_colored, remaining_colored, uncolored_idx])

    out_intens = intensities[order] if intensities is not None else None
    return pts[order], rgb[order], rgb_mask[order], out_intens


# ─── Ground plane alignment ────────────────────────────────────────

def _rotation_from_vectors(from_vec, to_vec):
    """Compute rotation matrix that rotates from_vec to to_vec."""
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    dot = np.dot(from_vec, to_vec)
    if dot > 0.9999:
        return np.eye(3)
    cross = np.cross(from_vec, to_vec)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-10:
        return np.diag([1.0, -1.0, -1.0])
    axis = cross / cross_norm
    angle = np.arccos(np.clip(dot, -1, 1))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _rot_x(angle_deg):
    """Rotation matrix around X axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _rot_y(angle_deg):
    """Rotation matrix around Y axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def align_to_ground_plane(pts, bag_path=None, imu_topic='/livox/imu',
                          ground_thickness=0.04):
    """Two-pass ground alignment.

    Pass 1: IMU gravity vector for coarse alignment.
    Pass 2: Brute-force search over pitch/roll angles to maximize
            the number of points within a thin horizontal band.
    """
    # ── Pass 1: IMU gravity alignment ────────────────────────────
    R1 = np.eye(3)
    if bag_path:
        try:
            from rosbag_reader import read_rosbag
            _, imu_msgs, _ = read_rosbag(bag_path, '/livox/lidar', imu_topic)
            if len(imu_msgs) >= 20:
                accs = np.array([m.acc for m in imu_msgs[:200]])
                gravity = accs.mean(axis=0)
                gravity_dir = gravity / np.linalg.norm(gravity)
                up = -gravity_dir
                if up[2] < 0:
                    up = -up
                R1 = _rotation_from_vectors(up, np.array([0.0, 0.0, 1.0]))
                tilt1 = np.degrees(np.arccos(np.clip(R1[2, 2], -1, 1)))
                print(f"  [Align] Pass 1 — IMU gravity: {tilt1:.1f}° correction")
        except Exception as e:
            print(f"  [Align] IMU read failed ({e}), skipping pass 1")

    pts_r1 = (R1 @ pts.T).T

    # ── Pass 2: Brute-force tilt optimization ─────────────────────
    # Subsample for speed (100K points is enough for density counting)
    if len(pts_r1) > 100000:
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(len(pts_r1), 100000, replace=False)
        pts_sub = pts_r1[sub_idx]
    else:
        pts_sub = pts_r1

    half_t = ground_thickness / 2.0

    def count_ground_pts(rx_deg, ry_deg, pts_s):
        """Count how many points fall within a thin Z band after rotation."""
        R = _rot_x(rx_deg) @ _rot_y(ry_deg)
        z_rot = pts_s @ R[2, :]  # only need Z component
        # Find the Z with the most points in the band
        z_min, z_max = z_rot.min(), z_rot.max()
        # Histogram approach: bin Z values and find the peak
        bins = np.arange(z_min, z_max + ground_thickness, ground_thickness)
        if len(bins) < 2:
            return 0
        counts, _ = np.histogram(z_rot, bins=bins)
        return counts.max()

    # Coarse search: ±15° in 1° steps
    best_rx, best_ry, best_count = 0.0, 0.0, 0
    for rx in np.arange(-15, 15.5, 1.0):
        for ry in np.arange(-15, 15.5, 1.0):
            c = count_ground_pts(rx, ry, pts_sub)
            if c > best_count:
                best_count = c
                best_rx, best_ry = rx, ry

    # Fine search: ±2° around best, in 0.1° steps
    for rx in np.arange(best_rx - 2, best_rx + 2.05, 0.1):
        for ry in np.arange(best_ry - 2, best_ry + 2.05, 0.1):
            c = count_ground_pts(rx, ry, pts_sub)
            if c > best_count:
                best_count = c
                best_rx, best_ry = rx, ry

    R2 = _rot_x(best_rx) @ _rot_y(best_ry)
    tilt2 = np.sqrt(best_rx**2 + best_ry**2)
    print(f"  [Align] Pass 2 — tilt optimization: rx={best_rx:.1f}° ry={best_ry:.1f}° "
          f"({best_count:,} pts in {ground_thickness*100:.0f}cm band)")

    R_total = R2 @ R1
    return R_total


# ─── Viewer data packing ───────────────────────────────────────────

def pack_viewer_dataset(name, pts, rgb, rgb_mask, total_points=None,
                        display_count=None, intensities=None):
    """Pack one pointcloud dataset for embedding in the HTML viewer.

    Args:
        display_count: how many of the N points to render (setDrawRange).
                       If None, all points are rendered. Points should already
                       be sorted (colored first) via sort_colored_first().
        intensities:   Per-point LiDAR return intensity/reflectivity (float32).
                       If None, zeros are used.
    """
    pts = np.asarray(pts, dtype=np.float32)
    N = len(pts)
    dc = int(min(display_count, N)) if display_count is not None else N
    # Intensity: use real data if provided, else zeros
    if intensities is not None:
        ints = np.asarray(intensities, dtype=np.float32)[:N]
        if len(ints) < N:
            ints = np.pad(ints, (0, N - len(ints)))
    else:
        ints = np.zeros(N, dtype=np.float32)
    ts = np.arange(N, dtype=np.float32) / max(N - 1, 1)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = pts.mean(axis=0)
    extent = maxs - mins

    int_max = float(ints.max())

    packed = np.empty((N, 5), dtype=np.float32)
    packed[:, :3] = pts
    packed[:, 3] = ints
    packed[:, 4] = ts
    b64_data = base64.b64encode(zlib.compress(packed.tobytes(), 9)).decode('ascii')

    rgb = np.clip(rgb[:, :3], 0, 255).astype(np.uint8)
    rgb_mask_u8 = rgb_mask.astype(np.uint8)
    rgb_b64 = base64.b64encode(zlib.compress(np.ascontiguousarray(rgb).tobytes(), 9)).decode('ascii')
    rgbmask_b64 = base64.b64encode(zlib.compress(np.ascontiguousarray(rgb_mask_u8).tobytes(), 9)).decode('ascii')
    rgb_point_count = int(np.count_nonzero(rgb_mask_u8))

    return {
        "name": name,
        "total_points": int(total_points if total_points is not None else N),
        "meta": {
            "num_points": N,
            "display_count": dc,
            "center": [float(x) for x in center],
            "mins": [float(x) for x in mins],
            "maxs": [float(x) for x in maxs],
            "extent": [float(x) for x in extent],
            "intensity_max": int_max,
            "time_max": 1.0,
            "has_rgb": True,
            "rgb_point_count": rgb_point_count,
            "compressed": True,
        },
        "b64_data": b64_data,
        "rgb_b64": rgb_b64,
        "rgbmask_b64": rgbmask_b64,
    }


# ─── HTML viewer generation (adapted from lio_standalone) ──────────

def build_viewer_html(datasets):
    """Build a self-contained multi-dataset pointcloud viewer HTML.

    Features: RGB + 6 color modes, slice tools with CSV export,
    heatmap sweep (density maps across all slices along an axis),
    adjustable point size, background brightness, reset view.
    """
    datasets_json = json.dumps(datasets)
    html = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Point Cloud Viewer</title>
<style>
  body { margin: 0; overflow: hidden; background: #1a1a2e; font-family: monospace; }
  #info {
    position: absolute; top: 10px; left: 10px; color: #eee;
    background: rgba(0,0,0,0.72); padding: 10px 15px; border-radius: 6px;
    font-size: 13px; line-height: 1.5; z-index: 10; pointer-events: none;
  }
  #info b { color: #4fc3f7; }
  #controls {
    position: absolute; bottom: 10px; left: 10px; color: #aaa;
    background: rgba(0,0,0,0.72); padding: 8px 12px; border-radius: 6px;
    font-size: 11px; z-index: 10; pointer-events: none;
  }
  #datasetselector {
    position: absolute; top: 10px; right: 10px; z-index: 10;
    background: rgba(0,0,0,0.72); padding: 8px 12px; border-radius: 6px;
    color: #eee; font-size: 12px;
  }
  #datasetselector select {
    background: #24243a; color: #eee; border: 1px solid #555;
    border-radius: 4px; padding: 4px 8px; font-family: monospace; font-size: 12px;
    margin-left: 4px;
  }
  #colormode {
    position: absolute; top: 50px; right: 10px; z-index: 10;
    background: rgba(0,0,0,0.72); padding: 8px; border-radius: 6px;
  }
  #colormode button {
    background: #333; color: #eee; border: 1px solid #555;
    padding: 5px 10px; margin: 2px; border-radius: 4px; cursor: pointer;
    font-family: monospace; font-size: 12px;
  }
  #colormode button.active { background: #4fc3f7; color: #000; border-color: #4fc3f7; }
  #colormode button:hover { background: #555; }
  #colormode button:disabled { opacity: 0.45; cursor: not-allowed; }
  #pointsize {
    position: absolute; top: 98px; right: 10px; z-index: 10;
    background: rgba(0,0,0,0.72); padding: 8px 12px; border-radius: 6px;
    color: #aaa; font-size: 12px;
  }
  #pointsize input { width: 100px; vertical-align: middle; }
  #bgbox {
    position: absolute; top: 138px; right: 10px; z-index: 10;
    background: rgba(0,0,0,0.72); padding: 8px 12px; border-radius: 6px;
    color: #aaa; font-size: 12px;
  }
  #bgbox input { width: 100px; vertical-align: middle; }
  #slicebox {
    position: absolute; top: 178px; right: 10px; z-index: 10;
    background: rgba(0,0,0,0.72); padding: 10px 12px; border-radius: 6px;
    color: #ddd; font-size: 12px; width: 330px;
  }
  .panel-section { margin: 4px 0; }
  .panel-section label { display: inline-block; min-width: 90px; color: #aaa; }
  .panel-section select, .panel-section input[type=number] {
    background: #24243a; color: #eee; border: 1px solid #555;
    border-radius: 4px; padding: 3px 6px; font-family: monospace; font-size: 12px;
  }
  .panel-section input[type=range] { width: 160px; vertical-align: middle; }
  .panel-section button, #slicebox button {
    background: #333; color: #eee; border: 1px solid #555;
    padding: 5px 8px; margin: 2px 2px 0 0; border-radius: 4px; cursor: pointer;
    font-family: monospace; font-size: 11px;
  }
  .panel-section button.active, #slicebox button.active { background: #4fc3f7; color: #000; border-color: #4fc3f7; }
  .divider { border-top: 1px solid #444; margin: 8px 0; }
  .hint { color: #9aa; font-size: 11px; margin-top: 4px; }
  /* Heatmap overlay */
  #heatmapOverlay {
    display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.92); z-index: 100;
    padding: 20px; box-sizing: border-box;
    flex-direction: column; overflow: hidden;
  }
  #heatmapOverlay.visible { display: flex; }
  #heatmapOverlay .hm-header {
    color: #eee; font-size: 16px; margin-bottom: 12px;
    display: flex; justify-content: space-between; align-items: center;
    flex-shrink: 0;
  }
  #heatmapOverlay .hm-close {
    background: #c33; color: #fff; border: none; padding: 6px 16px;
    border-radius: 4px; cursor: pointer; font-family: monospace; font-size: 13px;
  }
  #heatmapOverlay .hm-detail {
    display: flex; gap: 16px; margin-bottom: 16px; align-items: flex-start;
    flex-shrink: 0;
  }
  #heatmapOverlay .hm-detail canvas {
    border: 1px solid #555; image-rendering: pixelated;
  }
  #heatmapOverlay .hm-detail-info {
    color: #ccc; font-size: 12px; line-height: 1.6; min-width: 180px;
  }
  #heatmapOverlay .hm-dl-row { margin: 10px 0; flex-shrink: 0; }
  #heatmapOverlay .hm-dl-row button {
    background: #2a6; color: #fff; border: none; padding: 6px 14px;
    border-radius: 4px; cursor: pointer; font-family: monospace; font-size: 12px;
    margin-right: 8px;
  }
  #heatmapOverlay .hm-filmstrip {
    display: flex; flex-wrap: wrap; gap: 6px;
    overflow-y: auto; flex: 1 1 auto; min-height: 0;
    align-content: flex-start;
  }
  #heatmapOverlay .hm-thumb {
    cursor: pointer; border: 2px solid transparent; border-radius: 3px;
    position: relative;
  }
  #heatmapOverlay .hm-thumb.selected { border-color: #4fc3f7; }
  #heatmapOverlay .hm-thumb canvas { image-rendering: pixelated; display: block; }
  #heatmapOverlay .hm-thumb-label {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: rgba(0,0,0,0.7); color: #ccc; font-size: 9px;
    text-align: center; padding: 1px 0;
  }
</style>
</head>
<body>
<div id="info"></div>
<div id="cursorCoords" style="position:fixed;bottom:8px;left:8px;background:rgba(0,0,0,0.7);color:#4fc3f7;font-family:monospace;font-size:13px;padding:4px 10px;border-radius:4px;pointer-events:none;z-index:100;">X: —  Y: —  Z: —</div>
<div id="controls">
  Drag: orbit | Scroll: zoom | Right-drag: pan | R: reset view
</div>
<div id="datasetselector">
  <label>Dataset:</label>
  <select id="datasetSelect"></select>
</div>
<div id="colormode">
  <button data-mode="rgb" class="active">RGB</button>
  <button data-mode="intensity">Intensity</button>
  <button data-mode="height">Height</button>
  <button data-mode="distance">Distance</button>
  <button data-mode="x">X-axis</button>
  <button data-mode="y">Y-axis</button>
  <button data-mode="z">Z-axis</button>
</div>
<div id="pointsize">
  Size: <input id="pointsizeRange" type="range" min="0.001" max="0.5" step="0.001" value="0.02">
  <span id="psval">0.02</span>
</div>
<div id="bgbox">
  BG: <input id="bgRange" type="range" min="0" max="255" step="1" value="26">
  <span id="bgval">26</span>
</div>
<div id="slicebox">
  <div><b>Single Slice</b></div>
  <div class="panel-section"><label>Axis</label>
    <select id="sliceAxis">
      <option value="0">X</option>
      <option value="1">Y</option>
      <option value="2" selected>Z</option>
    </select>
  </div>
  <div class="panel-section"><label>Position</label>
    <input id="slicePosRange" type="range" min="0" max="1000" step="1" value="500">
    <span id="slicePosLabel">-</span>
  </div>
  <div class="panel-section"><label>Thickness (m)</label>
    <input id="sliceThickness" type="number" min="0.001" step="0.01" value="0.10">
  </div>
  <div class="panel-section">
    <button id="slicePreviewBtn">Preview</button>
    <button id="sliceClearBtn">Clear</button>
    <button id="sliceDownloadBtn">CSV</button>
  </div>
  <div class="panel-section"><span id="sliceStatus">No slice active.</span></div>

  <div class="divider"></div>
  <div><b>Heatmap Sweep</b></div>
  <div class="hint">Generate density heatmaps across all slices along an axis.</div>
  <div class="panel-section"><label>Sweep axis</label>
    <select id="sweepAxis">
      <option value="0">X</option>
      <option value="1">Y</option>
      <option value="2" selected>Z</option>
    </select>
  </div>
  <div class="panel-section"><label>Num slices</label>
    <input id="sweepNumSlices" type="number" min="2" max="500" value="50">
  </div>
  <div class="panel-section"><label>Grid res (px)</label>
    <input id="sweepGridRes" type="number" min="32" max="1024" value="256">
  </div>
  <div class="hint" style="margin-top:6px">Bounding box (leave as-is for full extent):</div>
  <div class="panel-section"><label>X min / max</label>
    <input id="sweepXmin" type="number" step="0.1" style="width:70px">
    <input id="sweepXmax" type="number" step="0.1" style="width:70px">
  </div>
  <div class="panel-section"><label>Y min / max</label>
    <input id="sweepYmin" type="number" step="0.1" style="width:70px">
    <input id="sweepYmax" type="number" step="0.1" style="width:70px">
  </div>
  <div class="panel-section"><label>Z min / max</label>
    <input id="sweepZmin" type="number" step="0.1" style="width:70px">
    <input id="sweepZmax" type="number" step="0.1" style="width:70px">
  </div>
  <div class="panel-section">
    <button id="sweepRunBtn">Generate Heatmaps</button>
    <button id="sweepResetBoundsBtn">Reset Bounds</button>
  </div>
  <div class="panel-section">
    <a id="fullSliceLink" href="slices/slice_viewer_z.html" target="_blank"
       style="color:#4fc3f7;font-size:12px;text-decoration:none;">
       Open Full-Resolution Slice Analysis &rarr;</a>
    <div style="font-size:10px;color:#666;margin-top:2px;">
      (uses all points from map.pcd, not just embedded subset)</div>
  </div>
  <div class="panel-section"><span id="sweepStatus"></span></div>
</div>
<div id="heatmapOverlay">
  <div class="hm-header">
    <span id="hmTitle">Density Heatmaps</span>
    <div style="display:flex;gap:6px;align-items:center;">
      <button id="hmPrevBtn" style="background:#444;color:#eee;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-family:monospace;font-size:16px;">&larr; Prev</button>
      <button id="hmNextBtn" style="background:#444;color:#eee;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-family:monospace;font-size:16px;">Next &rarr;</button>
      <button id="hmMarkerBtn" style="background:#444;color:#eee;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-family:monospace;font-size:13px;">Place Marker</button>
      <button id="hmClearMarkersBtn" style="background:#444;color:#eee;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-family:monospace;font-size:13px;">Clear Markers</button>
      <span id="hmMarkerCount" style="color:#aaa;font-size:12px;"></span>
      <button class="hm-close" id="hmCloseBtn">Close</button>
    </div>
  </div>
  <div class="hm-detail" id="hmDetail"></div>
  <div id="hmCoordReadout" style="color:#0f0;font-family:monospace;font-size:12px;height:16px;margin-bottom:4px;"></div>
  <div class="hm-dl-row">
    <button id="hmDlCurrent">Download Current Slice PNG</button>
    <button id="hmDlAll">Download All as PNGs</button>
  </div>
  <div class="hm-filmstrip" id="hmFilmstrip"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATASETS = __DATASETS_JSON__;
const datasetCache = new Map();
let activeDatasetIndex = 0;
let currentColorMode = 'rgb';
let slicePreviewEnabled = false;
let currentSliceSelection = null;

function decodeB64(b64str, compressed) {
  const raw = Uint8Array.from(atob(b64str), c => c.charCodeAt(0));
  return compressed ? pako.inflate(raw) : raw;
}

function decodeDataset(index) {
  if (datasetCache.has(index)) return datasetCache.get(index);
  const ds = DATASETS[index];
  const compressed = !!(ds.meta && ds.meta.compressed);
  const buf = decodeB64(ds.b64_data, compressed);
  const floats = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const N = ds.meta.num_points;
  const STRIDE = 5;
  const positions = new Float32Array(N * 3);
  const intensities = new Float32Array(N);
  const timestamps = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    positions[i*3]   = floats[i*STRIDE];
    positions[i*3+1] = floats[i*STRIDE+1];
    positions[i*3+2] = floats[i*STRIDE+2];
    intensities[i]   = floats[i*STRIDE+3];
    timestamps[i]    = floats[i*STRIDE+4];
  }
  const decoded = { ...ds, positions, intensities, timestamps };
  if (ds.rgb_b64) {
    decoded.rgb = new Uint8Array(decodeB64(ds.rgb_b64, compressed));
  } else {
    decoded.rgb = null;
  }
  if (ds.rgbmask_b64) {
    decoded.rgbMask = new Uint8Array(decodeB64(ds.rgbmask_b64, compressed));
  } else {
    decoded.rgbMask = null;
  }
  datasetCache.set(index, decoded);
  return decoded;
}

function getActiveDataset() {
  return decodeDataset(activeDatasetIndex);
}

function datasetSupportsRgb(ds) {
  return !!(ds && ds.meta && ds.meta.has_rgb && ds.rgb);
}

function datasetHasIntensity(ds) {
  return !!(ds && ds.meta && ds.meta.intensity_max > 0);
}

function syncColorModeButtons(ds) {
  const rgbSupported = datasetSupportsRgb(ds);
  const intensitySupported = datasetHasIntensity(ds);
  document.querySelectorAll('#colormode button').forEach(b => {
    b.classList.remove('active');
    if (b.dataset.mode === 'rgb') {
      b.disabled = !rgbSupported;
    } else if (b.dataset.mode === 'intensity') {
      b.disabled = !intensitySupported;
    } else {
      b.disabled = false;
    }
    if (b.dataset.mode === currentColorMode) b.classList.add('active');
  });
}

function turboColormap(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 0.13572138 + t*(4.6153926 + t*(-42.66032258 + t*(132.13108234 + t*(-152.54895899 + t*58.9161376))))));
  const g = Math.max(0, Math.min(1, 0.09140261 + t*(2.19418839 + t*(4.84296658 + t*(-14.18503333 + t*(4.27729857 + t*2.82956604))))));
  const b = Math.max(0, Math.min(1, 0.1066733 + t*(12.64194608 + t*(-60.58204836 + t*(110.36276771 + t*(-89.90310912 + t*27.34824973))))));
  return [r, g, b];
}

function colorByMode(mode, positions, intensities, timestamps, rgb, rgbMask, N, meta) {
  const colors = new Float32Array(N * 3);
  let i;
  if (mode === 'rgb') {
    if (!meta.has_rgb || !rgb) return colorByMode('height', positions, intensities, timestamps, rgb, rgbMask, N, meta);
    for (i = 0; i < N; i++) {
      const hasc = rgbMask ? (rgbMask[i] > 0) : true;
      if (hasc) {
        colors[i*3]   = rgb[i*3]   / 255.0;
        colors[i*3+1] = rgb[i*3+1] / 255.0;
        colors[i*3+2] = rgb[i*3+2] / 255.0;
      } else {
        colors[i*3] = 0.25; colors[i*3+1] = 0.25; colors[i*3+2] = 0.25;
      }
    }
    return colors;
  }

  if (mode === 'intensity') {
    const imax = meta.intensity_max || 1;
    for (i = 0; i < N; i++) {
      const t = Math.min(1, intensities[i] / imax);
      const c = turboColormap(t);
      colors[i*3] = c[0]; colors[i*3+1] = c[1]; colors[i*3+2] = c[2];
    }
    return colors;
  }

  let vmin = Infinity, vmax = -Infinity;
  const cx = meta.center[0], cy = meta.center[1], cz = meta.center[2];
  if (mode === 'height') {
    for (i = 0; i < N; i++) { const v = positions[i*3+2]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  } else if (mode === 'x') {
    for (i = 0; i < N; i++) { const v = positions[i*3]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  } else if (mode === 'y') {
    for (i = 0; i < N; i++) { const v = positions[i*3+1]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  } else if (mode === 'z') {
    for (i = 0; i < N; i++) { const v = positions[i*3+2]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  } else if (mode === 'distance') {
    for (i = 0; i < N; i++) {
      const dx = positions[i*3]-cx, dy = positions[i*3+1]-cy, dz = positions[i*3+2]-cz;
      const v = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (v < vmin) vmin = v; if (v > vmax) vmax = v;
    }
  } else {
    return colorByMode('rgb', positions, intensities, timestamps, rgb, rgbMask, N, meta);
  }

  const range = (vmax - vmin) || 1;
  for (i = 0; i < N; i++) {
    let t = 0;
    if (mode === 'height' || mode === 'z') t = (positions[i*3+2] - vmin) / range;
    else if (mode === 'x') t = (positions[i*3] - vmin) / range;
    else if (mode === 'y') t = (positions[i*3+1] - vmin) / range;
    else if (mode === 'distance') {
      const dx = positions[i*3]-cx, dy = positions[i*3+1]-cy, dz = positions[i*3+2]-cz;
      t = (Math.sqrt(dx*dx + dy*dy + dz*dz) - vmin) / range;
    }
    const c = turboColormap(t);
    colors[i*3] = c[0]; colors[i*3+1] = c[1]; colors[i*3+2] = c[2];
  }
  return colors;
}

function makeSliceSelection(ds, axis, position, thickness) {
  const half = Math.max(0.0005, thickness / 2);
  const pos = ds.positions;
  const outIdx = [];
  for (let i = 0; i < ds.meta.num_points; i++) {
    const v = pos[i*3 + axis];
    if (Math.abs(v - position) <= half) outIdx.push(i);
  }
  return { axis, position, thickness, indices: outIdx };
}

function materializeSelection(ds, selection) {
  if (!selection) {
    return {
      positions: ds.positions, intensities: ds.intensities, timestamps: ds.timestamps,
      rgb: ds.rgb, rgbMask: ds.rgbMask, count: ds.meta.num_points,
    };
  }
  const M = selection.indices.length;
  const pos = new Float32Array(M * 3);
  const inten = new Float32Array(M);
  const ts = new Float32Array(M);
  const rgb = ds.rgb ? new Uint8Array(M * 3) : null;
  const rgbMask = ds.rgbMask ? new Uint8Array(M) : null;
  for (let j = 0; j < M; j++) {
    const i = selection.indices[j];
    pos[j*3] = ds.positions[i*3];
    pos[j*3+1] = ds.positions[i*3+1];
    pos[j*3+2] = ds.positions[i*3+2];
    inten[j] = ds.intensities[i];
    ts[j] = ds.timestamps[i];
    if (rgb) {
      rgb[j*3] = ds.rgb[i*3];
      rgb[j*3+1] = ds.rgb[i*3+1];
      rgb[j*3+2] = ds.rgb[i*3+2];
    }
    if (rgbMask) rgbMask[j] = ds.rgbMask[i];
  }
  return { positions: pos, intensities: inten, timestamps: ts, rgb, rgbMask, count: M };
}

/* ── Three.js scene ── */

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 10000);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.minPolarAngle = 0;
controls.maxPolarAngle = Math.PI;
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.DOLLY,
  RIGHT: THREE.MOUSE.PAN,
};

const geometry = new THREE.BufferGeometry();
const material = new THREE.PointsMaterial({ size: 0.02, vertexColors: true, sizeAttenuation: true });
const points = new THREE.Points(geometry, material);
scene.add(points);

const axesHelper = new THREE.AxesHelper(1);
scene.add(axesHelper);

/* ── Slice helpers ── */

function getSliceInputs() {
  const axis = parseInt(document.getElementById('sliceAxis').value, 10);
  const slider = parseInt(document.getElementById('slicePosRange').value, 10);
  const ds = getActiveDataset();
  const minv = ds.meta.mins[axis];
  const maxv = ds.meta.maxs[axis];
  const position = minv + (maxv - minv) * (slider / 1000.0);
  const thickness = Math.max(0.001, parseFloat(document.getElementById('sliceThickness').value || '0.1'));
  return { axis, position, thickness };
}

function updateSlicePositionLabel() {
  const inp = getSliceInputs();
  const axes = ['X', 'Y', 'Z'];
  document.getElementById('slicePosLabel').textContent = axes[inp.axis] + '=' + inp.position.toFixed(3) + ' m';
}

/* ── Display logic ── */

function updateInfo(ds, visibleCount, sliceActive, effectiveColorMode) {
  const m = ds.meta;
  const info = document.getElementById('info');
  info.innerHTML =
    '<b>' + ds.name + '</b><br>' +
    'Rendered: ' + visibleCount.toLocaleString() +
    ' | Embedded: ' + m.num_points.toLocaleString() +
    (ds.total_points !== m.num_points ? ' (of ' + ds.total_points.toLocaleString() + ')' : '') + '<br>' +
    'Center: (' + m.center[0].toFixed(1) + ', ' + m.center[1].toFixed(1) + ', ' + m.center[2].toFixed(1) + ')<br>' +
    'Extent: ' + m.extent[0].toFixed(1) + ' x ' + m.extent[1].toFixed(1) + ' x ' + m.extent[2].toFixed(1) + ' m<br>' +
    'RGB: ' + (m.has_rgb ? (m.rgb_point_count.toLocaleString() + ' pts') : 'No') + '<br>' +
    'Color: ' + effectiveColorMode.toUpperCase() + '<br>' +
    'Slice: ' + (sliceActive ? 'ON' : 'OFF');
}

function applyDisplay(resetView) {
  const ds = getActiveDataset();
  const display = slicePreviewEnabled ? materializeSelection(ds, currentSliceSelection) :
                                        materializeSelection(ds, null);
  let effectiveColorMode = currentColorMode;
  if (effectiveColorMode === 'rgb' && !datasetSupportsRgb(ds)) {
    effectiveColorMode = 'height';
  }
  if (effectiveColorMode === 'intensity' && !datasetHasIntensity(ds)) {
    effectiveColorMode = 'height';
  }
  const colors = colorByMode(effectiveColorMode, display.positions, display.intensities,
                             display.timestamps, display.rgb, display.rgbMask,
                             display.count, ds.meta);
  geometry.setAttribute('position', new THREE.BufferAttribute(display.positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  // Use setDrawRange to limit rendered points while keeping all for slicing
  const dc = ds.meta.display_count || display.count;
  if (!slicePreviewEnabled && dc < display.count) {
    geometry.setDrawRange(0, dc);
  } else {
    geometry.setDrawRange(0, display.count);
  }

  geometry.computeBoundingSphere();
  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;

  const maxExt = Math.max(ds.meta.extent[0], ds.meta.extent[1], ds.meta.extent[2], 1.0);
  axesHelper.scale.setScalar(maxExt * 0.05);
  axesHelper.position.set(ds.meta.center[0], ds.meta.center[1], ds.meta.center[2]);

  if (resetView) {
    const cx = ds.meta.center[0], cy = ds.meta.center[1], cz = ds.meta.center[2];
    camera.far = Math.max(1000, maxExt * 20);
    camera.updateProjectionMatrix();
    camera.position.set(cx, cy - maxExt*1.2, cz + maxExt*0.6);
    controls.target.set(cx, cy, cz);
    controls.update();
  }

  const visibleCount = slicePreviewEnabled ? display.count : Math.min(dc, display.count);
  updateInfo(ds, visibleCount, slicePreviewEnabled, effectiveColorMode);
}

function setColorMode(mode) {
  const ds = getActiveDataset();
  if (mode === 'rgb' && !datasetSupportsRgb(ds)) return;
  if (mode === 'intensity' && !datasetHasIntensity(ds)) return;
  currentColorMode = mode;
  syncColorModeButtons(ds);
  applyDisplay(false);
}

function previewSlice() {
  const ds = getActiveDataset();
  const inp = getSliceInputs();
  currentSliceSelection = makeSliceSelection(ds, inp.axis, inp.position, inp.thickness);
  slicePreviewEnabled = true;
  document.getElementById('slicePreviewBtn').classList.add('active');
  const axisNames = ['X', 'Y', 'Z'];
  document.getElementById('sliceStatus').textContent =
    'Previewing ' + axisNames[inp.axis] + ' slice: ' + currentSliceSelection.indices.length.toLocaleString() + ' points';
  applyDisplay(false);
}

function clearSlicePreview() {
  slicePreviewEnabled = false;
  currentSliceSelection = null;
  document.getElementById('slicePreviewBtn').classList.remove('active');
  document.getElementById('sliceStatus').textContent = 'No slice preview active.';
  applyDisplay(false);
}

function downloadSliceCsv() {
  const ds = getActiveDataset();
  const inp = getSliceInputs();
  const sel = makeSliceSelection(ds, inp.axis, inp.position, inp.thickness);
  const hasRgb = !!(ds.meta && ds.meta.has_rgb && ds.rgb);
  const lines = [hasRgb ? 'x,y,z,r,g,b,has_color' : 'x,y,z'];
  for (const i of sel.indices) {
    if (hasRgb) {
      const hasc = ds.rgbMask ? (ds.rgbMask[i] > 0 ? 1 : 0) : 1;
      lines.push(
        ds.positions[i*3].toFixed(6) + ',' +
        ds.positions[i*3+1].toFixed(6) + ',' +
        ds.positions[i*3+2].toFixed(6) + ',' +
        String(ds.rgb[i*3]) + ',' +
        String(ds.rgb[i*3+1]) + ',' +
        String(ds.rgb[i*3+2]) + ',' +
        String(hasc)
      );
    } else {
      lines.push(
        ds.positions[i*3].toFixed(6) + ',' +
        ds.positions[i*3+1].toFixed(6) + ',' +
        ds.positions[i*3+2].toFixed(6)
      );
    }
  }
  const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  const axisNames = ['x','y','z'];
  a.href = URL.createObjectURL(blob);
  a.download = 'slice_' + axisNames[inp.axis] + '_' + inp.position.toFixed(3) + 'm.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(a.href), 500);
  document.getElementById('sliceStatus').textContent =
    'Downloaded slice CSV (' + sel.indices.length.toLocaleString() + ' points).';
}

/* ── Heatmap Sweep ── */

let sweepResults = null;
let sweepSelectedIdx = 0;

function runHeatmapSweep() {
  const ds = getActiveDataset();
  const axis = parseInt(document.getElementById('sweepAxis').value, 10);
  const numSlices = Math.max(2, Math.min(500, parseInt(document.getElementById('sweepNumSlices').value) || 50));
  const gridRes = Math.max(32, Math.min(1024, parseInt(document.getElementById('sweepGridRes').value) || 256));

  const axisNames = ['X', 'Y', 'Z'];
  // The two axes perpendicular to the sweep axis
  const ax1 = axis === 0 ? 1 : 0;
  const ax2 = axis <= 1 ? 2 : 1;

  const N = ds.meta.num_points;
  const pos = ds.positions;

  // Read bounding box from inputs
  const bxMin = parseFloat(document.getElementById('sweepXmin').value);
  const bxMax = parseFloat(document.getElementById('sweepXmax').value);
  const byMin = parseFloat(document.getElementById('sweepYmin').value);
  const byMax = parseFloat(document.getElementById('sweepYmax').value);
  const bzMin = parseFloat(document.getElementById('sweepZmin').value);
  const bzMax = parseFloat(document.getElementById('sweepZmax').value);
  const bounds = [
    isNaN(bxMin) ? -Infinity : bxMin, isNaN(bxMax) ? Infinity : bxMax,
    isNaN(byMin) ? -Infinity : byMin, isNaN(byMax) ? Infinity : byMax,
    isNaN(bzMin) ? -Infinity : bzMin, isNaN(bzMax) ? Infinity : bzMax,
  ];

  // Filter points by bounding box
  const filteredIdx = [];
  for (let i = 0; i < N; i++) {
    const x = pos[i*3], y = pos[i*3+1], z = pos[i*3+2];
    if (x >= bounds[0] && x <= bounds[1] &&
        y >= bounds[2] && y <= bounds[3] &&
        z >= bounds[4] && z <= bounds[5]) {
      filteredIdx.push(i);
    }
  }

  if (filteredIdx.length === 0) {
    document.getElementById('sweepStatus').textContent = 'No points in bounding box!';
    return;
  }

  // Compute actual extent of filtered points along each axis
  let fMins = [Infinity, Infinity, Infinity];
  let fMaxs = [-Infinity, -Infinity, -Infinity];
  for (const i of filteredIdx) {
    for (let a = 0; a < 3; a++) {
      const v = pos[i*3 + a];
      if (v < fMins[a]) fMins[a] = v;
      if (v > fMaxs[a]) fMaxs[a] = v;
    }
  }

  const sweepMin = fMins[axis];
  const sweepMax = fMaxs[axis];
  const sliceThickness = (sweepMax - sweepMin) / numSlices;

  const u_min = fMins[ax1], u_max = fMaxs[ax1];
  const v_min = fMins[ax2], v_max = fMaxs[ax2];
  const u_range = u_max - u_min || 1;
  const v_range = v_max - v_min || 1;

  // Make grid square by aspect ratio
  let gridW, gridH;
  if (u_range >= v_range) {
    gridW = gridRes;
    gridH = Math.max(1, Math.round(gridRes * v_range / u_range));
  } else {
    gridH = gridRes;
    gridW = Math.max(1, Math.round(gridRes * u_range / v_range));
  }

  document.getElementById('sweepStatus').textContent = 'Computing (' + filteredIdx.length.toLocaleString() + ' pts in bounds)...';

  // Bin filtered points by slice
  const sliceBins = new Array(numSlices);
  for (let s = 0; s < numSlices; s++) sliceBins[s] = [];

  for (const i of filteredIdx) {
    const sv = pos[i*3 + axis];
    let si = Math.floor((sv - sweepMin) / sliceThickness);
    if (si < 0) si = 0;
    if (si >= numSlices) si = numSlices - 1;
    sliceBins[si].push(i);
  }

  // For each slice, build 2D density grid and render to canvas
  let globalMaxDensity = 0;
  const grids = [];
  for (let s = 0; s < numSlices; s++) {
    const grid = new Float32Array(gridW * gridH);
    for (const idx of sliceBins[s]) {
      const u = pos[idx*3 + ax1];
      const v = pos[idx*3 + ax2];
      let gu = Math.floor((u - u_min) / u_range * (gridW - 1));
      let gv = Math.floor((v - v_min) / v_range * (gridH - 1));
      if (gu < 0) gu = 0; if (gu >= gridW) gu = gridW - 1;
      if (gv < 0) gv = 0; if (gv >= gridH) gv = gridH - 1;
      grid[gv * gridW + gu]++;
    }
    let mx = 0;
    for (let j = 0; j < grid.length; j++) if (grid[j] > mx) mx = grid[j];
    if (mx > globalMaxDensity) globalMaxDensity = mx;
    grids.push(grid);
  }

  // Render canvases with consistent scale
  const slices = [];
  for (let s = 0; s < numSlices; s++) {
    const slicePos = sweepMin + (s + 0.5) * sliceThickness;
    const canvas = document.createElement('canvas');
    canvas.width = gridW;
    canvas.height = gridH;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(gridW, gridH);
    const grid = grids[s];
    for (let j = 0; j < gridH; j++) {
      for (let i = 0; i < gridW; i++) {
        const density = grid[j * gridW + i];
        const t = globalMaxDensity > 0 ? density / globalMaxDensity : 0;
        const c = turboColormapArr(t);
        const px = ((gridH - 1 - j) * gridW + i) * 4; // flip Y
        if (density === 0) {
          imgData.data[px] = 15; imgData.data[px+1] = 15; imgData.data[px+2] = 25; imgData.data[px+3] = 255;
        } else {
          imgData.data[px] = c[0]; imgData.data[px+1] = c[1]; imgData.data[px+2] = c[2]; imgData.data[px+3] = 255;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
    slices.push({
      canvas, position: slicePos, pointCount: sliceBins[s].length,
      gridW, gridH,
    });
  }

  sweepResults = {
    axis, axisNames, ax1, ax2, numSlices, sliceThickness,
    gridW, gridH, globalMaxDensity, slices,
    u_min, u_max, v_min, v_max,
  };
  sweepSelectedIdx = 0;

  document.getElementById('sweepStatus').textContent =
    numSlices + ' slices generated. Max density: ' + globalMaxDensity + ' pts/cell.';
  showHeatmapOverlay();
}

function turboColormapArr(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 0.13572138 + t*(4.6153926 + t*(-42.66032258 + t*(132.13108234 + t*(-152.54895899 + t*58.9161376))))));
  const g = Math.max(0, Math.min(1, 0.09140261 + t*(2.19418839 + t*(4.84296658 + t*(-14.18503333 + t*(4.27729857 + t*2.82956604))))));
  const b = Math.max(0, Math.min(1, 0.1066733 + t*(12.64194608 + t*(-60.58204836 + t*(110.36276771 + t*(-89.90310912 + t*27.34824973))))));
  return [Math.round(r*255), Math.round(g*255), Math.round(b*255)];
}

function showHeatmapOverlay() {
  if (!sweepResults) return;
  const overlay = document.getElementById('heatmapOverlay');
  overlay.classList.add('visible');

  const sr = sweepResults;
  const axNames = ['X','Y','Z'];
  document.getElementById('hmTitle').textContent =
    'Density Heatmaps - ' + sr.numSlices + ' ' + axNames[sr.axis] +
    ' slices (' + axNames[sr.ax1] + ' vs ' + axNames[sr.ax2] + ')';

  // Build filmstrip
  const filmstrip = document.getElementById('hmFilmstrip');
  filmstrip.innerHTML = '';
  const thumbSize = 80;
  for (let s = 0; s < sr.slices.length; s++) {
    const sl = sr.slices[s];
    const div = document.createElement('div');
    div.className = 'hm-thumb' + (s === sweepSelectedIdx ? ' selected' : '');
    div.dataset.idx = s;
    const tc = document.createElement('canvas');
    const aspect = sl.gridH / sl.gridW;
    tc.width = thumbSize;
    tc.height = Math.max(1, Math.round(thumbSize * aspect));
    tc.style.width = thumbSize + 'px';
    tc.style.height = tc.height + 'px';
    const tctx = tc.getContext('2d');
    tctx.imageSmoothingEnabled = false;
    tctx.drawImage(sl.canvas, 0, 0, tc.width, tc.height);
    div.appendChild(tc);
    const lbl = document.createElement('div');
    lbl.className = 'hm-thumb-label';
    lbl.textContent = axNames[sr.axis] + '=' + sl.position.toFixed(2);
    div.appendChild(lbl);
    div.addEventListener('click', () => selectHeatmapSlice(s));
    filmstrip.appendChild(div);
  }

  selectHeatmapSlice(sweepSelectedIdx);
}

function selectHeatmapSlice(idx) {
  if (!sweepResults) return;
  sweepSelectedIdx = idx;
  const sr = sweepResults;
  const sl = sr.slices[idx];
  const axNames = ['X','Y','Z'];

  // Update detail view
  const detail = document.getElementById('hmDetail');
  detail.innerHTML = '';
  const bigCanvas = document.createElement('canvas');
  const displayW = 400;
  const aspect = sl.gridH / sl.gridW;
  const displayH = Math.max(1, Math.round(displayW * aspect));
  bigCanvas.width = displayW;
  bigCanvas.height = displayH;
  bigCanvas.style.width = displayW + 'px';
  bigCanvas.style.height = displayH + 'px';
  const bctx = bigCanvas.getContext('2d');
  bctx.imageSmoothingEnabled = false;
  bctx.drawImage(sl.canvas, 0, 0, displayW, displayH);
  detail.appendChild(bigCanvas);
  // Marker tool: wire canvas for marker interaction
  hmDetailCanvas = bigCanvas;
  bigCanvas.style.cursor = hmMarkerMode ? 'crosshair' : '';
  bigCanvas.addEventListener('mousemove', onHmDetailMouseMove);
  bigCanvas.addEventListener('click', onHmDetailClick);
  bigCanvas.addEventListener('mouseleave', () => {
    if (hmMarkerMode) redrawHmDetail();
    const readout = document.getElementById('hmCoordReadout');
    if (readout) readout.textContent = '';
  });
  redrawHmDetail();

  const info = document.createElement('div');
  info.className = 'hm-detail-info';
  info.innerHTML =
    '<b>Slice ' + (idx+1) + '/' + sr.numSlices + '</b><br>' +
    axNames[sr.axis] + ' = ' + sl.position.toFixed(3) + ' m<br>' +
    'Thickness: ' + sr.sliceThickness.toFixed(4) + ' m<br>' +
    'Points in slice: ' + sl.pointCount.toLocaleString() + '<br>' +
    'Grid: ' + sl.gridW + ' x ' + sl.gridH + ' px<br>' +
    'Max density: ' + sr.globalMaxDensity + ' pts/cell<br><br>' +
    '<b>Axes:</b><br>' +
    'Horizontal: ' + axNames[sr.ax1] + ' [' + sr.u_min.toFixed(2) + ', ' + sr.u_max.toFixed(2) + '] m<br>' +
    'Vertical: ' + axNames[sr.ax2] + ' [' + sr.v_min.toFixed(2) + ', ' + sr.v_max.toFixed(2) + '] m';
  detail.appendChild(info);

  // Update filmstrip selection — scroll only within filmstrip container
  const filmstrip = document.getElementById('hmFilmstrip');
  document.querySelectorAll('#hmFilmstrip .hm-thumb').forEach(t => {
    const isSelected = parseInt(t.dataset.idx) === idx;
    t.classList.toggle('selected', isSelected);
    if (isSelected) {
      const ft = filmstrip.getBoundingClientRect();
      const tt = t.getBoundingClientRect();
      if (tt.top < ft.top || tt.bottom > ft.bottom) {
        filmstrip.scrollTop += (tt.top - ft.top) - ft.height / 2 + tt.height / 2;
      }
    }
  });
}

function downloadCurrentHeatmap() {
  if (!sweepResults) return;
  const sl = sweepResults.slices[sweepSelectedIdx];
  const axNames = ['x','y','z'];
  const a = document.createElement('a');
  a.href = sl.canvas.toDataURL('image/png');
  a.download = 'heatmap_' + axNames[sweepResults.axis] + '_' + sl.position.toFixed(3) + 'm.png';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

function downloadAllHeatmaps() {
  if (!sweepResults) return;
  const axNames = ['x','y','z'];
  sweepResults.slices.forEach((sl, i) => {
    setTimeout(() => {
      const a = document.createElement('a');
      a.href = sl.canvas.toDataURL('image/png');
      a.download = 'heatmap_' + axNames[sweepResults.axis] + '_' +
        String(i).padStart(3,'0') + '_' + sl.position.toFixed(3) + 'm.png';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
    }, i * 100);
  });
}

/* ── Heatmap Marker Tool ── */

let hmMarkerMode = false;
const hmMarkers = [];  // {canvasPx: [cx,cy], sliceIdx: int, worldPos: [x,y,z], mesh: THREE.Mesh}
let hmDetailCanvas = null;

function orientRingToAxis(mesh, axis) {
  mesh.rotation.set(0, 0, 0);
  if (axis === 0) mesh.rotation.y = Math.PI / 2;
  else if (axis === 1) mesh.rotation.x = Math.PI / 2;
}

function toggleHmMarkerMode() {
  hmMarkerMode = !hmMarkerMode;
  const btn = document.getElementById('hmMarkerBtn');
  if (hmMarkerMode) {
    btn.style.background = '#4fc3f7';
    btn.style.color = '#000';
  } else {
    btn.style.background = '#444';
    btn.style.color = '#eee';
  }
  if (hmDetailCanvas) {
    hmDetailCanvas.style.cursor = hmMarkerMode ? 'crosshair' : '';
  }
}

function hmMarkerRadius() {
  if (!sweepResults) return 5;
  const sr = sweepResults;
  const displayW = hmDetailCanvas ? hmDetailCanvas.width : 400;
  const displayH = hmDetailCanvas ? hmDetailCanvas.height : 400;
  const radiusWorld = 0.08;  // 16cm diameter = 8cm radius
  const rU = radiusWorld / (sr.u_max - sr.u_min) * displayW;
  const rV = radiusWorld / (sr.v_max - sr.v_min) * displayH;
  return (rU + rV) / 2;
}

function hmCanvasToWorld(px, py) {
  if (!sweepResults || !hmDetailCanvas) return null;
  const sr = sweepResults;
  const displayW = hmDetailCanvas.width;
  const displayH = hmDetailCanvas.height;
  // Direct mapping: left→u_min, right→u_max; top→v_max, bottom→v_min (Y flipped)
  const u = sr.u_min + (px / displayW) * (sr.u_max - sr.u_min);
  const v = sr.v_max - (py / displayH) * (sr.v_max - sr.v_min);
  const world = [0, 0, 0];
  world[sr.axis] = sr.slices[sweepSelectedIdx].position;
  world[sr.ax1] = u;
  world[sr.ax2] = v;
  return world;
}

function redrawHmDetail() {
  if (!hmDetailCanvas || !sweepResults) return;
  const sr = sweepResults;
  const sl = sr.slices[sweepSelectedIdx];
  const ctx = hmDetailCanvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(sl.canvas, 0, 0, hmDetailCanvas.width, hmDetailCanvas.height);
  // Draw placed markers on this slice
  const r = hmMarkerRadius();
  for (const m of hmMarkers) {
    if (m.sliceIdx === sweepSelectedIdx) {
      ctx.beginPath();
      ctx.arc(m.canvasPx[0], m.canvasPx[1], r, 0, Math.PI * 2);
      ctx.strokeStyle = '#ff3333';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

function hmCanvasCoords(event) {
  // Account for CSS border (1px) when mapping clientX/Y to canvas pixels
  const rect = hmDetailCanvas.getBoundingClientRect();
  const border = 1; // matches CSS: border: 1px solid #555
  const contentW = rect.width - border * 2;
  const contentH = rect.height - border * 2;
  const cx = (event.clientX - rect.left - border) / contentW * hmDetailCanvas.width;
  const cy = (event.clientY - rect.top - border) / contentH * hmDetailCanvas.height;
  return [cx, cy];
}

function onHmDetailMouseMove(event) {
  if (!hmDetailCanvas || !sweepResults) return;
  const [cx, cy] = hmCanvasCoords(event);
  // Update coordinate readout always (even when not in marker mode)
  const world = hmCanvasToWorld(cx, cy);
  if (world) {
    const axNames = ['X','Y','Z'];
    const readout = document.getElementById('hmCoordReadout');
    if (readout) readout.textContent = 'Cursor: ' + axNames[0] + '=' + world[0].toFixed(3) + '  ' + axNames[1] + '=' + world[1].toFixed(3) + '  ' + axNames[2] + '=' + world[2].toFixed(3);
  }
  if (!hmMarkerMode) return;
  redrawHmDetail();
  // Draw cursor circle
  const r = hmMarkerRadius();
  const ctx = hmDetailCanvas.getContext('2d');
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function onHmDetailClick(event) {
  if (!hmMarkerMode || !hmDetailCanvas || !sweepResults) return;
  const [cx, cy] = hmCanvasCoords(event);
  const world = hmCanvasToWorld(cx, cy);
  const sr = sweepResults;
  console.log('[Marker] canvas=(' + cx.toFixed(1) + ',' + cy.toFixed(1) + ')/' + hmDetailCanvas.width + 'x' + hmDetailCanvas.height +
    ' world=(' + world[0].toFixed(3) + ',' + world[1].toFixed(3) + ',' + world[2].toFixed(3) + ')' +
    ' extent: ax' + sr.ax1 + '=[' + sr.u_min.toFixed(2) + ',' + sr.u_max.toFixed(2) + '] ax' + sr.ax2 + '=[' + sr.v_min.toFixed(2) + ',' + sr.v_max.toFixed(2) + ']');
  if (!world) return;
  // Create highly visible 3D marker group: filled disc + tall vertical post + crosshairs
  const group = new THREE.Group();
  group.position.set(world[0], world[1], world[2]);
  // Filled disc (16cm diameter, solid)
  const discGeo = new THREE.CircleGeometry(0.08, 64);
  const discMat = new THREE.MeshBasicMaterial({ color: 0xff2222, side: THREE.DoubleSide, depthTest: false, transparent: true, opacity: 0.7 });
  const disc = new THREE.Mesh(discGeo, discMat);
  disc.renderOrder = 998;
  orientRingToAxis(disc, sweepResults.axis);
  group.add(disc);
  // Bright ring outline on top
  const ringGeo = new THREE.RingGeometry(0.07, 0.09, 64);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.DoubleSide, depthTest: false });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.renderOrder = 999;
  orientRingToAxis(ring, sweepResults.axis);
  group.add(ring);
  // Tall vertical post (±1m so visible from any angle)
  const postPts = [new THREE.Vector3(0, 0, -1.0), new THREE.Vector3(0, 0, 1.0)];
  const postGeo = new THREE.BufferGeometry().setFromPoints(postPts);
  const postMat = new THREE.LineBasicMaterial({ color: 0xff0000, depthTest: false });
  const post = new THREE.Line(postGeo, postMat);
  post.renderOrder = 999;
  group.add(post);
  // Horizontal crosshair lines through center (±0.12m)
  const crossPts = [
    new THREE.Vector3(-0.12, 0, 0), new THREE.Vector3(0.12, 0, 0),
    new THREE.Vector3(0, -0.12, 0), new THREE.Vector3(0, 0.12, 0),
  ];
  const crossGeo = new THREE.BufferGeometry().setFromPoints(crossPts);
  const crossMat = new THREE.LineBasicMaterial({ color: 0xffff00, depthTest: false });
  const cross = new THREE.LineSegments(crossGeo, crossMat);
  cross.renderOrder = 999;
  orientRingToAxis(cross, sweepResults.axis);
  group.add(cross);
  scene.add(group);
  hmMarkers.push({ canvasPx: [cx, cy], sliceIdx: sweepSelectedIdx, worldPos: world, mesh: group });
  updateHmMarkerCount();
  redrawHmDetail();
}

function clearHmMarkers() {
  for (const m of hmMarkers) {
    scene.remove(m.mesh);
    m.mesh.traverse(child => { if (child.geometry) child.geometry.dispose(); if (child.material) child.material.dispose(); });
  }
  hmMarkers.length = 0;
  updateHmMarkerCount();
  redrawHmDetail();
}

function updateHmMarkerCount() {
  const el = document.getElementById('hmMarkerCount');
  el.textContent = hmMarkers.length > 0 ? hmMarkers.length + ' marker' + (hmMarkers.length > 1 ? 's' : '') : '';
}

/* ── Init UI ── */

function initUi() {
  // Populate dataset selector
  const datasetSelect = document.getElementById('datasetSelect');
  DATASETS.forEach((ds, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = ds.name;
    datasetSelect.appendChild(opt);
  });
  datasetSelect.addEventListener('change', () => {
    activeDatasetIndex = parseInt(datasetSelect.value, 10);
    slicePreviewEnabled = false;
    currentSliceSelection = null;
    document.getElementById('slicePreviewBtn').classList.remove('active');
    document.getElementById('sliceStatus').textContent = 'No slice active.';
    const ds = getActiveDataset();
    syncColorModeButtons(ds);
    resetSweepBounds();
    updateSlicePositionLabel();
    applyDisplay(true);
  });
  document.querySelectorAll('#colormode button').forEach(btn => {
    btn.addEventListener('click', () => setColorMode(btn.dataset.mode));
  });
  document.getElementById('pointsizeRange').addEventListener('input', e => {
    material.size = parseFloat(e.target.value);
    document.getElementById('psval').textContent = e.target.value;
  });
  document.getElementById('bgRange').addEventListener('input', e => {
    const v = parseInt(e.target.value);
    scene.background = new THREE.Color(v/255, v/255, v/255);
    document.getElementById('bgval').textContent = String(v);
  });
  document.getElementById('sliceAxis').addEventListener('change', () => { updateSlicePositionLabel(); if (slicePreviewEnabled) previewSlice(); });
  document.getElementById('slicePosRange').addEventListener('input', () => { updateSlicePositionLabel(); if (slicePreviewEnabled) previewSlice(); });
  document.getElementById('sliceThickness').addEventListener('change', () => { if (slicePreviewEnabled) previewSlice(); });
  document.getElementById('slicePreviewBtn').addEventListener('click', previewSlice);
  document.getElementById('sliceClearBtn').addEventListener('click', clearSlicePreview);
  document.getElementById('sliceDownloadBtn').addEventListener('click', downloadSliceCsv);
  // Heatmap sweep
  document.getElementById('sweepRunBtn').addEventListener('click', runHeatmapSweep);
  document.getElementById('sweepResetBoundsBtn').addEventListener('click', resetSweepBounds);
  document.getElementById('hmCloseBtn').addEventListener('click', () => {
    document.getElementById('heatmapOverlay').classList.remove('visible');
  });
  document.getElementById('hmPrevBtn').addEventListener('click', () => {
    if (sweepResults && sweepSelectedIdx > 0) selectHeatmapSlice(sweepSelectedIdx - 1);
  });
  document.getElementById('hmNextBtn').addEventListener('click', () => {
    if (sweepResults && sweepSelectedIdx < sweepResults.slices.length - 1) selectHeatmapSlice(sweepSelectedIdx + 1);
  });
  document.getElementById('hmDlCurrent').addEventListener('click', downloadCurrentHeatmap);
  document.getElementById('hmDlAll').addEventListener('click', downloadAllHeatmaps);
  // Heatmap marker tool
  document.getElementById('hmMarkerBtn').addEventListener('click', toggleHmMarkerMode);
  document.getElementById('hmClearMarkersBtn').addEventListener('click', clearHmMarkers);
  // Initialize bounding box inputs from dataset
  resetSweepBounds();
}

function resetSweepBounds() {
  const ds = getActiveDataset();
  document.getElementById('sweepXmin').value = ds.meta.mins[0].toFixed(2);
  document.getElementById('sweepXmax').value = ds.meta.maxs[0].toFixed(2);
  document.getElementById('sweepYmin').value = ds.meta.mins[1].toFixed(2);
  document.getElementById('sweepYmax').value = ds.meta.maxs[1].toFixed(2);
  document.getElementById('sweepZmin').value = ds.meta.mins[2].toFixed(2);
  document.getElementById('sweepZmax').value = ds.meta.maxs[2].toFixed(2);
}

window.addEventListener('keydown', e => {
  if (e.key === 'r' || e.key === 'R') applyDisplay(true);
  // Arrow keys for heatmap overlay navigation
  const overlayVisible = document.getElementById('heatmapOverlay').classList.contains('visible');
  if (overlayVisible && sweepResults) {
    if (e.key === 'ArrowLeft' && sweepSelectedIdx > 0) {
      e.preventDefault();
      selectHeatmapSlice(sweepSelectedIdx - 1);
    } else if (e.key === 'ArrowRight' && sweepSelectedIdx < sweepResults.slices.length - 1) {
      e.preventDefault();
      selectHeatmapSlice(sweepSelectedIdx + 1);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      if (hmMarkerMode) { toggleHmMarkerMode(); }
      else { document.getElementById('heatmapOverlay').classList.remove('visible'); }
    }
  }
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── 3D cursor coordinate tracker ──
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.05;
const mouse = new THREE.Vector2();
let cursorThrottle = 0;

renderer.domElement.addEventListener('mousemove', function(event) {
  const now = Date.now();
  if (now - cursorThrottle < 50) return;  // 20 Hz max
  cursorThrottle = now;

  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const pts = scene.children.filter(c => c.isPoints);
  const intersects = raycaster.intersectObjects(pts);

  const el = document.getElementById('cursorCoords');
  if (intersects.length > 0) {
    const p = intersects[0].point;
    el.textContent = 'X: ' + p.x.toFixed(3) + '  Y: ' + p.y.toFixed(3) + '  Z: ' + p.z.toFixed(3);
    el.style.color = '#4fc3f7';
  } else {
    el.textContent = 'X: \u2014  Y: \u2014  Z: \u2014';
    el.style.color = '#666';
  }
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

initUi();
const ds0 = getActiveDataset();
syncColorModeButtons(ds0);
updateSlicePositionLabel();
applyDisplay(true);
animate();
</script>
</body>
</html>'''
    return html.replace("__DATASETS_JSON__", datasets_json)


# ─── Main ───────────────────────────────────────────────────────────

def colorize_and_pack(name, pts, poses, images, cam_K, cam_dist, cam_w, cam_h,
                      Rcl, Pcl, ext_R, ext_T, display_count, total_points=None,
                      intensities=None):
    """Colorize a point cloud and pack it as a viewer dataset.

    Sorts colored points first for optimal display via setDrawRange.
    """
    total = total_points or len(pts)
    print(f"\nColorizing {len(pts):,} points for '{name}'...")
    colors, colored_mask = colorize_points(
        pts, poses, images, cam_K, cam_dist, cam_w, cam_h,
        Rcl, Pcl, ext_R, ext_T
    )
    dc = min(display_count, len(pts))
    pts, colors, colored_mask, intensities = sort_colored_first(
        pts, colors, colored_mask, dc, intensities=intensities)
    return pack_viewer_dataset(
        name=name, pts=pts, rgb=colors, rgb_mask=colored_mask,
        total_points=total, display_count=dc, intensities=intensities,
    )


def main():
    parser = argparse.ArgumentParser(description='Colorize point cloud from camera images')
    parser.add_argument('--pcd', required=True, help='Input PCD file (LIVO2 map)')
    parser.add_argument('--bag', required=True, help='Rosbag with images and LiDAR')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (TUM format)')
    parser.add_argument('--config', required=True, help='Main YAML config (avia.yaml)')
    parser.add_argument('--camera', required=True, help='Camera YAML config')
    parser.add_argument('--output', default=None, help='Output HTML file')
    parser.add_argument('--max-points', type=int, default=500000,
                        help='Max points to display per dataset')
    parser.add_argument('--embed-limit', type=int, default=1500000,
                        help='Max points to embed per dataset (for slicing)')
    parser.add_argument('--voxel-size', type=float, default=0.02,
                        help='Voxel size in meters for downsampling (default: 0.02 = 2cm)')
    parser.add_argument('--min-voxel-pts', type=int, default=1,
                        help='Min points per voxel to keep (default: 1, use 2-5 to filter noise)')
    parser.add_argument('--num-variants', type=int, default=11,
                        help='Number of thickened variants for Mode 3')
    parser.add_argument('--max-shift-ms', type=float, default=1.0,
                        help='Max IMU time shift in ms for Mode 3')
    parser.add_argument('--skip-thickened', action='store_true',
                        help='Skip Mode 3 (re-runs SLAM pipeline N times)')
    parser.add_argument('--align-ground', action='store_true',
                        help='Align point cloud so ground plane is horizontal (Z-up)')
    args = parser.parse_args()

    display_count = args.max_points
    embed_limit = args.embed_limit

    # ── Load configs ─────────────────────────────────────────────
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(args.camera, 'r') as f:
        cam_cfg = yaml.safe_load(f)

    # Extrinsics
    ext = cfg.get('extrin_calib', {})
    ext_R = np.array(ext.get('extrinsic_R', [1,0,0,0,1,0,0,0,1]),
                     dtype=np.float64).reshape(3, 3)
    ext_T = np.array(ext.get('extrinsic_T', [0,0,0]), dtype=np.float64)
    Rcl = np.array(ext.get('Rcl', [1,0,0,0,1,0,0,0,1]),
                   dtype=np.float64).reshape(3, 3)
    Pcl = np.array(ext.get('Pcl', [0,0,0]), dtype=np.float64)

    img_time_offset = cfg.get('time_offset', {}).get('img_time_offset', 0.0)
    img_topic = cfg.get('common', {}).get('img_topic', '/left_camera/image')
    lid_topic = cfg.get('common', {}).get('lid_topic', '/livox/lidar')
    imu_topic = cfg.get('common', {}).get('imu_topic', '/livox/imu')
    blind = cfg.get('preprocess', {}).get('blind', 0.8)

    # Camera intrinsics
    cam_w = int(cam_cfg.get('cam_width', 1280))
    cam_h = int(cam_cfg.get('cam_height', 1024))
    fx = float(cam_cfg.get('cam_fx', 0))
    fy = float(cam_cfg.get('cam_fy', 0))
    cx = float(cam_cfg.get('cam_cx', 0))
    cy = float(cam_cfg.get('cam_cy', 0))
    scale = float(cam_cfg.get('scale', 1.0))
    d0 = float(cam_cfg.get('cam_d0', 0))
    d1 = float(cam_cfg.get('cam_d1', 0))
    d2 = float(cam_cfg.get('cam_d2', 0))
    d3 = float(cam_cfg.get('cam_d3', 0))

    print(f"[Config] Camera: {cam_w}x{cam_h}, fx={fx:.1f}, fy={fy:.1f}, scale={scale}")
    print(f"[Config] Rcl:\n{Rcl}")
    print(f"[Config] Pcl: {Pcl}")
    print(f"[Config] ext_R:\n{ext_R}")
    print(f"[Config] ext_T: {ext_T}")
    print(f"[Config] img_topic: {img_topic}, lid_topic: {lid_topic}")

    # ── Shared data ──────────────────────────────────────────────
    print(f"\nReading trajectory: {args.trajectory}")
    poses = read_trajectory(args.trajectory)
    print(f"  {len(poses)} poses")

    print(f"\nExtracting images from: {args.bag}")
    images = extract_images(args.bag, img_topic, img_time_offset)
    has_images = len(images) > 0
    if not has_images:
        print("WARNING: No images found — viewer will use Height/Intensity only (no RGB).")

    if has_images:
        actual_h, actual_w = images[0][1].shape[:2]
        print(f"  Actual image size: {actual_w}x{actual_h}")

        if actual_w != cam_w or actual_h != cam_h:
            if abs(actual_w - cam_w * scale) < 2 and abs(actual_h - cam_h * scale) < 2:
                print(f"  Applying scale={scale} to intrinsics")
                fx *= scale; fy *= scale; cx *= scale; cy *= scale
                cam_w = actual_w; cam_h = actual_h
            else:
                sx = actual_w / cam_w; sy = actual_h / cam_h
                print(f"  Scaling intrinsics by ({sx:.3f}, {sy:.3f})")
                fx *= sx; fy *= sy; cx *= sx; cy *= sy
                cam_w = actual_w; cam_h = actual_h

    cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    cam_dist_coeffs = np.array([d0, d1, d2, d3], dtype=np.float64)
    if has_images:
        print(f"  Final intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"  Working resolution: {cam_w}x{cam_h}")

    cam_args = (cam_K, cam_dist_coeffs, cam_w, cam_h, Rcl, Pcl, ext_R, ext_T)

    datasets = []

    # ── Mode 2: LIVO2 Map (existing map.pcd) ────────────────────
    print(f"\n{'='*60}")
    print("Mode 2: LIVO2 Map")
    print(f"{'='*60}")
    pts_map = read_pcd_ascii(args.pcd)
    print(f"  {len(pts_map):,} points from map.pcd")
    # map.pcd has no intensity data
    intens_map = np.zeros(len(pts_map), dtype=np.float32)

    # Colorize once — reuse for both individual dataset and overlay
    print(f"\nColorizing {len(pts_map):,} points for 'LIVO2 Map'...")
    colors_map, mask_map = colorize_points(pts_map, poses, images, *cam_args)

    # Ground plane alignment — compute rotation from LIVO2 Map (most points)
    R_align = None
    z_shift = 0.0
    if args.align_ground:
        R_align = align_to_ground_plane(pts_map, bag_path=args.bag)
        tilt_deg = np.degrees(np.arccos(np.clip(R_align[2, 2], -1, 1)))
        print(f"  Ground alignment: {tilt_deg:.1f}° total correction")
        pts_map = (R_align @ pts_map.T).T
        # Shift Z so the densest band (ground) is at Z=0
        z_vals = pts_map[:, 2]
        z_candidates = np.linspace(z_vals.min(), z_vals.max(), 200)
        ht = 0.02
        best_z = 0.0
        best_c = 0
        for zc in z_candidates:
            c = np.sum((z_vals >= zc - ht) & (z_vals < zc + ht))
            if c > best_c:
                best_c = c
                best_z = zc
        z_shift = -best_z
        pts_map[:, 2] += z_shift
        print(f"  Z-shift: {z_shift:.2f}m (ground now at Z=0)")

    total_map = len(pts_map)
    voxel_size = args.voxel_size
    min_vp = args.min_voxel_pts
    if voxel_size > 0 and (total_map > embed_limit or min_vp > 1):
        idx = voxel_downsample(pts_map, voxel_size, max_points=embed_limit, min_count=min_vp)
        print(f"  Voxel downsample ({voxel_size*100:.0f}cm, >={min_vp}pts): {total_map:,} -> {len(idx):,}")
        pts_map = pts_map[idx]
        colors_map = colors_map[idx]
        mask_map = mask_map[idx]
        intens_map = intens_map[idx]
    dc = min(display_count, len(pts_map))
    pts_s, colors_s, mask_s, intens_s = sort_colored_first(
        pts_map.copy(), colors_map.copy(), mask_map.copy(), dc,
        intensities=intens_map.copy())
    ds_map = pack_viewer_dataset("LIVO2 Map", pts_s, colors_s, mask_s,
                                  total_points=total_map, display_count=dc,
                                  intensities=intens_s)
    datasets.append(ds_map)

    # ── Mode 0: Raw LiDAR Cloud (sensor frame) ─────────────────
    print(f"\n{'='*60}")
    print("Mode 0: Raw LiDAR Cloud (sensor frame)")
    print(f"{'='*60}")
    lidar_scans = extract_lidar_scans(args.bag, lid_topic, imu_topic)
    pts_raw_full, intens_raw_full = build_raw_lidar_cloud(lidar_scans, blind=blind)
    total_raw = len(pts_raw_full)

    # Save raw cloud to PCD file
    output_dir = os.path.dirname(
        args.output or os.path.join(os.path.dirname(args.pcd), 'viewer.html'))
    raw_pcd_path = os.path.join(output_dir, 'raw_lidar.pcd')
    _save_pcd_simple(raw_pcd_path, pts_raw_full)
    print(f"  Saved raw LiDAR cloud: {raw_pcd_path} ({os.path.getsize(raw_pcd_path)/1e6:.1f} MB)")

    # Downsample for viewer
    if voxel_size > 0 and (total_raw > embed_limit or min_vp > 1):
        idx = voxel_downsample(pts_raw_full, voxel_size, max_points=embed_limit, min_count=min_vp)
        pts_raw = pts_raw_full[idx]
        intens_raw = intens_raw_full[idx]
        print(f"  Voxel downsample ({voxel_size*100:.0f}cm, >={min_vp}pts): {total_raw:,} -> {len(pts_raw):,}")
    else:
        pts_raw = pts_raw_full
        intens_raw = intens_raw_full

    # No colorization for raw cloud — no poses to project from
    colors_raw = np.zeros((len(pts_raw), 3), dtype=np.uint8)
    mask_raw = np.zeros(len(pts_raw), dtype=bool)

    dc = min(display_count, len(pts_raw))
    pts_s, colors_s, mask_s, intens_s = sort_colored_first(
        pts_raw.copy(), colors_raw.copy(), mask_raw.copy(), dc,
        intensities=intens_raw.copy())
    ds_raw = pack_viewer_dataset("Raw LiDAR (sensor frame)", pts_s, colors_s, mask_s,
                                  total_points=total_raw, display_count=dc,
                                  intensities=intens_s)
    datasets.append(ds_raw)

    # ── Mode 1: Odometry Cloud (raw scans + odometry poses) ────
    print(f"\n{'='*60}")
    print("Mode 1: Odometry Cloud (raw scans + odometry poses)")
    print(f"{'='*60}")
    pts_odom_full, intens_odom_full = build_odometry_cloud(
        lidar_scans, poses, ext_R, ext_T, blind=blind)
    total_odom = len(pts_odom_full)

    # Save odometry cloud to PCD file
    odom_pcd_path = os.path.join(output_dir, 'odometry_cloud.pcd')
    _save_pcd_simple(odom_pcd_path, pts_odom_full)
    print(f"  Saved odometry cloud: {odom_pcd_path} ({os.path.getsize(odom_pcd_path)/1e6:.1f} MB)")

    # Voxel downsample if over embed limit or filtering by density
    if voxel_size > 0 and (total_odom > embed_limit or min_vp > 1):
        idx = voxel_downsample(pts_odom_full, voxel_size, max_points=embed_limit, min_count=min_vp)
        pts_odom = pts_odom_full[idx]
        intens_odom = intens_odom_full[idx]
        print(f"  Voxel downsample ({voxel_size*100:.0f}cm, >={min_vp}pts): {total_odom:,} -> {len(pts_odom):,}")
    else:
        pts_odom = pts_odom_full
        intens_odom = intens_odom_full

    # Colorize
    print(f"\nColorizing {len(pts_odom):,} points for 'Odometry Cloud'...")
    colors_odom, mask_odom = colorize_points(pts_odom, poses, images, *cam_args)

    if R_align is not None:
        pts_odom = (R_align @ pts_odom.T).T
        pts_odom[:, 2] += z_shift

    dc = min(display_count, len(pts_odom))
    pts_s, colors_s, mask_s, intens_s = sort_colored_first(
        pts_odom.copy(), colors_odom.copy(), mask_odom.copy(), dc,
        intensities=intens_odom.copy())
    ds_odom = pack_viewer_dataset("Odometry Cloud", pts_s, colors_s, mask_s,
                                   total_points=total_odom, display_count=dc,
                                   intensities=intens_s)
    datasets.append(ds_odom)

    # ── Mode 3: Thickened (Time-Shifted Variants) ───────────────
    if not args.skip_thickened:
        print(f"\n{'='*60}")
        print(f"Mode 3: Thickened ({args.num_variants} variants, "
              f"+/-{args.max_shift_ms:.1f}ms)")
        print(f"{'='*60}")
        output_dir = os.path.dirname(
            args.output or os.path.join(os.path.dirname(args.pcd), 'viewer.html'))
        thickened_dir = os.path.join(output_dir, 'thickened_variants')

        variants = run_thickened_variants(
            args.bag, args.config, args.camera,
            thickened_dir, args.num_variants, args.max_shift_ms / 1000.0)

        if variants:
            all_variant_pts = np.concatenate([v[1] for v in variants], axis=0)
            print(f"  Combined: {len(all_variant_pts):,} points "
                  f"from {len(variants)} variants")
            # Downsample combined to 2M max
            max_thickened = 2_000_000
            if len(all_variant_pts) > max_thickened:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(all_variant_pts), max_thickened, replace=False)
                idx.sort()
                all_variant_pts = all_variant_pts[idx]
                print(f"  Downsampled to {len(all_variant_pts):,} points")

            if R_align is not None:
                all_variant_pts = (R_align @ all_variant_pts.T).T

            ds_thick = colorize_and_pack(
                "Thickened (Time-Shifted)", all_variant_pts, poses, images,
                *cam_args, display_count=display_count,
                total_points=len(all_variant_pts))
            datasets.append(ds_thick)
        else:
            print("  No variants produced, skipping Mode 3")

    # ── Mode 4: Overlay (Odometry + LIVO2 Map) ──────────────────
    print(f"\n{'='*60}")
    print("Mode 4: Overlay (Odom + LIVO2)")
    print(f"{'='*60}")
    pts_overlay = np.concatenate([pts_map, pts_odom], axis=0)
    colors_overlay = np.concatenate([colors_map, colors_odom], axis=0)
    mask_overlay = np.concatenate([mask_map, mask_odom], axis=0)
    intens_overlay = np.concatenate([intens_map, intens_odom], axis=0)
    total_overlay_raw = len(pts_overlay)

    # Voxel downsample overlay if over embed limit or filtering by density
    if voxel_size > 0 and (total_overlay_raw > embed_limit or min_vp > 1):
        idx = voxel_downsample(pts_overlay, voxel_size, max_points=embed_limit, min_count=min_vp)
        pts_overlay = pts_overlay[idx]
        colors_overlay = colors_overlay[idx]
        mask_overlay = mask_overlay[idx]
        intens_overlay = intens_overlay[idx]
        print(f"  Voxel downsample ({voxel_size*100:.0f}cm, >={min_vp}pts): {total_overlay_raw:,} -> {len(pts_overlay):,}")

    total_overlay = len(pts_overlay)
    dc_overlay = min(display_count, total_overlay)
    pts_overlay, colors_overlay, mask_overlay, intens_overlay = sort_colored_first(
        pts_overlay, colors_overlay, mask_overlay, dc_overlay,
        intensities=intens_overlay)
    ds_overlay = pack_viewer_dataset(
        name="Overlay (Odom + LIVO2)",
        pts=pts_overlay, rgb=colors_overlay, rgb_mask=mask_overlay,
        total_points=total_overlay_raw, display_count=dc_overlay,
        intensities=intens_overlay)
    datasets.append(ds_overlay)
    print(f"  {total_overlay_raw:,} total, {total_overlay:,} embedded")

    # ── Generate HTML ────────────────────────────────────────────
    output_path = args.output or os.path.splitext(args.pcd)[0] + '_colored.html'
    print(f"\n{'='*60}")
    print(f"Generating viewer with {len(datasets)} datasets: {output_path}")
    for i, ds in enumerate(datasets):
        m = ds['meta']
        print(f"  [{i}] {ds['name']}: {m['num_points']:,} pts "
              f"(display {m['display_count']:,})")
    html = build_viewer_html(datasets)
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nDone! {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
