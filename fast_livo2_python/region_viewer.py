#!/usr/bin/env python3
"""Region viewer — generate a high-density viewer for a cropped section of a point cloud.

Reads the full map.pcd (millions of points), crops to a specified X/Y/Z range,
then generates a viewer with ALL points in that region (up to embed limit).
This gives much higher density than the default viewer which subsamples the entire scene.

Usage:
    # Crop to a 5m x 5m x 3m box
    python region_viewer.py outputs/output_scan/map.pcd --x 0 5 --y -2 3 --z -1 2

    # Also include odometry cloud
    python region_viewer.py outputs/output_scan/map.pcd \
        --odometry outputs/output_scan/odometry_cloud.pcd \
        --x 0 5 --y -2 3

    # With bag for colorization and ground alignment
    python region_viewer.py outputs/output_scan/map.pcd \
        --bag Bags/scan.bag --config ../FAST-LIVO2-main/config/avia.yaml \
        --camera ../FAST-LIVO2-main/config/camera_pinhole.yaml \
        --x 5 15 --y -3 3

    # Interactive: prints bounds and lets you pick a region
    python region_viewer.py outputs/output_scan/map.pcd --info
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def read_pcd(path):
    """Read ASCII PCD file -> Nx3 float32."""
    with open(path, 'r', errors='replace') as f:
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
            try:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    return np.array(pts, dtype=np.float32)


def crop_points(pts, x_range=None, y_range=None, z_range=None):
    """Crop points to specified ranges. Returns (cropped_pts, mask)."""
    mask = np.ones(len(pts), dtype=bool)
    if x_range is not None:
        mask &= (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1])
    if y_range is not None:
        mask &= (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1])
    if z_range is not None:
        mask &= (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
    return pts[mask], mask


def main():
    parser = argparse.ArgumentParser(
        description='Generate a high-density viewer for a cropped region of a point cloud.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'EXAMPLES:\n'
            '  python region_viewer.py outputs/output_scan/map.pcd --x 0 5 --y -2 3\n'
            '  python region_viewer.py outputs/output_scan/map.pcd --info\n'
            '  python region_viewer.py outputs/output_scan/map.pcd --odometry outputs/output_scan/odometry_cloud.pcd --x 5 15\n'
        ),
    )
    parser.add_argument('pcd', help='Path to map.pcd (SLAM output)')
    parser.add_argument('--odometry', default=None,
                        help='Path to odometry_cloud.pcd (optional, adds as second dataset)')
    parser.add_argument('--raw', default=None,
                        help='Path to raw_lidar.pcd (optional, adds raw sensor-frame cloud)')
    parser.add_argument('--x', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='X range to crop (meters)')
    parser.add_argument('--y', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Y range to crop (meters)')
    parser.add_argument('--z', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Z range to crop (meters)')
    parser.add_argument('--embed-limit', type=int, default=300000,
                        help='Max points per dataset to embed (default: 300000)')
    parser.add_argument('--voxel-size', type=float, default=0.02,
                        help='Voxel size for downsampling if over embed limit (default: 0.02)')
    parser.add_argument('--output', default=None,
                        help='Output HTML path (default: <pcd_dir>/region_viewer.html)')
    parser.add_argument('--bag', default=None,
                        help='Bag file for colorization and ground alignment')
    parser.add_argument('--config', default=None,
                        help='YAML config for camera/extrinsics')
    parser.add_argument('--camera', default=None,
                        help='Camera YAML config')
    parser.add_argument('--trajectory', default=None,
                        help='Trajectory file for colorization (default: auto-detect from pcd dir)')
    parser.add_argument('--no-open', action='store_true',
                        help="Don't open viewer in browser")
    parser.add_argument('--no-align', action='store_true',
                        help='Skip ground alignment')
    parser.add_argument('--info', action='store_true',
                        help='Print point cloud bounds (after alignment) and exit')

    args = parser.parse_args()

    pcd_path = os.path.abspath(args.pcd)
    if not os.path.isfile(pcd_path):
        print(f'ERROR: PCD file not found: {pcd_path}')
        sys.exit(1)

    pcd_dir = os.path.dirname(pcd_path)

    # Read main point cloud
    print(f'Reading {pcd_path}...')
    import time as _time
    t0 = _time.time()
    pts_map = read_pcd(pcd_path)
    print(f'  Loaded {len(pts_map):,} points in {_time.time()-t0:.1f}s')

    # Ground alignment (before cropping so Z ranges are meaningful)
    R_align = None
    z_shift = 0.0
    if not args.no_align:
        from colorize_cloud import align_to_ground_plane
        # Auto-detect bag for IMU gravity
        bag_for_align = args.bag
        if bag_for_align is None:
            # Try to find a bag in the Bags/ directory
            bags_dir = os.path.join(SCRIPT_DIR, 'Bags')
            if os.path.isdir(bags_dir):
                bags = [f for f in os.listdir(bags_dir) if f.endswith('.bag')]
                if len(bags) == 1:
                    bag_for_align = os.path.join(bags_dir, bags[0])

        R_align = align_to_ground_plane(pts_map, bag_path=bag_for_align)
        tilt_deg = np.degrees(np.arccos(np.clip(R_align[2, 2], -1, 1)))
        print(f'  Ground alignment: {tilt_deg:.1f}° correction')
        pts_map = (R_align @ pts_map.T).T

        # Z-shift: ground at Z=0
        z_vals = pts_map[:, 2]
        z_candidates = np.linspace(z_vals.min(), z_vals.max(), 200)
        ht = 0.02
        best_z, best_c = 0.0, 0
        for zc in z_candidates:
            c = np.sum((z_vals >= zc - ht) & (z_vals < zc + ht))
            if c > best_c:
                best_c = c
                best_z = zc
        z_shift = -best_z
        pts_map[:, 2] += z_shift
        print(f'  Z-shift: {z_shift:.2f}m (ground at Z=0)')

    print(f'  Bounds (after alignment):')
    print(f'    X: [{pts_map[:,0].min():.2f}, {pts_map[:,0].max():.2f}]  '
          f'({pts_map[:,0].max()-pts_map[:,0].min():.1f}m)')
    print(f'    Y: [{pts_map[:,1].min():.2f}, {pts_map[:,1].max():.2f}]  '
          f'({pts_map[:,1].max()-pts_map[:,1].min():.1f}m)')
    print(f'    Z: [{pts_map[:,2].min():.2f}, {pts_map[:,2].max():.2f}]  '
          f'({pts_map[:,2].max()-pts_map[:,2].min():.1f}m)')

    if args.info:
        # Also check odometry and raw if they exist
        for name, fname in [('odometry_cloud.pcd', 'odometry'), ('raw_lidar.pcd', 'raw')]:
            p = os.path.join(pcd_dir, name)
            if os.path.isfile(p):
                sz = os.path.getsize(p) / 1e6
                print(f'\n  {name}: {sz:.1f} MB')
        sys.exit(0)

    if args.x is None and args.y is None and args.z is None:
        print('\nERROR: Specify at least one range (--x, --y, or --z). '
              'Use --info to see bounds.')
        sys.exit(1)

    # Crop
    print(f'\nCropping:')
    if args.x: print(f'  X: [{args.x[0]}, {args.x[1]}]')
    if args.y: print(f'  Y: [{args.y[0]}, {args.y[1]}]')
    if args.z: print(f'  Z: [{args.z[0]}, {args.z[1]}]')

    pts_cropped, mask = crop_points(pts_map, args.x, args.y, args.z)
    print(f'  LIVO2 Map: {len(pts_map):,} -> {len(pts_cropped):,} points in region')

    if len(pts_cropped) == 0:
        print('ERROR: No points in specified region.')
        sys.exit(1)

    # Import colorize_cloud functions
    from colorize_cloud import (voxel_downsample, pack_viewer_dataset,
                                 build_viewer_html, sort_colored_first,
                                 read_trajectory, extract_images, colorize_points)
    import yaml

    # Load configs for colorization if provided
    poses = []
    images = []
    cam_args = None
    has_images = False

    traj_path = args.trajectory
    if traj_path is None:
        auto_traj = os.path.join(pcd_dir, 'trajectory.txt')
        if os.path.isfile(auto_traj):
            traj_path = auto_traj

    if traj_path and os.path.isfile(traj_path):
        poses = read_trajectory(traj_path)
        print(f'  Trajectory: {len(poses)} poses')

    if args.bag and args.config and args.camera:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        with open(args.camera, 'r') as f:
            cam_cfg = yaml.safe_load(f)

        ext = cfg.get('extrin_calib', {})
        ext_R = np.array(ext.get('extrinsic_R', [1,0,0,0,1,0,0,0,1]),
                         dtype=np.float64).reshape(3, 3)
        ext_T = np.array(ext.get('extrinsic_T', [0,0,0]), dtype=np.float64)
        Rcl = np.array(ext.get('Rcl', [1,0,0,0,1,0,0,0,1]),
                       dtype=np.float64).reshape(3, 3)
        Pcl = np.array(ext.get('Pcl', [0,0,0]), dtype=np.float64)

        img_time_offset = cfg.get('time_offset', {}).get('img_time_offset', 0.0)
        img_topic = cfg.get('common', {}).get('img_topic', '/left_camera/image')

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

        images = extract_images(args.bag, img_topic, img_time_offset)
        has_images = len(images) > 0

        if has_images:
            actual_h, actual_w = images[0][1].shape[:2]
            if actual_w != cam_w or actual_h != cam_h:
                if abs(actual_w - cam_w * scale) < 2 and abs(actual_h - cam_h * scale) < 2:
                    fx *= scale; fy *= scale; cx *= scale; cy *= scale
                    cam_w = actual_w; cam_h = actual_h
                else:
                    sx = actual_w / cam_w; sy = actual_h / cam_h
                    fx *= sx; fy *= sy; cx *= sx; cy *= sy
                    cam_w = actual_w; cam_h = actual_h

        cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        cam_dist = np.array([d0, d1, d2, d3], dtype=np.float64)
        cam_args = (cam_K, cam_dist, cam_w, cam_h, Rcl, Pcl, ext_R, ext_T)
        print(f'  Images: {len(images)} frames')

    # Build datasets
    datasets = []
    embed_limit = args.embed_limit
    voxel_size = args.voxel_size
    display_count = embed_limit

    def align_cloud(pts):
        """Apply the same ground alignment as the map."""
        if R_align is not None:
            pts = (R_align @ pts.T).T
            pts[:, 2] += z_shift
        return pts

    def process_cloud(name, pts_full, already_aligned=False):
        """Align, crop, optionally voxel downsample, colorize, pack.

        Colorization uses pre-alignment coordinates (original world frame)
        so that camera projection math works correctly, then the aligned
        coordinates are used for the viewer geometry.
        """
        if not already_aligned:
            # Keep pre-alignment copy for colorization
            pts_orig = pts_full.copy()
            pts_full = align_cloud(pts_full)
        else:
            # Already aligned — reverse to get original coords for colorization
            pts_orig = pts_full.copy()
            if R_align is not None:
                pts_tmp = pts_orig.copy()
                pts_tmp[:, 2] -= z_shift
                pts_orig = (R_align.T @ pts_tmp.T).T

        pts_c, crop_mask = crop_points(pts_full, args.x, args.y, args.z)
        pts_c_orig = pts_orig[crop_mask]  # matching original coords for colorization
        total = len(pts_c)
        print(f'\n  {name}: {total:,} points in region')

        if total == 0:
            return None

        # Voxel downsample if needed
        if voxel_size > 0 and total > embed_limit:
            idx = voxel_downsample(pts_c, voxel_size, max_points=embed_limit)
            pts_c = pts_c[idx]
            pts_c_orig = pts_c_orig[idx]
            print(f'    Voxel downsample: {total:,} -> {len(pts_c):,}')

        # Colorize using original (pre-alignment) coordinates
        if cam_args and poses and has_images:
            print(f'    Colorizing {len(pts_c):,} points...')
            colors, mask = colorize_points(pts_c_orig, poses, images, *cam_args)
        else:
            colors = np.zeros((len(pts_c), 3), dtype=np.uint8)
            mask = np.zeros(len(pts_c), dtype=bool)

        intens = np.zeros(len(pts_c), dtype=np.float32)
        dc = min(display_count, len(pts_c))
        pts_s, colors_s, mask_s, intens_s = sort_colored_first(
            pts_c.copy(), colors.copy(), mask.copy(), dc, intensities=intens.copy())
        ds = pack_viewer_dataset(name, pts_s, colors_s, mask_s,
                                  total_points=total, display_count=dc,
                                  intensities=intens_s)
        return ds

    # LIVO2 Map (already aligned above)
    ds = process_cloud("LIVO2 Map (region)", pts_cropped, already_aligned=True)
    if ds:
        datasets.append(ds)

    # Odometry cloud (optional)
    if args.odometry and os.path.isfile(args.odometry):
        print(f'\nReading odometry cloud: {args.odometry}')
        pts_odom = read_pcd(args.odometry)
        print(f'  {len(pts_odom):,} points')
        ds = process_cloud("Odometry Cloud (region)", pts_odom)
        if ds:
            datasets.append(ds)
    else:
        # Auto-detect
        auto_odom = os.path.join(pcd_dir, 'odometry_cloud.pcd')
        if os.path.isfile(auto_odom):
            print(f'\nReading odometry cloud: {auto_odom}')
            pts_odom = read_pcd(auto_odom)
            print(f'  {len(pts_odom):,} points')
            ds = process_cloud("Odometry Cloud (region)", pts_odom)
            if ds:
                datasets.append(ds)

    # Raw LiDAR (optional)
    if args.raw and os.path.isfile(args.raw):
        print(f'\nReading raw LiDAR cloud: {args.raw}')
        pts_raw = read_pcd(args.raw)
        print(f'  {len(pts_raw):,} points')
        ds = process_cloud("Raw LiDAR (region)", pts_raw)
        if ds:
            datasets.append(ds)

    if not datasets:
        print('ERROR: No datasets to display.')
        sys.exit(1)

    # Generate HTML
    output_path = args.output
    if output_path is None:
        region_str = ''
        if args.x: region_str += f'_x{args.x[0]:.0f}to{args.x[1]:.0f}'
        if args.y: region_str += f'_y{args.y[0]:.0f}to{args.y[1]:.0f}'
        if args.z: region_str += f'_z{args.z[0]:.0f}to{args.z[1]:.0f}'
        output_path = os.path.join(pcd_dir, f'region_viewer{region_str}.html')

    print(f'\nGenerating viewer with {len(datasets)} datasets...')
    for i, ds in enumerate(datasets):
        m = ds['meta']
        print(f'  [{i}] {ds["name"]}: {m["num_points"]:,} pts')

    html = build_viewer_html(datasets)
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\nViewer: {output_path} ({size_mb:.1f} MB)')

    if not args.no_open:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(output_path))


if __name__ == '__main__':
    main()
