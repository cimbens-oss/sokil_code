#!/usr/bin/env python3
"""FAST-LIVO2 Pipeline Runner

Put a .bag file in Bags/, then run:  python run.py

That's it. It runs SLAM, generates a 3D viewer, and opens it in your browser.

SETUP:
    bash setup.sh              (or: pip install -r requirements.txt)

COMMON COMMANDS:
    python run.py                                          # interactive
    python run.py Bags/scan.bag --non-interactive           # automated
    python run.py Bags/scan.bag --non-interactive --skip-slam  # re-gen viewer only
    python run.py --info                                    # inspect bag files

OUTPUTS (in outputs/output_<bagname>/):
    colored_viewer.html    3D viewer — open in browser
    map.pcd                SLAM point cloud (full resolution)
    odometry_cloud.pcd     Odometry point cloud (raw scans + poses)
    raw_lidar.pcd          Raw LiDAR in sensor frame
    slices/                Density heatmaps + slice viewer

OTHER TOOLS:
    python region_viewer.py outputs/.../map.pcd --x 10 20 --y -2 7 --z -1 1
        High-density viewer for a cropped region (uses all points in the box)

    python slice_analysis.py outputs/.../map.pcd --axis z --num-slices 20
        Full-resolution density heatmaps

    python run.py --help       Full list of options
"""

import argparse
import os
import re
import subprocess
import sys
import time
import webbrowser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Auto-detection helpers ───────────────────────────────────────

def find_bag_files():
    """Scan known locations for .bag / .db3 files.

    Searches (in order):
      1. ../lio_standalone/Scans/   (main scan archive)
      2. ../                        (project root)
      3. .                          (current directory)

    Returns list of absolute paths, sorted by name.
    """
    search_dirs = [
        os.path.join(SCRIPT_DIR, 'Bags'),
        os.path.join(SCRIPT_DIR, '..', 'lio_standalone', 'Scans'),
        os.path.join(SCRIPT_DIR, '..'),
        os.getcwd(),
    ]

    bags = set()
    for d in search_dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            # Don't recurse into output directories or hidden dirs
            dirs[:] = [x for x in dirs
                       if not x.startswith('.') and not x.startswith('output')]
            for f in files:
                if f.endswith('.bag') or f.endswith('.db3'):
                    bags.add(os.path.abspath(os.path.join(root, f)))

    return sorted(bags)


def auto_detect_config():
    """Find avia.yaml in the standard location.

    Looks for FAST-LIVO2-main/config/avia.yaml relative to the project.
    """
    candidates = [
        os.path.join(SCRIPT_DIR, '..', 'FAST-LIVO2-main', 'config', 'avia.yaml'),
        os.path.join(SCRIPT_DIR, 'avia.yaml'),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    return None


def auto_detect_camera(config_path=None):
    """Find camera_pinhole.yaml in the standard location.

    Looks in the same directory as the main config, or known paths.
    """
    candidates = []
    if config_path:
        config_dir = os.path.dirname(config_path)
        candidates.append(os.path.join(config_dir, 'camera_pinhole.yaml'))
    candidates.extend([
        os.path.join(SCRIPT_DIR, '..', 'FAST-LIVO2-main', 'config', 'camera_pinhole.yaml'),
        os.path.join(SCRIPT_DIR, 'camera_pinhole.yaml'),
    ])
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(c):
            return c
    return None


def sanitize_output_name(bag_path):
    """Generate a clean output directory name from a bag filename.

    'front_room_scan (1).bag' → 'output_front_room_scan'
    """
    stem = os.path.splitext(os.path.basename(bag_path))[0]
    # Remove trailing numbers in parens like " (1)"
    stem = re.sub(r'\s*\(\d+\)\s*$', '', stem)
    # Replace non-alphanumeric with underscore
    stem = re.sub(r'[^a-zA-Z0-9]+', '_', stem).strip('_').lower()
    return os.path.join(SCRIPT_DIR, 'outputs', f'output_{stem}')


# ─── Bag analysis ────────────────────────────────────────────────

def analyze_bag(bag_path):
    """Analyze a rosbag and print a detailed summary of its contents."""
    from rosbags.rosbag1 import Reader as Reader1
    import pathlib

    name = os.path.basename(bag_path)
    size_mb = os.path.getsize(bag_path) / (1024 * 1024)

    print(f'\n{"="*60}')
    print(f'  Bag Analysis: {name}')
    print(f'{"="*60}')
    print(f'  Path: {bag_path}')
    print(f'  Size: {size_mb:.1f} MB')

    try:
        with Reader1(pathlib.Path(bag_path)) as reader:
            topics = {}
            for c in reader.connections:
                if c.topic not in topics:
                    topics[c.topic] = {'type': c.msgtype, 'count': c.msgcount}
                else:
                    topics[c.topic]['count'] += c.msgcount

            # Classify topics
            lidar_topics = []
            imu_topics = []
            image_topics = []
            other_topics = []

            for topic, info in topics.items():
                msgtype = info['type']
                if 'Imu' in msgtype:
                    imu_topics.append((topic, info))
                elif 'Image' in msgtype or 'image' in topic.lower() or 'camera' in topic.lower():
                    image_topics.append((topic, info))
                elif 'PointCloud' in msgtype or 'CustomMsg' in msgtype or 'lidar' in topic.lower():
                    lidar_topics.append((topic, info))
                else:
                    other_topics.append((topic, info))

            # Compute totals
            total_lidar = sum(info['count'] for _, info in lidar_topics)
            total_imu = sum(info['count'] for _, info in imu_topics)
            total_images = sum(info['count'] for _, info in image_topics)

            # Estimate duration from message counts
            # LiDAR at ~10 Hz is typical for Livox
            est_duration = total_lidar / 10.0 if total_lidar > 0 else 0

            print(f'\n  {"─"*56}')
            print(f'  Topics ({len(topics)} total):')
            print(f'  {"─"*56}')
            for topic, info in sorted(topics.items()):
                print(f'    {topic:40s} [{info["type"]}]')
                print(f'    {"":40s}  {info["count"]:,} messages')

            print(f'\n  {"─"*56}')
            print(f'  Summary:')
            print(f'  {"─"*56}')
            print(f'    LiDAR scans:   {total_lidar:>8,}', end='')
            if lidar_topics:
                print(f'  ({", ".join(t for t, _ in lidar_topics)})')
            else:
                print(f'  (none found)')

            print(f'    IMU messages:  {total_imu:>8,}', end='')
            if imu_topics:
                print(f'  ({", ".join(t for t, _ in imu_topics)})')
            else:
                print(f'  (none found)')

            print(f'    Camera images: {total_images:>8,}', end='')
            if image_topics:
                print(f'  ({", ".join(t for t, _ in image_topics)})')
            else:
                print(f'  (none found)')

            if est_duration > 0:
                mins = int(est_duration) // 60
                secs = int(est_duration) % 60
                print(f'    Est. duration: {mins}m {secs}s  (assuming ~10 Hz LiDAR)')

            # Data availability assessment
            print(f'\n  {"─"*56}')
            print(f'  Pipeline compatibility:')
            print(f'  {"─"*56}')

            has_lidar = total_lidar > 0
            has_imu = total_imu > 0
            has_images = total_images > 0

            if has_lidar and has_imu and has_images:
                print(f'    SLAM:          YES  (LiDAR + IMU)')
                print(f'    Visual data:   YES  ({total_images} camera frames)')
                print(f'    Viewer modes:  Height, Intensity, RGB color, Overlay')
                print(f'    Rating:        FULL PIPELINE  (best results)')
            elif has_lidar and has_imu:
                print(f'    SLAM:          YES  (LiDAR + IMU)')
                print(f'    Visual data:   NO   (no camera images)')
                print(f'    Viewer modes:  Height, Intensity only (no RGB)')
                print(f'    Rating:        LiDAR-ONLY  (no color, but geometry works)')
            elif has_lidar:
                print(f'    SLAM:          MAYBE  (LiDAR only, no IMU — may drift)')
                print(f'    Visual data:   {"YES" if has_images else "NO"}')
                print(f'    Rating:        LIMITED  (no IMU for motion compensation)')
            else:
                print(f'    SLAM:          NO  (no LiDAR data found)')
                print(f'    Rating:        INCOMPATIBLE')

            # Check for existing output
            output_dir = sanitize_output_name(bag_path)
            if os.path.isdir(output_dir):
                files = os.listdir(output_dir)
                has_pcd = 'map.pcd' in files
                has_viewer = 'colored_viewer.html' in files
                print(f'\n  {"─"*56}')
                print(f'  Existing output: {os.path.relpath(output_dir)}/')
                print(f'  {"─"*56}')
                print(f'    map.pcd:            {"YES" if has_pcd else "NO"}')
                print(f'    colored_viewer.html: {"YES" if has_viewer else "NO"}')
                if has_pcd:
                    pcd_size = os.path.getsize(os.path.join(output_dir, 'map.pcd')) / 1e6
                    print(f'    map.pcd size:       {pcd_size:.1f} MB')
                if has_viewer:
                    v_size = os.path.getsize(os.path.join(output_dir, 'colored_viewer.html')) / 1e6
                    print(f'    viewer size:        {v_size:.1f} MB')

    except Exception as e:
        print(f'\n  ERROR reading bag: {e}')

    print(f'\n{"="*60}\n')


def run_info(bag_path):
    """Run --info mode: analyze one bag or all bags in Bags/."""
    if bag_path:
        # Analyze specific bag
        analyze_bag(bag_path)
    else:
        # Analyze all bags in Bags/ directory
        bags_dir = os.path.join(SCRIPT_DIR, 'Bags')
        if not os.path.isdir(bags_dir):
            print(f'ERROR: No Bags/ directory found at {bags_dir}')
            sys.exit(1)

        bag_files = sorted(
            os.path.join(bags_dir, f)
            for f in os.listdir(bags_dir)
            if f.endswith('.bag') or f.endswith('.db3')
        )

        if not bag_files:
            print('No bag files found in Bags/')
            sys.exit(1)

        print(f'\nAnalyzing {len(bag_files)} bag file(s) in Bags/...')
        for bf in bag_files:
            analyze_bag(bf)

        # Print compact comparison table
        print(f'{"="*80}')
        print(f'  Summary Table')
        print(f'{"="*80}')
        print(f'  {"Bag":<35s} {"Size":>7s}  {"LiDAR":>6s}  {"IMU":>6s}  {"Imgs":>6s}  {"Visual?"}')
        print(f'  {"─"*75}')

        from rosbags.rosbag1 import Reader as Reader1
        import pathlib
        for bf in bag_files:
            name = os.path.basename(bf)
            size_mb = os.path.getsize(bf) / (1024 * 1024)
            try:
                with Reader1(pathlib.Path(bf)) as reader:
                    lidar = imu = imgs = 0
                    for c in reader.connections:
                        mt = c.msgtype
                        if 'Imu' in mt:
                            imu += c.msgcount
                        elif 'Image' in mt or 'image' in c.topic.lower() or 'camera' in c.topic.lower():
                            imgs += c.msgcount
                        elif 'PointCloud' in mt or 'CustomMsg' in mt or 'lidar' in c.topic.lower():
                            lidar += c.msgcount
                    visual = 'YES' if imgs > 0 else 'NO'
                    print(f'  {name:<35s} {size_mb:>6.0f}M  {lidar:>6d}  {imu:>6d}  {imgs:>6d}  {visual}')
            except Exception as e:
                print(f'  {name:<35s} ERROR: {e}')

        print(f'  {"─"*75}')
        print()


# ─── Interactive prompts ──────────────────────────────────────────

def prompt_yes_no(prompt, default=True):
    """Ask a yes/no question. Returns True/False."""
    suffix = '[Y/n]' if default else '[y/N]'
    while True:
        ans = input(f'{prompt} {suffix}: ').strip().lower()
        if ans == '':
            return default
        if ans in ('y', 'yes'):
            return True
        if ans in ('n', 'no'):
            return False
        print('Please enter y or n.')


def prompt_bag_selection(bag_files):
    """Present a numbered list and let the user pick a bag file."""
    if not bag_files:
        print('ERROR: No .bag files found.')
        print('Place .bag files in ../lio_standalone/Scans/ or provide a path.')
        sys.exit(1)

    if len(bag_files) == 1:
        rel = os.path.relpath(bag_files[0])
        size_mb = os.path.getsize(bag_files[0]) / 1e6
        print(f'\nFound 1 bag file: {rel}  ({size_mb:.1f} MB)')
        if prompt_yes_no('Process this bag?', default=True):
            return bag_files[0]
        else:
            print('Aborted.')
            sys.exit(0)

    print(f'\nFound {len(bag_files)} bag file(s):\n')
    for i, bf in enumerate(bag_files, 1):
        rel = os.path.relpath(bf)
        size_mb = os.path.getsize(bf) / 1e6
        print(f'  [{i}] {rel}  ({size_mb:.1f} MB)')

    print()
    while True:
        choice = input(f'Select bag [1-{len(bag_files)}]: ').strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(bag_files):
                return bag_files[idx - 1]
        except ValueError:
            pass
        print(f'Please enter a number between 1 and {len(bag_files)}.')


# ─── Pipeline execution ──────────────────────────────────────────

def run_slam(bag_path, config_path, camera_path, output_dir):
    """Run the SLAM pipeline (fast_livo2.py) as a subprocess."""
    script = os.path.join(SCRIPT_DIR, 'fast_livo2.py')
    cmd = [
        sys.executable, script,
        '--bag', bag_path,
        '--config', config_path,
        '--output', output_dir,
    ]
    if camera_path:
        cmd.extend(['--camera-config', camera_path])

    print(f'\n{"="*60}')
    print('Step 1: Running SLAM pipeline (fast_livo2.py)')
    print(f'{"="*60}')
    sys.stdout.flush()
    t0 = time.time()

    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'\nERROR: SLAM pipeline failed (exit code {result.returncode})')
        sys.exit(1)

    # Verify outputs
    map_path = os.path.join(output_dir, 'map.pcd')
    traj_path = os.path.join(output_dir, 'trajectory.txt')
    if not os.path.isfile(map_path):
        print(f'ERROR: SLAM did not produce {map_path}')
        sys.exit(1)
    if not os.path.isfile(traj_path):
        print(f'ERROR: SLAM did not produce {traj_path}')
        sys.exit(1)

    print(f'\nSLAM completed in {elapsed:.1f}s')
    print(f'  map.pcd:       {os.path.getsize(map_path) / 1e6:.1f} MB')
    print(f'  trajectory.txt: {os.path.getsize(traj_path) / 1e3:.1f} KB')
    return map_path, traj_path


def run_colorize(bag_path, map_path, traj_path, config_path, camera_path,
                 output_html, max_points, embed_limit, skip_thickened,
                 num_variants, max_shift_ms, align_ground=False,
                 voxel_size=0.02, min_voxel_pts=1):
    """Run the colorization + viewer pipeline (colorize_cloud.py)."""
    script = os.path.join(SCRIPT_DIR, 'colorize_cloud.py')
    cmd = [
        sys.executable, script,
        '--pcd', map_path,
        '--bag', bag_path,
        '--trajectory', traj_path,
        '--config', config_path,
        '--camera', camera_path,
        '--output', output_html,
        '--max-points', str(max_points),
        '--embed-limit', str(embed_limit),
        '--num-variants', str(num_variants),
        '--max-shift-ms', str(max_shift_ms),
        '--voxel-size', str(voxel_size),
        '--min-voxel-pts', str(min_voxel_pts),
    ]
    if skip_thickened:
        cmd.append('--skip-thickened')
    if align_ground:
        cmd.append('--align-ground')

    print(f'\n{"="*60}')
    print('Step 2: Colorizing + generating viewer (colorize_cloud.py)')
    print(f'{"="*60}')
    sys.stdout.flush()
    t0 = time.time()

    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'\nERROR: Colorization failed (exit code {result.returncode})')
        print(f'SLAM output is still available at: {os.path.dirname(map_path)}/')
        sys.exit(1)

    if not os.path.isfile(output_html):
        print(f'ERROR: Viewer was not produced at {output_html}')
        sys.exit(1)

    size_mb = os.path.getsize(output_html) / 1e6
    print(f'\nViewer generated in {elapsed:.1f}s')
    print(f'  {output_html}  ({size_mb:.1f} MB)')
    return output_html


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FAST-LIVO2 Pipeline Runner — rosbag to HTML viewer.\n\n'
                    'Run with no arguments for interactive mode.\n'
                    'Use --non-interactive for AI/script usage.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'EXAMPLES:\n'
            '  Interactive:        python run.py\n'
            '  With bag path:      python run.py path/to/scan.bag\n'
            '  AI (full pipeline): python run.py scan.bag --non-interactive --skip-thickened\n'
            '  AI (viewer only):   python run.py scan.bag --non-interactive --skip-slam --output-dir output_front_room\n'
        ),
    )
    parser.add_argument('bag', nargs='?', default=None,
                        help='Path to .bag file (omit for interactive selection)')
    parser.add_argument('--config', default=None,
                        help='Main YAML config (default: auto-detect avia.yaml)')
    parser.add_argument('--camera', default=None,
                        help='Camera YAML config (default: auto-detect camera_pinhole.yaml)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: ./output_<bagname>/)')
    parser.add_argument('--max-points', type=int, default=200000,
                        help='Display limit per dataset (default: 200000)')
    parser.add_argument('--embed-limit', type=int, default=300000,
                        help='Embed limit per dataset (default: 300000)')
    parser.add_argument('--full', action='store_true',
                        help='Embed full point cloud in viewer (larger file, slower to load)')
    parser.add_argument('--skip-slam', action='store_true',
                        help='Skip SLAM — reuse existing map.pcd + trajectory.txt')
    parser.add_argument('--skip-colorize', action='store_true',
                        help='Skip colorization — just run SLAM')
    parser.add_argument('--skip-thickened', action='store_true',
                        help='Skip Mode 3 thickened variants (much faster)')
    parser.add_argument('--num-variants', type=int, default=11,
                        help='Thickened variant count (default: 11)')
    parser.add_argument('--max-shift-ms', type=float, default=1.0,
                        help='Max IMU shift in ms for thickening (default: 1.0)')
    parser.add_argument('--non-interactive', action='store_true',
                        help='No prompts — use defaults for everything')
    parser.add_argument('--no-align-ground', action='store_true',
                        help='Disable automatic ground plane alignment')
    parser.add_argument('--min-voxel-pts', type=int, default=1,
                        help='Min points per 2cm voxel to keep (default: 1, use 2-5 to filter noise)')
    parser.add_argument('--skip-slices', action='store_true',
                        help='Skip automatic slice analysis on full point cloud')
    parser.add_argument('--no-open', action='store_true',
                        help="Don't open viewer in browser when done")
    parser.add_argument('--info', action='store_true',
                        help='Analyze bag file(s) and print summary — does not run pipeline')

    args = parser.parse_args()

    # ── Full mode overrides ──────────────────────────────────────
    if args.full:
        args.embed_limit = 1500000
        args.max_points = 500000

    # ── Info mode (analyze bag, don't run pipeline) ───────────────
    if args.info:
        bag_path = os.path.abspath(args.bag) if args.bag else None
        if bag_path and not os.path.isfile(bag_path):
            print(f'ERROR: Bag file not found: {bag_path}')
            sys.exit(1)
        run_info(bag_path)
        sys.exit(0)

    interactive = not args.non_interactive

    # ── Banner ────────────────────────────────────────────────────
    if interactive:
        print()
        print('=' * 58)
        print('  FAST-LIVO2 Pipeline Runner')
        print('  Rosbag → SLAM → Colorized 3D Viewer')
        print('=' * 58)

    # ── Resolve bag path ──────────────────────────────────────────
    if args.bag:
        bag_path = os.path.abspath(args.bag)
        if not os.path.isfile(bag_path):
            print(f'ERROR: Bag file not found: {bag_path}')
            sys.exit(1)
    elif args.skip_slam:
        # No bag needed if skipping SLAM, but we still need it for colorization
        bag_path = None
    else:
        if not interactive:
            print('ERROR: No bag file provided. Use: python run.py path/to/scan.bag --non-interactive')
            sys.exit(1)
        bag_files = find_bag_files()
        bag_path = prompt_bag_selection(bag_files)

    # ── Resolve config ────────────────────────────────────────────
    if args.config:
        config_path = os.path.abspath(args.config)
    else:
        config_path = auto_detect_config()
        if config_path:
            if interactive:
                print(f'\nConfig: {os.path.relpath(config_path)} (auto-detected)')
        else:
            if interactive:
                path = input('Config YAML path (e.g., avia.yaml): ').strip()
                config_path = os.path.abspath(path)
            else:
                print('ERROR: Could not auto-detect config. Use --config.')
                sys.exit(1)

    if not os.path.isfile(config_path):
        print(f'ERROR: Config not found: {config_path}')
        sys.exit(1)

    # ── Resolve camera config ─────────────────────────────────────
    if args.camera:
        camera_path = os.path.abspath(args.camera)
    else:
        camera_path = auto_detect_camera(config_path)
        if camera_path:
            if interactive:
                print(f'Camera: {os.path.relpath(camera_path)} (auto-detected)')
        else:
            if interactive:
                path = input('Camera YAML path (e.g., camera_pinhole.yaml, or Enter to skip): ').strip()
                camera_path = os.path.abspath(path) if path else None
            else:
                print('WARNING: Could not auto-detect camera config. Colorization may fail.')
                camera_path = None

    if camera_path and not os.path.isfile(camera_path):
        print(f'ERROR: Camera config not found: {camera_path}')
        sys.exit(1)

    # ── Resolve output directory ──────────────────────────────────
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    elif bag_path:
        output_dir = sanitize_output_name(bag_path)
    else:
        print('ERROR: --output-dir required when --skip-slam and no bag provided.')
        sys.exit(1)

    if interactive:
        print(f'Output: {os.path.relpath(output_dir)}/')

    # ── Resolve pipeline options ──────────────────────────────────
    do_slam = not args.skip_slam
    do_colorize = not args.skip_colorize
    skip_thickened = args.skip_thickened

    if interactive:
        print()
        if not args.skip_slam:
            do_slam = prompt_yes_no('Run SLAM pipeline?', default=True)
        if do_slam or os.path.isfile(os.path.join(output_dir, 'map.pcd')):
            if not args.skip_colorize:
                do_colorize = prompt_yes_no('Generate colorized viewer?', default=True)
            if do_colorize and not args.skip_thickened:
                skip_thickened = not prompt_yes_no(
                    'Include thickened variants (re-runs SLAM ~11 times, slow)?',
                    default=False)

    # ── Resolve bag for colorization ──────────────────────────────
    if do_colorize and not bag_path:
        if interactive:
            path = input('Bag file path (needed for colorization): ').strip()
            bag_path = os.path.abspath(path)
        else:
            print('ERROR: Bag file required for colorization. '
                  'Provide a bag path or use --skip-colorize.')
            sys.exit(1)

        if not os.path.isfile(bag_path):
            print(f'ERROR: Bag file not found: {bag_path}')
            sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────
    sys.stdout.flush()
    print(f'\n{"─"*58}')
    print('Pipeline Summary')
    print(f'{"─"*58}')
    if bag_path:
        print(f'  Bag:       {os.path.relpath(bag_path)}')
    print(f'  Config:    {os.path.relpath(config_path)}')
    if camera_path:
        print(f'  Camera:    {os.path.relpath(camera_path)}')
    print(f'  Output:    {os.path.relpath(output_dir)}/')
    print(f'  SLAM:      {"Yes" if do_slam else "Skip (reuse existing)"}')
    if do_colorize:
        thk = 'with thickened' if not skip_thickened else 'skip thickened'
        print(f'  Colorize:  Yes ({thk})')
    else:
        print(f'  Colorize:  Skip')
    print(f'{"─"*58}')

    if interactive:
        if not prompt_yes_no('\nProceed?', default=True):
            print('Aborted.')
            sys.exit(0)

    # ── Execute pipeline ──────────────────────────────────────────
    sys.stdout.flush()
    os.makedirs(output_dir, exist_ok=True)
    t_total = time.time()

    map_path = os.path.join(output_dir, 'map.pcd')
    traj_path = os.path.join(output_dir, 'trajectory.txt')

    # Step 1: SLAM
    if do_slam:
        map_path, traj_path = run_slam(
            bag_path, config_path, camera_path, output_dir)
    else:
        # Verify existing SLAM output
        if not os.path.isfile(map_path):
            print(f'ERROR: --skip-slam but no map.pcd at {map_path}')
            sys.exit(1)
        if not os.path.isfile(traj_path):
            print(f'ERROR: --skip-slam but no trajectory.txt at {traj_path}')
            sys.exit(1)
        print(f'\nUsing existing SLAM output:')
        print(f'  {map_path}  ({os.path.getsize(map_path) / 1e6:.1f} MB)')
        print(f'  {traj_path}')

    # Step 2: Colorize + Viewer
    viewer_path = None
    if do_colorize:
        if not camera_path:
            print('WARNING: No camera config — skipping colorization.')
        else:
            output_html = os.path.join(output_dir, 'colored_viewer.html')
            viewer_path = run_colorize(
                bag_path, map_path, traj_path, config_path, camera_path,
                output_html, args.max_points, args.embed_limit,
                skip_thickened, args.num_variants, args.max_shift_ms,
                align_ground=not args.no_align_ground,
                voxel_size=0.02,
                min_voxel_pts=args.min_voxel_pts)

    # Step 3: Slice analysis on full point cloud
    if not args.skip_slices and os.path.isfile(map_path):
        slice_script = os.path.join(SCRIPT_DIR, 'slice_analysis.py')
        if os.path.isfile(slice_script):
            print(f'\n{"="*60}')
            print('Step 3: Slice analysis (full point cloud)')
            print(f'{"="*60}')
            sys.stdout.flush()
            slice_cmd = [
                sys.executable, slice_script,
                map_path,
                '--axis', 'z',
                '--num-slices', '10',
                '--grid', '150',
                '--no-open',
            ]
            result = subprocess.run(slice_cmd)
            if result.returncode != 0:
                print('WARNING: Slice analysis failed (non-fatal)')

    # Step 4: Open viewer
    total_elapsed = time.time() - t_total

    print(f'\n{"="*58}')
    print(f'  Pipeline complete!  ({total_elapsed:.1f}s total)')
    print(f'{"="*58}')
    print(f'\nOutputs in: {output_dir}/')
    for f in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            print(f'  {f}  ({os.path.getsize(fp) / 1e6:.1f} MB)')

    if viewer_path and not args.no_open:
        print(f'\nOpening viewer in browser...')
        webbrowser.open('file://' + os.path.abspath(viewer_path))
    elif viewer_path:
        print(f'\nViewer ready at: {viewer_path}')
        print(f'Open with: open "{viewer_path}"')


if __name__ == '__main__':
    main()
