# FAST-LIVO2 Python Pipeline

Turns rosbag files (LiDAR + IMU + optional camera) into interactive 3D point cloud viewers you can open in a browser.

## Quick Start

### 1. Install dependencies

```bash
bash setup.sh
```

Or manually: `pip install -r requirements.txt` (needs Python 3.8+, numpy, opencv-python, pyyaml, rosbags).

### 2. Put your .bag file in `Bags/`

### 3. Run the pipeline

```bash
python run.py
```

That's it. It will find your bag file, run SLAM, generate a viewer, and open it in your browser.

## What it produces

After running, you'll find in `outputs/output_<bagname>/`:

| File | What it is |
|------|-----------|
| `colored_viewer.html` | Interactive 3D viewer (open in browser) |
| `map.pcd` | Full SLAM point cloud |
| `odometry_cloud.pcd` | Full odometry point cloud (raw scans + poses) |
| `raw_lidar.pcd` | Raw LiDAR data in sensor frame |
| `slices/` | Density heatmap PNGs + interactive slice viewer |

## Common Commands

```bash
# Run full pipeline (interactive)
python run.py

# Run full pipeline on a specific bag (no prompts)
python run.py Bags/scan.bag --non-interactive --skip-thickened

# Re-generate viewer without re-running SLAM (much faster)
python run.py Bags/scan.bag --non-interactive --skip-slam --skip-thickened

# See what's in a bag file before processing
python run.py --info
python run.py Bags/scan.bag --info

# Generate a high-density viewer for a specific region
python region_viewer.py outputs/output_scan/map.pcd --info          # see bounds
python region_viewer.py outputs/output_scan/map.pcd --x 10 20 --y -2 7 --z -1 1

# Region viewer with camera colorization
python region_viewer.py outputs/output_scan/map.pcd \
    --x 10 20 --y -2 7 --z -1 1 \
    --bag Bags/scan.bag \
    --config config/avia.yaml \
    --camera config/camera_pinhole.yaml

# Full-resolution slice analysis (heatmaps from ALL points)
python slice_analysis.py outputs/output_scan/map.pcd
python slice_analysis.py outputs/output_scan/map.pcd --axis z --num-slices 20
```

## Pipeline Options

| Flag | What it does |
|------|-------------|
| `--info` | Show bag contents (LiDAR/IMU/camera counts) without running anything |
| `--skip-slam` | Skip SLAM, reuse existing map.pcd (for re-generating viewer quickly) |
| `--skip-thickened` | Skip thickened variants (saves a lot of time) |
| `--full` | Embed 1.5M points instead of 300K (bigger file, more detail in viewer) |
| `--min-voxel-pts N` | Only keep voxels with N+ points (2-5 filters noise) |
| `--no-align-ground` | Disable automatic ground leveling |
| `--non-interactive` | No prompts, use all defaults |
| `--no-open` | Don't auto-open the viewer in browser |

## Tools

| File | Purpose |
|------|---------|
| `run.py` | Main pipeline: bag -> SLAM -> viewer (the only file you need) |
| `region_viewer.py` | High-density viewer for a cropped X/Y/Z region of the point cloud |
| `slice_analysis.py` | Full-resolution density heatmaps from map.pcd |

## How It Works

1. **SLAM** (`fast_livo2.py`) — processes LiDAR + IMU to build a 3D map
2. **Viewer** (`colorize_cloud.py`) — colorizes the map, generates an HTML viewer with:
   - LIVO2 Map (SLAM output)
   - Raw LiDAR (sensor frame, no transforms)
   - Odometry Cloud (raw scans projected using poses)
   - Overlay (both combined)
3. **Slice Analysis** (`slice_analysis.py`) — Z-axis density heatmaps on full point cloud
4. **Ground Alignment** — automatic two-pass leveling (IMU gravity + tilt optimization)

## Directory Layout

```
Bags/                   Put .bag files here
config/                 Sensor config files (auto-detected)
  avia.yaml               LiDAR/IMU extrinsics and SLAM parameters
  camera_pinhole.yaml     Camera intrinsics
outputs/                All outputs go here automatically
  output_<bagname>/     One folder per scan
    colored_viewer.html   3D viewer (open in browser)
    map.pcd               SLAM point cloud
    odometry_cloud.pcd    Odometry point cloud
    raw_lidar.pcd         Raw sensor-frame cloud
    slices/               Heatmap analysis
run.py                  Main entry point
region_viewer.py        Regional high-density viewer
slice_analysis.py       Full-resolution heatmap tool
setup.sh                Install script
requirements.txt        Python dependencies
```

## Config Files

The `config/` folder contains sensor calibration files for the Livox Avia LiDAR + camera setup:

- **`avia.yaml`** — LiDAR-IMU extrinsics, topic names, SLAM parameters (voxel size, iteration counts, etc.)
- **`camera_pinhole.yaml`** — Camera intrinsics (focal length, principal point, distortion)

These are auto-detected by `run.py`. If you're using a different sensor setup, replace these files with your own calibration. If you move the config files elsewhere, use `--config` and `--camera` flags:

```bash
python run.py Bags/scan.bag --config /path/to/your_config.yaml --camera /path/to/your_camera.yaml
```
