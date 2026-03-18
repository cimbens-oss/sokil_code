# FAST-LIVO2 Python Pipeline

Processes rosbag files (LiDAR + IMU + optional camera) into interactive 3D point cloud viewers with automatic ground alignment and slice analysis.

## Setup

```bash
pip install -r requirements.txt
```

Or use the setup script (creates a virtual environment):

```bash
bash setup.sh
```

Requires Python 3.8+.

## Usage

```bash
# Interactive — picks bag file, auto-detects configs
python run.py

# Specify a bag file directly
python run.py Bags/Scan1.bag

# Fully automated (no prompts)
python run.py Bags/Scan1.bag --non-interactive --skip-thickened

# Re-generate viewer only (skip SLAM, reuse existing map.pcd)
python run.py Bags/Scan1.bag --non-interactive --skip-slam --skip-thickened

# Analyze bag files without running the pipeline
python run.py --info
python run.py Bags/Scan1.bag --info

# Full-resolution slice analysis on an existing point cloud
python slice_analysis.py outputs/output_scan1/map.pcd
python slice_analysis.py outputs/output_scan1/map.pcd --axis z --num-slices 20
```

## Directory Structure

```
Bags/               Input .bag files
outputs/            Pipeline outputs (map.pcd, colored_viewer.html, slices/, etc.)
run.py              Main entry point — the only file you need to run
slice_analysis.py   Standalone full-resolution heatmap analysis
setup.sh            One-command install script
requirements.txt    Python dependencies
```

## Options

Run `python run.py --help` for all options.

| Flag | Description |
|------|-------------|
| `--info` | Analyze bag file contents (topics, visual data, compatibility) |
| `--full` | Embed more points in viewer (1.5M vs default 300K) |
| `--min-voxel-pts N` | Only keep voxels with N+ points (filters noise, default: 1) |
| `--skip-slam` | Reuse existing SLAM output (fast re-generation of viewer) |
| `--skip-thickened` | Skip slow thickened variants |
| `--skip-slices` | Skip automatic slice analysis step |
| `--no-align-ground` | Disable ground plane alignment (on by default) |
| `--non-interactive` | No prompts |
| `--no-open` | Don't auto-open viewer in browser |

## Ground Alignment

The pipeline automatically levels the point cloud using a two-pass system:
1. IMU gravity vector from the bag file (coarse alignment)
2. Brute-force tilt optimization maximizing points in a thin horizontal band (fine-tuning)

Disable with `--no-align-ground`.

## Viewer Modes

- **Height** — color by Z elevation
- **Intensity** — LiDAR return intensity
- **RGB** — camera color (when images are available)
- **Slice Preview** — interactive Z-slicing in the 3D viewer
- **Heatmap Sweep** — density heatmaps (subsampled in viewer, full resolution via slice_analysis.py)
