#!/usr/bin/env python3
"""Slice analysis tool — generate density heatmaps from a point cloud.

Reads a map.pcd file and slices it along an axis, producing PNG heatmaps
and an interactive HTML viewer. Works on the FULL point cloud regardless
of size (no embed limits).

Usage:
    python slice_analysis.py outputs/output_scan1/map.pcd
    python slice_analysis.py outputs/output_scan1/map.pcd --axis z --num-slices 20
    python slice_analysis.py outputs/output_scan1/map.pcd --axis x --grid 200 --output heatmaps/
"""
import argparse
import base64
import os
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── PCD reading ─────────────────────────────────────────────────

def read_pcd(path):
    """Read ASCII or binary PCD file -> Nx3 float32."""
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


# ─── Turbo colormap ──────────────────────────────────────────────

def turbo_colormap(t):
    """Apply turbo colormap to values in [0, 1]. Returns Nx3 uint8 (RGB)."""
    t = np.clip(t, 0.0, 1.0)
    # Simplified turbo approximation
    r = np.clip(0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))), 0, 1)
    g = np.clip(0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))), 0, 1)
    b = np.clip(0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ─── Slice + heatmap generation ──────────────────────────────────

def generate_heatmaps(pts, axis, num_slices, thickness, grid_res, output_dir):
    """Slice the point cloud and generate density heatmap PNGs."""
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    other_axes = [i for i in range(3) if i != axis_idx]
    axis_labels = ['X', 'Y', 'Z']
    h_label = axis_labels[other_axes[0]]
    v_label = axis_labels[other_axes[1]]

    axis_min = pts[:, axis_idx].min()
    axis_max = pts[:, axis_idx].max()
    axis_extent = axis_max - axis_min

    if thickness is None:
        thickness = axis_extent / num_slices

    step = axis_extent / num_slices

    # Bounding box for the 2D projection
    h_min, h_max = pts[:, other_axes[0]].min(), pts[:, other_axes[0]].max()
    v_min, v_max = pts[:, other_axes[1]].min(), pts[:, other_axes[1]].max()
    h_extent = h_max - h_min
    v_extent = v_max - v_min

    # Make grid square-ish by aspect ratio
    if h_extent > v_extent:
        grid_w = grid_res
        grid_h = max(1, int(grid_res * v_extent / h_extent))
    else:
        grid_h = grid_res
        grid_w = max(1, int(grid_res * h_extent / v_extent))

    os.makedirs(output_dir, exist_ok=True)
    global_max_density = 0

    print(f'\nSlicing {len(pts):,} points along {axis.upper()}-axis:')
    print(f'  Range: {axis_min:.2f} to {axis_max:.2f} ({axis_extent:.2f}m)')
    print(f'  Slices: {num_slices}, thickness: {thickness:.3f}m, step: {step:.3f}m')
    print(f'  Grid: {grid_w}x{grid_h}, output: {output_dir}/')
    print()

    grids = []
    slice_infos = []

    for i in range(num_slices):
        lo = axis_min + i * step
        hi = lo + thickness
        mid = (lo + hi) / 2.0

        mask = (pts[:, axis_idx] >= lo) & (pts[:, axis_idx] < hi)
        slice_pts = pts[mask]
        n_pts = len(slice_pts)

        # Bin points into 2D grid
        grid = np.zeros((grid_h, grid_w), dtype=np.float64)

        if n_pts > 0:
            h_coords = slice_pts[:, other_axes[0]]
            v_coords = slice_pts[:, other_axes[1]]

            # Normalize to grid indices
            h_idx = np.clip(((h_coords - h_min) / h_extent * (grid_w - 1)).astype(int), 0, grid_w - 1)
            v_idx = np.clip(((v_coords - v_min) / v_extent * (grid_h - 1)).astype(int), 0, grid_h - 1)

            # Count density
            np.add.at(grid, (v_idx, h_idx), 1)

        max_density = grid.max()
        if max_density > global_max_density:
            global_max_density = max_density

        grids.append(grid)
        slice_infos.append((i, lo, hi, mid, n_pts, max_density))

    # Render all grids with consistent color scale
    saved = []
    slice_data = []  # For HTML report
    for grid, (i, lo, hi, mid, n_pts, max_d) in zip(grids, slice_infos):
        if global_max_density > 0:
            normalized = grid / global_max_density
        else:
            normalized = grid

        # Apply colormap
        rgb = turbo_colormap(normalized.flatten()).reshape(grid_h, grid_w, 3)

        # Set empty cells to dark gray
        empty_mask = (grid == 0)
        rgb[empty_mask] = [30, 30, 30]

        # Scale up for visibility
        scale = max(1, 512 // max(grid_h, grid_w))
        if scale > 1:
            rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

        # Add text label
        img_h, img_w = rgb.shape[:2]
        label = f'{axis.upper()}={mid:.2f}m  ({n_pts:,} pts)'
        # Convert RGB to BGR for cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Axis labels
        cv2.putText(bgr, h_label, (img_w // 2, img_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(bgr, v_label, (4, img_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        filename = f'slice_{i:03d}_{axis}{mid:.2f}m.png'
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, bgr)
        saved.append(filepath)

        # Encode PNG for HTML embedding
        _, png_buf = cv2.imencode('.png', bgr)
        b64 = base64.b64encode(png_buf.tobytes()).decode('ascii')
        slice_data.append({
            'index': i, 'lo': lo, 'hi': hi, 'mid': mid,
            'n_pts': n_pts, 'max_density': max_d, 'b64': b64
        })

        status = f'  [{i+1:3d}/{num_slices}] {axis.upper()}={lo:.2f}..{hi:.2f}m  {n_pts:>8,} pts  max_density={max_d:.0f}'
        print(status)

    # Generate summary montage
    if len(saved) > 1:
        montage_path = os.path.join(output_dir, f'montage_{axis}_all.png')
        create_montage(saved, montage_path, num_slices)
        print(f'\n  Montage: {montage_path}')

    # Generate interactive HTML report
    html_path = os.path.join(output_dir, f'slice_viewer_{axis}.html')
    build_html_report(slice_data, axis, h_label, v_label, len(pts),
                      axis_min, axis_max, thickness, grid_w, grid_h,
                      html_path)
    print(f'  HTML viewer: {html_path}')

    print(f'\n  Saved {len(saved)} heatmaps to {output_dir}/')
    return saved


def build_html_report(slice_data, axis, h_label, v_label, total_pts,
                      axis_min, axis_max, thickness, grid_w, grid_h, output_path):
    """Build a self-contained HTML viewer with embedded heatmap PNGs."""
    import json

    slices_json = json.dumps([
        {'i': int(s['index']), 'lo': round(float(s['lo']), 3), 'hi': round(float(s['hi']), 3),
         'mid': round(float(s['mid']), 3), 'pts': int(s['n_pts']),
         'maxD': round(float(s['max_density']), 1), 'img': s['b64']}
        for s in slice_data
    ])

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Slice Analysis — {axis.upper()}-axis</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#1a1a2e; color:#eee; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
.header {{ background:#16213e; padding:16px 24px; border-bottom:1px solid #333; }}
.header h1 {{ font-size:20px; font-weight:600; }}
.header .stats {{ font-size:13px; color:#aaa; margin-top:4px; }}
.container {{ display:flex; height:calc(100vh - 70px); }}
.filmstrip {{ width:140px; overflow-y:auto; background:#0f0f23; border-right:1px solid #333; padding:8px; flex-shrink:0; }}
.thumb {{ cursor:pointer; margin-bottom:6px; border:2px solid transparent; border-radius:4px; overflow:hidden; transition: border-color 0.15s; }}
.thumb:hover {{ border-color:#555; }}
.thumb.active {{ border-color:#e94560; }}
.thumb img {{ width:100%; display:block; }}
.thumb .label {{ font-size:10px; text-align:center; padding:2px; background:#1a1a2e; color:#aaa; }}
.thumb.active .label {{ color:#e94560; }}
.main {{ flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:20px; }}
.main img {{ max-width:100%; max-height:calc(100vh - 160px); image-rendering:pixelated; border:1px solid #333; border-radius:4px; }}
.info {{ margin-top:12px; font-size:14px; color:#ccc; text-align:center; }}
.info span {{ color:#e94560; font-weight:600; }}
.nav {{ margin-top:10px; font-size:12px; color:#666; }}
.nav kbd {{ background:#333; padding:2px 6px; border-radius:3px; font-family:monospace; }}
</style>
</head>
<body>
<div class="header">
  <h1>Slice Analysis — {axis.upper()}-axis</h1>
  <div class="stats">{total_pts:,} points &bull; {len(slice_data)} slices &bull;
    {axis.upper()} range: {axis_min:.2f} to {axis_max:.2f}m &bull;
    thickness: {thickness:.3f}m &bull; grid: {grid_w}&times;{grid_h} &bull;
    axes: {h_label} (horizontal) / {v_label} (vertical)</div>
</div>
<div class="container">
  <div class="filmstrip" id="filmstrip"></div>
  <div class="main">
    <img id="mainImg" />
    <div class="info" id="info"></div>
    <div class="nav">
      <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> prev/next &nbsp;&bull;&nbsp;
      <kbd>Home</kbd> first &nbsp;&bull;&nbsp; <kbd>End</kbd> last
    </div>
  </div>
</div>
<script>
const SLICES = {slices_json};
let current = 0;

function select(idx) {{
  if (idx < 0 || idx >= SLICES.length) return;
  current = idx;
  const s = SLICES[idx];
  document.getElementById('mainImg').src = 'data:image/png;base64,' + s.img;
  document.getElementById('info').innerHTML =
    '<span>' + '{axis.upper()}=' + s.mid.toFixed(2) + 'm</span>' +
    ' &nbsp; range: ' + s.lo.toFixed(2) + ' to ' + s.hi.toFixed(2) + 'm' +
    ' &nbsp;&bull;&nbsp; <span>' + s.pts.toLocaleString() + '</span> points' +
    ' &nbsp;&bull;&nbsp; max density: ' + s.maxD.toFixed(0);
  document.querySelectorAll('.thumb').forEach((t, i) => {{
    t.classList.toggle('active', i === idx);
  }});
  document.querySelectorAll('.thumb')[idx]?.scrollIntoView({{ block:'nearest' }});
}}

// Build filmstrip
const fs = document.getElementById('filmstrip');
SLICES.forEach((s, i) => {{
  const div = document.createElement('div');
  div.className = 'thumb' + (i === 0 ? ' active' : '');
  div.innerHTML = '<img src="data:image/png;base64,' + s.img + '">' +
    '<div class="label">' + '{axis.upper()}=' + s.mid.toFixed(1) + 'm</div>';
  div.onclick = () => select(i);
  fs.appendChild(div);
}});

// Keyboard navigation
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{ select(current - 1); e.preventDefault(); }}
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{ select(current + 1); e.preventDefault(); }}
  if (e.key === 'Home') {{ select(0); e.preventDefault(); }}
  if (e.key === 'End') {{ select(SLICES.length - 1); e.preventDefault(); }}
}});

select(0);
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)


def create_montage(image_paths, output_path, num_images):
    """Create a montage of all slice heatmaps."""
    imgs = [cv2.imread(p) for p in image_paths]
    if not imgs:
        return

    # Determine grid layout
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols

    # Resize all to same size
    target_h, target_w = 256, 256
    resized = []
    for img in imgs:
        resized.append(cv2.resize(img, (target_w, target_h)))

    # Pad with black if needed
    while len(resized) < rows * cols:
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    # Assemble rows
    row_imgs = []
    for r in range(rows):
        row = np.hstack(resized[r * cols:(r + 1) * cols])
        row_imgs.append(row)
    montage = np.vstack(row_imgs)

    cv2.imwrite(output_path, montage)


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Slice analysis — generate density heatmaps from a point cloud PCD file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'EXAMPLES:\n'
            '  python slice_analysis.py outputs/output_scan1/map.pcd\n'
            '  python slice_analysis.py map.pcd --axis z --num-slices 20\n'
            '  python slice_analysis.py map.pcd --axis x --grid 200 --output heatmaps/\n'
        ),
    )
    parser.add_argument('pcd', help='Path to map.pcd file')
    parser.add_argument('--axis', default='z', choices=['x', 'y', 'z'],
                        help='Slice axis (default: z)')
    parser.add_argument('--num-slices', type=int, default=10,
                        help='Number of slices (default: 10)')
    parser.add_argument('--thickness', type=float, default=None,
                        help='Slice thickness in meters (default: auto)')
    parser.add_argument('--grid', type=int, default=100,
                        help='Heatmap grid resolution (default: 100)')
    parser.add_argument('--output', default=None,
                        help='Output directory for PNGs (default: <pcd_dir>/slices/)')
    parser.add_argument('--no-open', action='store_true',
                        help="Don't open HTML viewer in browser")

    args = parser.parse_args()

    pcd_path = os.path.abspath(args.pcd)
    if not os.path.isfile(pcd_path):
        print(f'ERROR: PCD file not found: {pcd_path}')
        sys.exit(1)

    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pcd_path), 'slices')
    output_dir = os.path.abspath(output_dir)

    print(f'{"="*60}')
    print(f'  Slice Analysis')
    print(f'{"="*60}')
    print(f'  PCD:    {pcd_path}')
    print(f'  Size:   {os.path.getsize(pcd_path) / 1e6:.1f} MB')

    print(f'\nReading point cloud...')
    t0 = __import__('time').time()
    pts = read_pcd(pcd_path)
    elapsed = __import__('time').time() - t0
    print(f'  Loaded {len(pts):,} points in {elapsed:.1f}s')

    if len(pts) == 0:
        print('ERROR: No points in PCD file.')
        sys.exit(1)

    print(f'  Bounds: X=[{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]  '
          f'Y=[{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]  '
          f'Z=[{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]')

    generate_heatmaps(pts, args.axis, args.num_slices, args.thickness,
                      args.grid, output_dir)

    # Open HTML viewer in browser
    html_path = os.path.join(output_dir, f'slice_viewer_{args.axis}.html')
    if os.path.isfile(html_path) and not args.no_open:
        import webbrowser
        print(f'\n  Opening viewer in browser...')
        webbrowser.open('file://' + html_path)

    print(f'\n{"="*60}')
    print(f'  Done!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
