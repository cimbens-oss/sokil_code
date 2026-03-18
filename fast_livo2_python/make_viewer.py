"""Generate a self-contained HTML point cloud viewer from a PCD file."""
import numpy as np
import base64
import sys
import os

def read_pcd_ascii(path):
    """Read ASCII PCD file, return Nx3 float32 array."""
    with open(path, 'r') as f:
        lines = f.readlines()
    # Find DATA line
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

def height_colormap(pts):
    """Color points by height (z-axis). Returns Nx3 uint8 RGB."""
    z = pts[:, 2]
    z_min, z_max = np.percentile(z, [2, 98])
    t = np.clip((z - z_min) / (z_max - z_min + 1e-8), 0, 1)

    # Turbo-ish colormap: blue -> cyan -> green -> yellow -> red
    r = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(t - 0.5) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)

    colors = np.stack([r, g, b], axis=1)
    return (colors * 255).astype(np.uint8)

def make_html(pts, colors, title="Point Cloud Viewer"):
    """Create self-contained HTML with embedded point cloud."""
    # Encode positions as base64 float32
    pos_b64 = base64.b64encode(pts.tobytes()).decode('ascii')
    col_b64 = base64.b64encode(colors.tobytes()).decode('ascii')
    n_points = len(pts)

    # Center of mass for camera target
    center = pts.mean(axis=0)
    extent = pts.max(axis=0) - pts.min(axis=0)
    cam_dist = float(np.linalg.norm(extent)) * 1.2

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #1a1a2e; }}
  #info {{
    position: absolute; top: 10px; left: 10px; color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px;
    background: rgba(0,0,0,0.6); padding: 12px 16px; border-radius: 8px;
    pointer-events: none; line-height: 1.6;
  }}
  #controls {{
    position: absolute; bottom: 10px; left: 10px; color: #aaa;
    font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px;
    background: rgba(0,0,0,0.5); padding: 8px 12px; border-radius: 6px;
    pointer-events: none;
  }}
  #sizeSlider {{
    position: absolute; top: 10px; right: 10px;
    background: rgba(0,0,0,0.6); padding: 12px 16px; border-radius: 8px;
    color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px;
  }}
  input[type=range] {{ width: 140px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="info">
  <strong>Point Cloud Viewer</strong><br>
  Points: {n_points:,}
</div>
<div id="controls">
  Orbit: Left-click + drag &nbsp;|&nbsp; Pan: Right-click + drag &nbsp;|&nbsp; Zoom: Scroll
</div>
<div id="sizeSlider">
  Point size: <input type="range" id="psize" min="0.5" max="8" step="0.5" value="2">
  <span id="psizeVal">2.0</span>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

// Decode embedded data
const posB64 = "{pos_b64}";
const colB64 = "{col_b64}";
const nPoints = {n_points};

function b64ToFloat32(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}}

function b64ToUint8(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return view;
}}

const positions = b64ToFloat32(posB64);
const colorsRaw = b64ToUint8(colB64);

// Convert colors to float [0,1]
const colors = new Float32Array(colorsRaw.length);
for (let i = 0; i < colorsRaw.length; i++) colors[i] = colorsRaw[i] / 255.0;

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(
  {center[0]:.3f} + {cam_dist:.1f} * 0.5,
  {center[1]:.3f} + {cam_dist:.1f} * 0.3,
  {center[2]:.3f} + {cam_dist:.1f} * 0.8
);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Point cloud
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

const material = new THREE.PointsMaterial({{
  size: 2,
  vertexColors: true,
  sizeAttenuation: true
}});

const pointCloud = new THREE.Points(geometry, material);
scene.add(pointCloud);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f});
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.update();

// Grid helper
const gridSize = {float(max(extent[0], extent[1])):.1f} * 1.5;
const grid = new THREE.GridHelper(gridSize, 20, 0x444466, 0x333355);
grid.position.set({center[0]:.3f}, {center[1]:.3f}, {float(pts[:, 2].min()):.3f});
grid.rotation.x = Math.PI / 2;
scene.add(grid);

// Ambient light
scene.add(new THREE.AmbientLight(0xffffff, 0.5));

// Point size slider
const slider = document.getElementById('psize');
const sliderVal = document.getElementById('psizeVal');
slider.addEventListener('input', () => {{
  material.size = parseFloat(slider.value);
  sliderVal.textContent = slider.value;
}});

// Resize
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// Animate
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""
    return html

if __name__ == '__main__':
    pcd_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    max_points = int(sys.argv[3]) if len(sys.argv) > 3 else 300000

    if not pcd_path:
        print("Usage: python make_viewer.py <input.pcd> <output.html> [max_points]")
        sys.exit(1)

    if not output_path:
        output_path = os.path.splitext(pcd_path)[0] + '_viewer.html'

    print(f"Reading {pcd_path}...")
    pts = read_pcd_ascii(pcd_path)
    print(f"  Loaded {len(pts):,} points")

    # Downsample if needed for browser performance
    if len(pts) > max_points:
        idx = np.random.default_rng(42).choice(len(pts), max_points, replace=False)
        idx.sort()
        pts = pts[idx]
        print(f"  Downsampled to {len(pts):,} points")

    print("Generating colors...")
    colors = height_colormap(pts)

    print(f"Writing {output_path}...")
    html = make_html(pts, colors, title=os.path.basename(pcd_path))
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! {size_mb:.1f} MB - open in any browser.")
