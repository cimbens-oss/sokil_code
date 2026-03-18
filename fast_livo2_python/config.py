"""Configuration loader for FAST-LIVO2 YAML configs."""
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Topics
    lid_topic: str = "/livox/lidar"
    imu_topic: str = "/livox/imu"
    img_topic: str = "/left_camera/image"
    img_en: bool = True
    lidar_en: bool = True

    # Extrinsics (LiDAR to IMU)
    extrinsic_T: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extrinsic_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    # Camera to LiDAR
    Rcl: np.ndarray = field(default_factory=lambda: np.eye(3))
    Pcl: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Time offsets
    imu_time_offset: float = 0.0
    img_time_offset: float = 0.0
    exposure_time_init: float = 0.0

    # Preprocess
    point_filter_num: int = 1
    filter_size_surf: float = 0.1
    lidar_type: int = 1
    scan_line: int = 6
    blind: float = 0.8

    # VIO
    vio_max_iterations: int = 5
    outlier_threshold: float = 1000.0
    img_point_cov: float = 100.0
    patch_size: int = 8
    patch_pyrimid_level: int = 4
    normal_en: bool = True
    raycast_en: bool = False
    inverse_composition_en: bool = False
    exposure_estimate_en: bool = True
    inv_expo_cov: float = 0.1

    # IMU
    imu_en: bool = True
    imu_int_frame: int = 30
    acc_cov: float = 0.5
    gyr_cov: float = 0.3
    b_acc_cov: float = 0.0001
    b_gyr_cov: float = 0.0001

    # LIO
    lio_max_iterations: int = 5
    dept_err: float = 0.02
    beam_err: float = 0.05
    min_eigen_value: float = 0.0025
    voxel_size: float = 0.5
    max_layer: int = 2
    max_points_num: int = 50
    layer_init_num: List[int] = field(default_factory=lambda: [5, 5, 5, 5, 5])

    # Local map
    map_sliding_en: bool = False
    half_map_size: int = 100
    sliding_thresh: float = 8.0

    # UAV
    gravity_align_en: bool = False

    # Publish
    dense_map_en: bool = True
    pub_scan_num: int = 1

    # PCD save
    pcd_save_en: bool = True
    pcd_save_type: int = 0
    filter_size_pcd: float = 0.15
    pcd_save_interval: int = -1

    # Camera intrinsics (loaded from camera yaml)
    cam_fx: float = 0.0
    cam_fy: float = 0.0
    cam_cx: float = 0.0
    cam_cy: float = 0.0
    cam_width: int = 0
    cam_height: int = 0
    cam_model: str = "pinhole"
    cam_distortion: np.ndarray = field(default_factory=lambda: np.zeros(4))


def load_config(yaml_path: str, camera_yaml_path: str = None) -> Config:
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    cfg = Config()

    # Common
    common = data.get('common', {})
    cfg.lid_topic = common.get('lid_topic', cfg.lid_topic)
    cfg.imu_topic = common.get('imu_topic', cfg.imu_topic)
    cfg.img_topic = common.get('img_topic', cfg.img_topic)
    cfg.img_en = bool(common.get('img_en', 1))
    cfg.lidar_en = bool(common.get('lidar_en', 1))

    # Extrinsics
    ext = data.get('extrin_calib', {})
    if 'extrinsic_T' in ext:
        cfg.extrinsic_T = np.array(ext['extrinsic_T'], dtype=np.float64)
    if 'extrinsic_R' in ext:
        cfg.extrinsic_R = np.array(ext['extrinsic_R'], dtype=np.float64).reshape(3, 3)
    if 'Rcl' in ext:
        cfg.Rcl = np.array(ext['Rcl'], dtype=np.float64).reshape(3, 3)
    if 'Pcl' in ext:
        cfg.Pcl = np.array(ext['Pcl'], dtype=np.float64)

    # Time offsets
    time_off = data.get('time_offset', {})
    cfg.imu_time_offset = time_off.get('imu_time_offset', 0.0)
    cfg.img_time_offset = time_off.get('img_time_offset', 0.0)
    cfg.exposure_time_init = time_off.get('exposure_time_init', 0.0)

    # Preprocess
    pre = data.get('preprocess', {})
    cfg.point_filter_num = pre.get('point_filter_num', 1)
    cfg.filter_size_surf = pre.get('filter_size_surf', 0.1)
    cfg.lidar_type = pre.get('lidar_type', 1)
    cfg.scan_line = pre.get('scan_line', 6)
    cfg.blind = pre.get('blind', 0.8)

    # VIO
    vio = data.get('vio', {})
    cfg.vio_max_iterations = vio.get('max_iterations', 5)
    cfg.outlier_threshold = vio.get('outlier_threshold', 1000.0)
    cfg.img_point_cov = vio.get('img_point_cov', 100.0)
    cfg.patch_size = vio.get('patch_size', 8)
    cfg.patch_pyrimid_level = vio.get('patch_pyrimid_level', 4)
    cfg.normal_en = vio.get('normal_en', True)
    cfg.raycast_en = vio.get('raycast_en', False)
    cfg.inverse_composition_en = vio.get('inverse_composition_en', False)
    cfg.exposure_estimate_en = vio.get('exposure_estimate_en', True)
    cfg.inv_expo_cov = vio.get('inv_expo_cov', 0.1)

    # IMU
    imu = data.get('imu', {})
    cfg.imu_en = imu.get('imu_en', True)
    cfg.imu_int_frame = imu.get('imu_int_frame', 30)
    cfg.acc_cov = imu.get('acc_cov', 0.5)
    cfg.gyr_cov = imu.get('gyr_cov', 0.3)
    cfg.b_acc_cov = imu.get('b_acc_cov', 0.0001)
    cfg.b_gyr_cov = imu.get('b_gyr_cov', 0.0001)

    # LIO
    lio = data.get('lio', {})
    cfg.lio_max_iterations = lio.get('max_iterations', 5)
    cfg.dept_err = lio.get('dept_err', 0.02)
    cfg.beam_err = lio.get('beam_err', 0.05)
    cfg.min_eigen_value = lio.get('min_eigen_value', 0.0025)
    cfg.voxel_size = lio.get('voxel_size', 0.5)
    cfg.max_layer = lio.get('max_layer', 2)
    cfg.max_points_num = lio.get('max_points_num', 50)
    cfg.layer_init_num = lio.get('layer_init_num', [5, 5, 5, 5, 5])

    # Local map
    lm = data.get('local_map', {})
    cfg.map_sliding_en = lm.get('map_sliding_en', False)
    cfg.half_map_size = lm.get('half_map_size', 100)
    cfg.sliding_thresh = lm.get('sliding_thresh', 8.0)

    # UAV
    uav = data.get('uav', {})
    cfg.gravity_align_en = uav.get('gravity_align_en', False)

    # Publish
    pub = data.get('publish', {})
    cfg.dense_map_en = pub.get('dense_map_en', True)
    cfg.pub_scan_num = pub.get('pub_scan_num', 1)

    # PCD save
    pcd = data.get('pcd_save', {})
    cfg.pcd_save_en = pcd.get('pcd_save_en', True)
    cfg.pcd_save_type = pcd.get('type', 0)
    cfg.filter_size_pcd = pcd.get('filter_size_pcd', 0.15)
    cfg.pcd_save_interval = pcd.get('interval', -1)

    # Camera intrinsics from separate yaml
    if camera_yaml_path:
        _load_camera_config(camera_yaml_path, cfg)

    return cfg


def _load_camera_config(camera_yaml_path: str, cfg: Config):
    """Load camera parameters from camera YAML."""
    with open(camera_yaml_path, 'r') as f:
        cam_data = yaml.safe_load(f)

    cam = cam_data.get('cam0', cam_data)
    cfg.cam_model = cam.get('cam_model', 'pinhole')
    resolution = cam.get('cam_resolution', cam.get('resolution', [0, 0]))
    if isinstance(resolution, list) and len(resolution) >= 2:
        cfg.cam_width = int(resolution[0])
        cfg.cam_height = int(resolution[1])
    intrinsics = cam.get('cam_intrinsics', cam.get('intrinsics', []))
    if len(intrinsics) >= 4:
        cfg.cam_fx = intrinsics[0]
        cfg.cam_fy = intrinsics[1]
        cfg.cam_cx = intrinsics[2]
        cfg.cam_cy = intrinsics[3]
    dist = cam.get('cam_distortion', cam.get('distortion_coeffs', [0, 0, 0, 0]))
    cfg.cam_distortion = np.array(dist, dtype=np.float64)
