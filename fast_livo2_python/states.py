"""State group and measurement data structures for FAST-LIVO2."""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from so3_math import exp_so3_xyz, log_so3

G_m_s2 = 9.81
DIM_STATE = 19
INIT_COV = 0.01

@dataclass
class IMUData:
    timestamp: float = 0.0
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyr: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class Pose6D:
    offset_time: float = 0.0
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyr: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))

class StatesGroup:
    """19-dimensional EKF state."""
    def __init__(self):
        self.rot_end = np.eye(3)
        self.pos_end = np.zeros(3)
        self.vel_end = np.zeros(3)
        self.bias_g = np.zeros(3)
        self.bias_a = np.zeros(3)
        self.gravity = np.zeros(3)
        self.inv_expo_time = 1.0
        self.cov = np.eye(DIM_STATE) * INIT_COV
        self.cov[6, 6] = 0.00001
        self.cov[10:19, 10:19] = np.eye(9) * 0.00001

    def copy(self):
        s = StatesGroup.__new__(StatesGroup)
        s.rot_end = self.rot_end.copy()
        s.pos_end = self.pos_end.copy()
        s.vel_end = self.vel_end.copy()
        s.bias_g = self.bias_g.copy()
        s.bias_a = self.bias_a.copy()
        s.gravity = self.gravity.copy()
        s.inv_expo_time = self.inv_expo_time
        s.cov = self.cov.copy()
        return s

    def add(self, dx):
        """state + dx (19x1 vector) -> new state."""
        s = StatesGroup.__new__(StatesGroup)
        s.rot_end = self.rot_end @ exp_so3_xyz(dx[0], dx[1], dx[2])
        s.pos_end = self.pos_end + dx[3:6]
        s.inv_expo_time = self.inv_expo_time + dx[6]
        s.vel_end = self.vel_end + dx[7:10]
        s.bias_g = self.bias_g + dx[10:13]
        s.bias_a = self.bias_a + dx[13:16]
        s.gravity = self.gravity + dx[16:19]
        s.cov = self.cov.copy()
        return s

    def iadd(self, dx):
        """In-place state += dx."""
        self.rot_end = self.rot_end @ exp_so3_xyz(dx[0], dx[1], dx[2])
        self.pos_end += dx[3:6]
        self.inv_expo_time += dx[6]
        self.vel_end += dx[7:10]
        self.bias_g += dx[10:13]
        self.bias_a += dx[13:16]
        self.gravity += dx[16:19]

    def diff(self, other):
        """self - other -> 19x1 vector."""
        a = np.zeros(DIM_STATE)
        rotd = other.rot_end.T @ self.rot_end
        a[0:3] = log_so3(rotd)
        a[3:6] = self.pos_end - other.pos_end
        a[6] = self.inv_expo_time - other.inv_expo_time
        a[7:10] = self.vel_end - other.vel_end
        a[10:13] = self.bias_g - other.bias_g
        a[13:16] = self.bias_a - other.bias_a
        a[16:19] = self.gravity - other.gravity
        return a

@dataclass
class PointWithVar:
    point_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_i: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    var_nostate: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    body_var: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    var: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    point_crossmat: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class PointToPlane:
    point_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    plane_var: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    body_cov: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    layer: int = 0
    d: float = 0.0
    dis_to_plane: float = 0.0

@dataclass
class MeasureGroup:
    vio_time: float = 0.0
    lio_time: float = 0.0
    imu: List[IMUData] = field(default_factory=list)
    img: Optional[np.ndarray] = None

# EKF state flags
WAIT = 0
VIO_FLAG = 1
LIO_FLAG = 2
LO_FLAG = 3

# SLAM modes
ONLY_LO = 0
ONLY_LIO = 1
LIVO = 2

# LiDAR types
AVIA = 1
VELO16 = 2
OUST64 = 3
L515 = 4
XT32 = 5
