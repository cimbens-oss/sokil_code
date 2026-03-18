"""SO(3) Lie group math utilities - Rodrigues rotation, Log map, Euler angles."""
import numpy as np

def skew_sym(v):
    """Create skew-symmetric matrix from 3-vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

def exp_so3(ang):
    """Exponential map: so(3) -> SO(3) via Rodrigues formula.
    ang: 3-vector (axis-angle)."""
    ang = np.asarray(ang, dtype=np.float64)
    norm = np.linalg.norm(ang)
    I = np.eye(3)
    if norm > 1e-7:
        r = ang / norm
        K = skew_sym(r)
        return I + np.sin(norm) * K + (1.0 - np.cos(norm)) * (K @ K)
    return I

def exp_so3_dt(ang_vel, dt):
    """Exponential map with angular velocity and time step."""
    ang_vel = np.asarray(ang_vel, dtype=np.float64)
    norm = np.linalg.norm(ang_vel)
    I = np.eye(3)
    if norm > 1e-7:
        r = ang_vel / norm
        K = skew_sym(r)
        theta = norm * dt
        return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return I

def exp_so3_xyz(v1, v2, v3):
    """Exponential map from 3 scalar components."""
    return exp_so3(np.array([v1, v2, v3]))

def log_so3(R):
    """Logarithm map: SO(3) -> so(3)."""
    trace = np.clip(R.trace(), -1.0 + 1e-10, 3.0 - 1e-6)
    theta = 0.0 if trace > 3.0 - 1e-6 else np.arccos(0.5 * (trace - 1.0))
    K = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if abs(theta) < 0.001:
        return 0.5 * K
    return 0.5 * theta / np.sin(theta) * K

def rot_to_euler(R):
    """Rotation matrix to Euler angles (roll, pitch, yaw)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])

def rot_to_quat(R):
    """Rotation matrix to quaternion [x, y, z, w]."""
    tr = R.trace()
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])

def quat_to_rot(q):
    """Quaternion [x, y, z, w] to rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])


def skew_sym_batch(v):
    """Batch skew-symmetric matrices from Nx3 array. Returns Nx3x3."""
    n = v.shape[0]
    out = np.zeros((n, 3, 3))
    out[:, 0, 1] = -v[:, 2]
    out[:, 0, 2] = v[:, 1]
    out[:, 1, 0] = v[:, 2]
    out[:, 1, 2] = -v[:, 0]
    out[:, 2, 0] = -v[:, 1]
    out[:, 2, 1] = v[:, 0]
    return out
