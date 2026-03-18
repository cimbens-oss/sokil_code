"""LiDAR point cloud preprocessing - filtering, downsampling."""
import numpy as np


def preprocess_pointcloud(points, times, blind=0.8, point_filter_num=1):
    """Filter and subsample raw LiDAR point cloud.

    Args:
        points: Nx3 array
        times: N array of offset times (seconds)
        blind: minimum range to filter
        point_filter_num: keep every Nth point

    Returns:
        filtered_points (Mx3), filtered_times (M)
    """
    if len(points) == 0:
        return points, times

    # Range filter
    ranges = np.linalg.norm(points, axis=1)
    mask = ranges > blind
    # Remove NaN/inf
    mask &= np.all(np.isfinite(points), axis=1)
    points = points[mask]
    times = times[mask]

    # Subsample
    if point_filter_num > 1:
        idx = np.arange(0, len(points), point_filter_num)
        points = points[idx]
        times = times[idx]

    return points, times


def voxel_downsample(points, voxel_size=0.1):
    """Voxel grid downsampling (vectorized).

    Args:
        points: Nx3 array
        voxel_size: size of each voxel

    Returns:
        downsampled Mx3 array
    """
    if len(points) == 0 or voxel_size <= 0:
        return points

    # Quantize to voxel grid
    keys = np.floor(points / voxel_size).astype(np.int64)

    # Use structured array for unique voxel identification
    # Encode 3 ints into a single unique key via linear combination
    # Shift to non-negative first
    keys_shifted = keys - keys.min(axis=0)
    dims = keys_shifted.max(axis=0) + 1
    linear_keys = (keys_shifted[:, 0] * dims[1] * dims[2] +
                   keys_shifted[:, 1] * dims[2] +
                   keys_shifted[:, 2])

    # Sort by key for grouping
    sort_idx = np.argsort(linear_keys)
    sorted_keys = linear_keys[sort_idx]
    sorted_pts = points[sort_idx]

    # Find group boundaries
    diff = np.diff(sorted_keys)
    boundaries = np.where(diff != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(sorted_keys)]])

    n_voxels = len(boundaries) - 1
    result = np.zeros((n_voxels, 3))
    for i in range(n_voxels):
        s, e = boundaries[i], boundaries[i + 1]
        result[i] = sorted_pts[s:e].mean(axis=0)

    return result
