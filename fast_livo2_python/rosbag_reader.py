"""Rosbag reader using rosbags (pure Python, no ROS install required).

Supports both ROS1 (.bag) and ROS2 (.db3) bag formats.
Handles Livox CustomMsg, standard PointCloud2, IMU, and Image messages.
"""
import numpy as np
import struct
from states import IMUData


def _register_livox_types(typestore):
    """Register Livox custom message types with the typestore."""
    from rosbags.typesys import Stores
    livox_msg_def = """
    # Livox custom point
    msg CustomPoint
      uint32 offset_time
      float32 x
      float32 y
      float32 z
      uint8 reflectivity
      uint8 tag
      uint8 line
    """
    # Register via rosbags typesys
    try:
        typestore.register({
            'livox_ros_driver/msg/CustomPoint': (
                [],
                [
                    ('offset_time', (1, 'uint32')),
                    ('x', (1, 'float32')),
                    ('y', (1, 'float32')),
                    ('z', (1, 'float32')),
                    ('reflectivity', (1, 'uint8')),
                    ('tag', (1, 'uint8')),
                    ('line', (1, 'uint8')),
                ],
            ),
            'livox_ros_driver/msg/CustomMsg': (
                [],
                [
                    ('header', (2, 'std_msgs/msg/Header')),
                    ('timebase', (1, 'uint64')),
                    ('point_num', (1, 'uint32')),
                    ('lidar_id', (1, 'uint8')),
                    ('rsvd', (4, ((1, 'uint8'), 3))),
                    ('points', (4, ((2, 'livox_ros_driver/msg/CustomPoint'), 0))),
                ],
            ),
        })
    except Exception:
        pass  # May already be registered or API differs


def _parse_livox_custom_raw(rawdata, timestamp_ns):
    """Parse Livox CustomMsg from raw binary data (ROS1 serialization).

    Livox CustomMsg binary layout (ROS1):
    - Header: uint32 seq, uint32 stamp.secs, uint32 stamp.nsecs, uint32 frame_id_len, char[] frame_id
    - uint64 timebase
    - uint32 point_num
    - uint8 lidar_id
    - uint8[3] rsvd
    - uint32 points_array_length
    - for each point: uint32 offset_time, float32 x, float32 y, float32 z, uint8 reflectivity, uint8 tag, uint8 line
    """
    data = bytes(rawdata)
    offset = 0

    # Parse Header
    seq = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_secs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_nsecs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    frame_id_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    frame_id = data[offset:offset+frame_id_len].decode('utf-8', errors='ignore'); offset += frame_id_len

    # Parse timebase, point_num, lidar_id, rsvd
    timebase = struct.unpack_from('<Q', data, offset)[0]; offset += 8
    point_num_field = struct.unpack_from('<I', data, offset)[0]; offset += 4
    lidar_id = struct.unpack_from('<B', data, offset)[0]; offset += 1
    rsvd = struct.unpack_from('<3B', data, offset); offset += 3

    # Parse points array length
    array_len = struct.unpack_from('<I', data, offset)[0]; offset += 4

    # Each point: uint32 + 3*float32 + 3*uint8 = 4 + 12 + 3 = 19 bytes
    point_size = 19
    points = np.zeros((array_len, 3), dtype=np.float64)
    times = np.zeros(array_len, dtype=np.float64)
    reflectivities = np.zeros(array_len, dtype=np.float32)
    valid = 0

    ts_sec = stamp_secs + stamp_nsecs * 1e-9

    for i in range(array_len):
        if offset + point_size > len(data):
            break
        offset_time = struct.unpack_from('<I', data, offset)[0]; offset += 4
        x = struct.unpack_from('<f', data, offset)[0]; offset += 4
        y = struct.unpack_from('<f', data, offset)[0]; offset += 4
        z = struct.unpack_from('<f', data, offset)[0]; offset += 4
        reflectivity = struct.unpack_from('<B', data, offset)[0]; offset += 1
        tag = struct.unpack_from('<B', data, offset)[0]; offset += 1
        line = struct.unpack_from('<B', data, offset)[0]; offset += 1

        if x == 0 and y == 0 and z == 0:
            continue
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            continue

        points[valid] = [x, y, z]
        # offset_time is in nanoseconds, convert to seconds relative to scan start
        times[valid] = offset_time * 1e-9
        reflectivities[valid] = float(reflectivity)
        valid += 1

    return ts_sec, points[:valid], times[:valid], reflectivities[:valid]


def _parse_imu_raw(rawdata):
    """Parse sensor_msgs/Imu from raw ROS1 binary data."""
    data = bytes(rawdata)
    offset = 0

    # Header
    seq = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_secs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_nsecs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    frame_id_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += frame_id_len  # skip frame_id string

    # Orientation (quaternion x,y,z,w) + covariance (9 doubles)
    orientation = struct.unpack_from('<4d', data, offset); offset += 32
    offset += 72  # orientation_covariance (9 doubles)

    # Angular velocity (x,y,z) + covariance
    gyr = struct.unpack_from('<3d', data, offset); offset += 24
    offset += 72  # angular_velocity_covariance

    # Linear acceleration (x,y,z) + covariance
    acc = struct.unpack_from('<3d', data, offset); offset += 24

    ts = stamp_secs + stamp_nsecs * 1e-9
    return ts, np.array(acc), np.array(gyr)


def _parse_image_raw(rawdata):
    """Parse sensor_msgs/Image from raw ROS1 binary data."""
    import cv2
    data = bytes(rawdata)
    offset = 0

    # Header
    seq = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_secs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_nsecs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    frame_id_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += frame_id_len

    # Image fields
    h = struct.unpack_from('<I', data, offset)[0]; offset += 4
    w = struct.unpack_from('<I', data, offset)[0]; offset += 4
    encoding_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    encoding = data[offset:offset+encoding_len].decode('utf-8', errors='ignore'); offset += encoding_len
    is_bigendian = struct.unpack_from('<B', data, offset)[0]; offset += 1
    step = struct.unpack_from('<I', data, offset)[0]; offset += 4
    data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    img_data = data[offset:offset+data_len]

    ts = stamp_secs + stamp_nsecs * 1e-9

    if encoding in ('mono8', '8UC1'):
        img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w)
    elif encoding in ('bgr8', '8UC3'):
        img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w, 3)
    elif encoding in ('rgb8',):
        img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding in ('mono16', '16UC1'):
        img = np.frombuffer(img_data, dtype=np.uint16).reshape(h, w)
    else:
        try:
            bpp = data_len // (h * w)
            if bpp == 1:
                img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w)
            elif bpp == 3:
                img = np.frombuffer(img_data, dtype=np.uint8).reshape(h, w, 3)
            else:
                return ts, None
        except Exception:
            return ts, None

    return ts, img


def read_rosbag(bag_path, lid_topic, imu_topic, img_topic=None,
                lidar_time_offset=0.0, imu_time_offset=0.0, img_time_offset=0.0):
    """Read all messages from a rosbag, sorted by timestamp.

    Uses raw binary parsing to avoid rosbags deserialization issues
    with custom message types.

    Returns:
        lidar_msgs: list of (timestamp, points_Nx3, times_N, intensities_N)
        imu_msgs: list of IMUData
        img_msgs: list of (timestamp, image_ndarray)
    """
    from rosbags.rosbag1 import Reader as Reader1
    import pathlib

    bag_path = str(bag_path)

    lidar_msgs = []
    imu_msgs = []
    img_msgs = []

    topics_of_interest = {lid_topic, imu_topic}
    if img_topic:
        topics_of_interest.add(img_topic)

    print(f"[RosBag] Reading {bag_path}")
    print(f"[RosBag] Topics: LiDAR={lid_topic}, IMU={imu_topic}, Img={img_topic}")

    with Reader1(pathlib.Path(bag_path)) as reader:
        # Print available topics
        available = {}
        for c in reader.connections:
            available[c.topic] = (c.msgtype, c.msgcount)
        print(f"[RosBag] Available topics:")
        for t, (mt, mc) in available.items():
            marker = " <--" if t in topics_of_interest else ""
            print(f"  {t} [{mt}] ({mc} msgs){marker}")

        conns = [c for c in reader.connections if c.topic in topics_of_interest]

        if not conns:
            print(f"[RosBag] WARNING: No matching topics found!")
            return lidar_msgs, imu_msgs, img_msgs

        msg_count = 0
        for conn, timestamp, rawdata in reader.messages(connections=conns):
            topic = conn.topic
            msgtype = conn.msgtype

            try:
                if topic == imu_topic:
                    ts, acc, gyr = _parse_imu_raw(rawdata)
                    imu = IMUData()
                    imu.timestamp = ts + imu_time_offset
                    imu.acc = acc
                    imu.gyr = gyr
                    imu_msgs.append(imu)

                elif topic == lid_topic:
                    if 'CustomMsg' in msgtype or 'livox' in msgtype.lower():
                        ts, pts, times, intens = _parse_livox_custom_raw(rawdata, timestamp)
                    else:
                        # Try PointCloud2 parsing
                        ts = timestamp * 1e-9
                        pts, times, intens = _parse_pointcloud2_raw(rawdata)

                    if len(pts) > 0:
                        lidar_msgs.append((ts + lidar_time_offset, pts, times, intens))

                elif topic == img_topic and img_topic:
                    if 'CompressedImage' in msgtype:
                        ts, img = _parse_compressed_image_raw(rawdata)
                    else:
                        ts, img = _parse_image_raw(rawdata)
                    if img is not None:
                        img_msgs.append((ts + img_time_offset, img))

            except Exception as e:
                msg_count += 1
                if msg_count <= 5:
                    print(f"[RosBag] Error parsing {topic} ({msgtype}): {e}")
                continue

    # Sort by timestamp
    imu_msgs.sort(key=lambda x: x.timestamp)
    lidar_msgs.sort(key=lambda x: x[0])
    img_msgs.sort(key=lambda x: x[0])

    print(f"[RosBag] Loaded {len(lidar_msgs)} LiDAR, {len(imu_msgs)} IMU, {len(img_msgs)} image messages")
    return lidar_msgs, imu_msgs, img_msgs


def _parse_pointcloud2_raw(rawdata):
    """Parse PointCloud2 from raw ROS1 binary data."""
    data = bytes(rawdata)
    offset = 0

    # Header
    seq = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_secs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_nsecs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    fid_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += fid_len

    # PointCloud2 fields
    height = struct.unpack_from('<I', data, offset)[0]; offset += 4
    width = struct.unpack_from('<I', data, offset)[0]; offset += 4

    # Fields array
    num_fields = struct.unpack_from('<I', data, offset)[0]; offset += 4
    fields = {}
    for _ in range(num_fields):
        name_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
        name = data[offset:offset+name_len].decode('utf-8', errors='ignore'); offset += name_len
        f_offset = struct.unpack_from('<I', data, offset)[0]; offset += 4
        datatype = struct.unpack_from('<B', data, offset)[0]; offset += 1
        count = struct.unpack_from('<I', data, offset)[0]; offset += 4
        fields[name] = (f_offset, datatype, count)

    is_bigendian = struct.unpack_from('<B', data, offset)[0]; offset += 1
    point_step = struct.unpack_from('<I', data, offset)[0]; offset += 4
    row_step = struct.unpack_from('<I', data, offset)[0]; offset += 4
    data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    cloud_data = data[offset:offset+data_len]

    n_points = height * width
    dtype_map = {
        1: ('<B', 1), 2: ('<b', 1), 3: ('<H', 2), 4: ('<h', 2),
        5: ('<I', 4), 6: ('<i', 4), 7: ('<f', 4), 8: ('<d', 8),
    }

    def read_field(off, field_name):
        if field_name not in fields:
            return 0.0
        f_off, f_dt, _ = fields[field_name]
        fmt, sz = dtype_map.get(f_dt, ('<f', 4))
        return float(struct.unpack_from(fmt, cloud_data, off + f_off)[0])

    # Find time field
    time_field = None
    for name in ['time', 'timestamp', 't', 'offset_time', 'curvature']:
        if name in fields:
            time_field = name
            break

    # Find intensity field
    intensity_field = None
    for name in ['intensity', 'reflectivity', 'i']:
        if name in fields:
            intensity_field = name
            break

    points = np.zeros((n_points, 3), dtype=np.float64)
    times = np.zeros(n_points, dtype=np.float64)
    intensities = np.zeros(n_points, dtype=np.float32)
    valid = 0

    for i in range(n_points):
        off = i * point_step
        if off + point_step > data_len:
            break
        x = read_field(off, 'x')
        y = read_field(off, 'y')
        z = read_field(off, 'z')
        if x == 0 and y == 0 and z == 0:
            continue
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            continue
        points[valid] = [x, y, z]
        if time_field:
            t = read_field(off, time_field)
            if time_field == 'curvature':
                t = t / 1000.0
            times[valid] = t
        if intensity_field:
            intensities[valid] = read_field(off, intensity_field)
        valid += 1

    return points[:valid], times[:valid], intensities[:valid]


def _parse_compressed_image_raw(rawdata):
    """Parse CompressedImage from raw binary data."""
    import cv2
    data = bytes(rawdata)
    offset = 0

    # Header
    seq = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_secs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    stamp_nsecs = struct.unpack_from('<I', data, offset)[0]; offset += 4
    fid_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += fid_len

    # Format string
    fmt_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    offset += fmt_len

    # Data
    data_len = struct.unpack_from('<I', data, offset)[0]; offset += 4
    img_bytes = data[offset:offset+data_len]

    ts = stamp_secs + stamp_nsecs * 1e-9
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    return ts, img
