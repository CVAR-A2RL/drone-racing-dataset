import argparse
import json
import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, CameraInfo
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from as2_gates_localization.msg import KeypointDetectionArray, KeypointDetection
from as2_msgs.msg import UInt16MultiArrayStamped
from yolo_inference_cpp.msg import BoundingBox as YoloBoundingBox, Keypoint as YoloKeypoint
from cv_bridge import CvBridge
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py


def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def convert_to_nanosec(timestamp_us):
    return timestamp_us * 1000


def create_imu_msg(timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
    assert len(str(timestamp)) == 19
    imu_msg = Imu()
    imu_msg.header.stamp.sec = timestamp // 1000000000
    imu_msg.header.stamp.nanosec = timestamp % 1000000000
    imu_msg.linear_acceleration.x = accel_x
    imu_msg.linear_acceleration.y = accel_y
    imu_msg.linear_acceleration.z = accel_z
    imu_msg.angular_velocity.x = gyro_x
    imu_msg.angular_velocity.y = gyro_y
    imu_msg.angular_velocity.z = gyro_z
    return imu_msg


def create_twist_msg(timestamp, twist):
    assert len(str(timestamp)) == 19
    msg = TwistStamped()
    msg.header.stamp.sec = timestamp // 1000000000
    msg.header.stamp.nanosec = timestamp % 1000000000
    msg.twist = twist
    return msg


def create_pose_msg(timestamp, pose):
    assert len(str(timestamp)) == 19
    pose_img = PoseStamped()
    pose_img.header.stamp.sec = timestamp // 1000000000
    pose_img.header.stamp.nanosec = timestamp % 1000000000
    pose_img.pose = pose
    return pose_img


def create_image_msg(image_path, timestamp, frame_id=''):
    assert len(str(timestamp)) == 19
    return create_image_msg_from_array(cv2.imread(image_path), timestamp, frame_id)


def create_image_msg_from_array(image, timestamp, frame_id=''):
    assert len(str(timestamp)) == 19
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
    image_msg.header.stamp.sec = timestamp // 1000000000
    image_msg.header.stamp.nanosec = timestamp % 1000000000
    image_msg.header.frame_id = frame_id
    return image_msg


def create_compressed_image_msg(image_path, timestamp, frame_id=''):
    assert len(str(timestamp)) == 19
    return create_compressed_image_msg_from_array(cv2.imread(image_path), timestamp, frame_id)


def create_compressed_image_msg_from_array(image, timestamp, frame_id=''):
    assert len(str(timestamp)) == 19
    bridge = CvBridge()
    compressed_msg = bridge.cv2_to_compressed_imgmsg(image, dst_format='jpeg')
    compressed_msg.header.stamp.sec = timestamp // 1000000000
    compressed_msg.header.stamp.nanosec = timestamp % 1000000000
    compressed_msg.header.frame_id = frame_id
    return compressed_msg


def create_camera_info_msg(calib, width, height, timestamp, frame_id=''):
    assert len(str(timestamp)) == 19
    msg = CameraInfo()
    msg.header.stamp.sec = timestamp // 1000000000
    msg.header.stamp.nanosec = timestamp % 1000000000
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = 'plumb_bob'
    K = calib['mtx']
    msg.k = [K[0][0], K[0][1], K[0][2],
             K[1][0], K[1][1], K[1][2],
             K[2][0], K[2][1], K[2][2]]
    msg.d = calib['dist'][0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [K[0][0], 0.0, K[0][2], 0.0,
             0.0, K[1][1], K[1][2], 0.0,
             0.0, 0.0, 1.0, 0.0]
    return msg


def create_rectified_camera_info_msg(new_K, width, height, timestamp, frame_id=''):
    assert len(str(timestamp)) == 19
    msg = CameraInfo()
    msg.header.stamp.sec = timestamp // 1000000000
    msg.header.stamp.nanosec = timestamp % 1000000000
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = 'plumb_bob'
    msg.k = [new_K[0][0], new_K[0][1], new_K[0][2],
             new_K[1][0], new_K[1][1], new_K[1][2],
             new_K[2][0], new_K[2][1], new_K[2][2]]
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [new_K[0][0], 0.0, new_K[0][2], 0.0,
             0.0, new_K[1][1], new_K[1][2], 0.0,
             0.0, 0.0, 1.0, 0.0]
    return msg


_KEYPOINT_NAMES = ["top_left_inner", "top_right_inner", "bottom_right_inner", "bottom_left_inner"]
_OUTER_KEYPOINT_NAMES = ["top_left_outer", "top_right_outer",
                         "bottom_right_outer", "bottom_left_outer"]


def _apply_occlusion(candidates):
    """Mark keypoints of far gates as not visible when they fall inside the frame
    region (outer polygon minus inner polygon) of any closer gate.

    candidates: list of [projected(N,2), visible(N,), depth], sorted closest-first.
    Requires 8 corners per gate (computed_corners mode).
    Modifies visible arrays in place.

    Internal corner order: [tli, tri, bri, bli, tlo, tro, bro, blo]
    Inner polygon: indices 0-3  (tli→tri→bri→bli)
    Outer polygon: indices 4-7  (tlo→tro→bro→blo)
    """
    for i in range(1, len(candidates)):
        projected_i, visible_i, _ = candidates[i]
        for j in range(i):  # gate j is closer than gate i
            projected_j = candidates[j][0]
            outer = projected_j[4:8].astype(np.float32)  # tlo, tro, bro, blo
            inner = projected_j[0:4].astype(np.float32)  # tli, tri, bri, bli
            for k in range(len(visible_i)):
                if not visible_i[k]:
                    continue
                pt = (float(projected_i[k, 0]), float(projected_i[k, 1]))
                if (cv2.pointPolygonTest(outer, pt, False) >= 0 and
                        cv2.pointPolygonTest(inner, pt, False) < 0):
                    visible_i[k] = False


def _find_nearest_idx(timestamps_arr, target):
    """Return index of closest value in a sorted numpy array."""
    idx = np.searchsorted(timestamps_arr, target)
    if idx == 0:
        return 0
    if idx >= len(timestamps_arr):
        return len(timestamps_arr) - 1
    if abs(timestamps_arr[idx] - target) < abs(timestamps_arr[idx - 1] - target):
        return idx
    return idx - 1


def create_keypoint_detection_array_from_3d(
    pose_row, gate_ids,
    gate_quats, gate_mean_centers, gate_mean_corners,
    computed_corners, inner_corners_local, outer_corners_local,
    K, dist, R_cam_inv, t_cam,
    img_width, img_height, timestamp_ns,
    min_visible_corners=3,
    noise_std=0.0,
    gate_depth=0.0,
    frame_id='',
):
    """Create KeypointDetectionArray by reprojecting 3D gate corners into the image.

    Uses flight-mean gate positions (stable, dropout-free) rather than per-frame CSV
    data. Per-frame CSV has mocap dropouts that shift individual markers, causing wild
    jumps in the projected corners.

    Uses the same corners as the TF tree:
    - Without --computed-corners: mean CSV mocap corners (4 inner).
    - With --computed-corners: computed inner + outer corners (8 total), reconstructed
      in world frame as mean_center + R_gate * local_pos, matching what is published in /tf.

    Coordinate chain: earth → drone body (from pose) → camera (from extrinsics) → 2D.
    Only gates with at least one corner inside the image frame are included.
    """
    assert len(str(timestamp_ns)) == 19
    msg = KeypointDetectionArray()
    msg.header.stamp.sec = timestamp_ns // 1000000000
    msg.header.stamp.nanosec = timestamp_ns % 1000000000
    msg.header.frame_id = frame_id

    pose = pose_row['pose']
    pos_drone = np.array([pose.position.x, pose.position.y, pose.position.z])
    R_drone_inv = Rotation.from_quat([
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w,
    ]).inv()

    kp_names = _KEYPOINT_NAMES + _OUTER_KEYPOINT_NAMES if computed_corners else _KEYPOINT_NAMES

    # Collect candidates: [projected, visible, depth]
    candidates = []
    for gate_id in gate_ids:
        center = gate_mean_centers[gate_id]

        R_gate = Rotation.from_quat(gate_quats[gate_id])
        gate_normal = R_gate.apply([1.0, 0.0, 0.0])
        dot = np.dot(gate_normal, pos_drone - center)

        if computed_corners:
            # Reconstruct world-frame positions from gate frame, identical to the TF tree
            corners_earth = np.array(
                [center + R_gate.apply(lp) for lp in inner_corners_local + outer_corners_local],
                dtype=np.float64,
            )
        else:
            corners_earth = gate_mean_corners[gate_id].astype(np.float64)

        # Shift corners onto the visible face of the gate (gate has physical depth).
        # The sign of the offset matches the side the drone is on.
        if gate_depth != 0.0:
            corners_earth += gate_normal * (gate_depth * np.sign(dot))

        # Gate normal is the X-axis of the gate frame in world coordinates.
        # If the drone is on the negative-normal side (viewing from behind), left and right
        # are mirrored in the image — swap corners so labels match their image positions.
        if dot < 0:
            swap = [1, 0, 3, 2]
            full_swap = swap + [i + 4 for i in swap] if computed_corners else swap
            corners_earth = corners_earth[full_swap]

        # Transform: earth → drone body → camera
        corners_body = R_drone_inv.apply(corners_earth - pos_drone)
        corners_cam = R_cam_inv.apply(corners_body - t_cam)

        # Project with lens distortion; shape (N, 1, 2) → (N, 2)
        projected, _ = cv2.projectPoints(
            corners_cam, np.zeros(3), np.zeros(3), K, dist)
        projected = projected.reshape(-1, 2)

        if noise_std > 0.0:
            projected = projected + np.random.normal(0.0, noise_std, projected.shape)

        # Guard against distortion polynomial wrap-around (barrel distortion, k1 < 0).
        # r_d = r_u*(1 + k1*r_u² + ...) has a maximum at r_u_crit = sqrt(1/(-3*k1))
        # and folds back for larger radii, mapping out-of-FOV points inside the image.
        # Legitimate edge points have r_u < r_crit; wrap-around artifacts have r_u > r_crit.
        # Note: checking undistorted pixel bounds would be wrong — barrel distortion makes
        # the undistorted projection of edge pixels fall outside the distorted image bounds.
        z = corners_cam[:, 2]
        k1 = float(dist[0])
        if k1 < 0:
            r_u = np.sqrt((corners_cam[:, 0] / z) ** 2 + (corners_cam[:, 1] / z) ** 2)
            r_crit = np.sqrt(1.0 / (-3.0 * k1))
            no_wraparound = r_u < r_crit
        else:
            no_wraparound = np.ones(len(corners_cam), dtype=bool)

        in_front = corners_cam[:, 2] > 0
        visible = (
            in_front & no_wraparound
            & (projected[:, 0] >= 0) & (projected[:, 0] < img_width)
            & (projected[:, 1] >= 0) & (projected[:, 1] < img_height)
        )

        if np.count_nonzero(visible) < min_visible_corners:
            continue

        depth = float(corners_cam[:, 2].mean())
        candidates.append([projected, visible, depth])

    # Sort closest-first; apply frame occlusion (requires inner + outer corners)
    candidates.sort(key=lambda c: c[2])
    if computed_corners:
        _apply_occlusion(candidates)

    # Build detections, re-filtering gates that fall below threshold after occlusion
    for projected, visible, _ in candidates:
        if np.count_nonzero(visible) < min_visible_corners:
            continue

        vis_pts = projected[visible]
        det = KeypointDetection()
        det.label = "gate"
        det.class_id = 0
        det.confidence = 1.0

        det.bounding_box = YoloBoundingBox()
        det.bounding_box.x1 = float(vis_pts[:, 0].min())
        det.bounding_box.y1 = float(vis_pts[:, 1].min())
        det.bounding_box.x2 = float(vis_pts[:, 0].max())
        det.bounding_box.y2 = float(vis_pts[:, 1].max())
        det.bounding_box.confidence = 1.0

        for i, kp_name in enumerate(kp_names):
            kp = YoloKeypoint()
            kp.name = kp_name
            kp.x = float(projected[i, 0])
            kp.y = float(projected[i, 1])
            kp.visible = bool(visible[i])
            kp.confidence = 1.0 if visible[i] else 0.0
            det.keypoints.append(kp)

        msg.detections.append(det)

    return msg


def create_keypoint_detection_array_msg(label_file, img_width, img_height, timestamp, frame_id=''):
    assert len(str(timestamp)) == 19
    msg = KeypointDetectionArray()
    msg.header.stamp.sec = timestamp // 1000000000
    msg.header.stamp.nanosec = timestamp % 1000000000
    msg.header.frame_id = frame_id

    if label_file is None:
        return msg  # no detections this frame

    with open(label_file) as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue
            class_id = int(values[0])
            bb_cx, bb_cy, bb_w, bb_h = [float(v) for v in values[1:5]]
            kps_raw = [float(v) for v in values[5:]]

            det = KeypointDetection()
            det.label = "gate"
            det.class_id = class_id
            det.confidence = 1.0

            det.bounding_box = YoloBoundingBox()
            det.bounding_box.x1 = (bb_cx - bb_w / 2) * img_width
            det.bounding_box.y1 = (bb_cy - bb_h / 2) * img_height
            det.bounding_box.x2 = (bb_cx + bb_w / 2) * img_width
            det.bounding_box.y2 = (bb_cy + bb_h / 2) * img_height
            det.bounding_box.confidence = 1.0

            for i in range(4):
                kx, ky, kvis = kps_raw[i * 3], kps_raw[i * 3 + 1], kps_raw[i * 3 + 2]
                kp = YoloKeypoint()
                kp.name = _KEYPOINT_NAMES[i]
                kp.x = kx * img_width
                kp.y = ky * img_height
                kp.visible = (kvis == 2)
                kp.confidence = 1.0 if kvis == 2 else 0.0
                det.keypoints.append(kp)

            msg.detections.append(det)
    return msg


_CORNER_NAMES = ['tli', 'tri', 'bri', 'bli']        # inner corners, marker1..marker4
_OUTER_CORNER_NAMES = ['tlo', 'tro', 'bro', 'blo']  # outer corners (computed mode only)

# Default RC channel values (16 channels): roll, pitch, yaw, throttle, arm, ?, offboard, ...
_RC_DEFAULTS = [1500, 1500, 1500, 990, 990, 990, 990, 990,
                1500, 1500, 1500, 1980, 1500, 1500, 1500, 1500]


def create_rc_msg(timestamp_ns, data):
    assert len(str(timestamp_ns)) == 19
    msg = UInt16MultiArrayStamped()
    msg.stamp.sec = timestamp_ns // 1000000000
    msg.stamp.nanosec = timestamp_ns % 1000000000
    msg.layout.dim = []
    msg.layout.data_offset = 0
    msg.data = list(data)
    return msg


def precompute_gate_orientations(df, gate_ids, fix_rotation):
    """Return (gate_quats, gate_rots, orig_gate_quats, orig_gate_rots) from corner CSV data.

    gate_quats/gate_rots are the active orientations (yaw-only if fix_rotation, else full SVD).
    orig_gate_quats/orig_gate_rots are always the full SVD orientations.
    All computed from mean corner positions across the flight (stable, noise-free).
    """
    gate_rots, gate_quats = {}, {}
    orig_gate_rots, orig_gate_quats = {}, {}
    gate_mean_centers, gate_mean_corners = {}, {}
    for gate_id in gate_ids:
        mean_corners = np.array([
            [df[f'gate{gate_id}_marker{m}_{ax}'].mean() for ax in ('x', 'y', 'z')]
            for m in range(1, 5)
        ])
        mean_center = mean_corners.mean(axis=0)
        gate_mean_centers[gate_id] = mean_center
        gate_mean_corners[gate_id] = mean_corners
        _, _, Vt = np.linalg.svd(mean_corners - mean_center)
        x_axis_3d = Vt[0]
        if np.dot(x_axis_3d, mean_corners[1] - mean_corners[0]) < 0:
            x_axis_3d = -x_axis_3d
        # Full 3D orientation from SVD (always computed).
        # Axis convention: X=normal (old Z), Y=width (old X), Z=up (old Y).
        normal = Vt[2]
        y_axis_orig = np.cross(normal, x_axis_3d)
        z_axis_orig = np.cross(x_axis_3d, y_axis_orig)
        orig_rot = np.column_stack([z_axis_orig, x_axis_3d, y_axis_orig])
        orig_gate_rots[gate_id] = Rotation.from_matrix(orig_rot).inv()
        orig_gate_quats[gate_id] = Rotation.from_matrix(orig_rot).as_quat()
        if fix_rotation:
            # Yaw-only: project gate normal (Vt[2]) onto XY plane.
            # Vt[2] (min variance = normal) is stable and nearly horizontal for a
            # vertical gate; Vt[0] is degenerate for a square gate and unreliable.
            # Axis convention: X=normal (old Z), Y=width (old X), Z=up (old Y).
            z_axis = np.array([normal[0], normal[1], 0.0])
            z_axis /= np.linalg.norm(z_axis)
            y_axis = np.array([0.0, 0.0, 1.0])
            x_axis = np.cross(y_axis, z_axis)
            fixed_rot = np.column_stack([z_axis, x_axis, y_axis])
            gate_rots[gate_id] = Rotation.from_matrix(fixed_rot).inv()
            gate_quats[gate_id] = Rotation.from_matrix(fixed_rot).as_quat()
        else:
            gate_rots[gate_id] = orig_gate_rots[gate_id]
            gate_quats[gate_id] = orig_gate_quats[gate_id]
    return gate_quats, gate_rots, orig_gate_quats, orig_gate_rots, gate_mean_centers, gate_mean_corners


def write_gate_tf_messages(writer, flight_dir, flight_name,
                           computed_corners=False,
                           interior_size=1.5,
                           exterior_size=2.7,
                           fix_rotation=False,
                           static_gates=False):
    """Write gate center and corner transforms to /tf (and optionally /tf_static).

    With static_gates=True, gate center TFs are published once to /tf_static using
    the flight-mean positions instead of per-frame to /tf.  Corners and original_gate
    frames are unaffected and continue to be written per-frame to /tf.
    """
    csv_path = os.path.join(flight_dir, 'csv_raw', f'gate_corners_{flight_name}.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    gate_ids = sorted({
        int(col.split('_')[0][4:])
        for col in df.columns if col.startswith('gate') and '_marker1_x' in col
    })

    gate_quats, gate_rots, orig_gate_quats, orig_gate_rots, gate_mean_centers, _ = \
        precompute_gate_orientations(df, gate_ids, fix_rotation)

    if static_gates:
        first_ts = convert_to_nanosec(int(df['timestamp'].iloc[0]))
        static_transforms = [
            create_transform_stamped(
                'earth', f'gate{gate_id - 1}',
                gate_mean_centers[gate_id], gate_quats[gate_id], first_ts)
            for gate_id in gate_ids
        ]
        writer.write('/tf_static', serialize_message(create_tf_msg(static_transforms)), first_ts)

    # Precompute corner positions in gate frame for computed mode.
    # Gate frame: x=normal (into gate), y=width (left→right), z=up
    if computed_corners:
        half_i = interior_size / 2.0
        half_o = exterior_size / 2.0
        inner_corners_local = [
            np.array([0.0, -half_i, half_i]),  # tli
            np.array([0.0, half_i, half_i]),  # tri
            np.array([0.0, half_i, -half_i]),  # bri
            np.array([0.0, -half_i, -half_i]),  # bli
        ]
        outer_corners_local = [
            np.array([0.0, -half_o, half_o]),  # tlo
            np.array([0.0, half_o, half_o]),  # tro
            np.array([0.0, half_o, -half_o]),  # bro
            np.array([0.0, -half_o, -half_o]),  # blo
        ]

    print("Converting GATE TF...")
    for _, row in df.iterrows():
        ts_ns = convert_to_nanosec(int(row['timestamp']))
        transforms = []
        for gate_id in gate_ids:
            gate_frame = f'gate{gate_id - 1}'
            corners = np.array([
                [row[f'gate{gate_id}_marker{m}_{ax}'] for ax in ('x', 'y', 'z')]
                for m in range(1, 5)
            ])
            center = corners.mean(axis=0)
            if not static_gates:
                transforms.append(create_transform_stamped(
                    'earth', gate_frame, center, gate_quats[gate_id], ts_ns))
            if fix_rotation:
                orig_frame = f'original_{gate_frame}'
                transforms.append(create_transform_stamped(
                    'earth', orig_frame, center, orig_gate_quats[gate_id], ts_ns))
                for corner_pos, corner_name in zip(corners, _CORNER_NAMES):
                    local_pos = orig_gate_rots[gate_id].apply(corner_pos - center)
                    transforms.append(create_transform_stamped(
                        orig_frame, f'{orig_frame}/{corner_name}', local_pos, [0., 0., 0., 1.], ts_ns))
            if not computed_corners:
                for corner_pos, corner_name in zip(corners, _CORNER_NAMES):
                    local_pos = gate_rots[gate_id].apply(corner_pos - center)
                    transforms.append(create_transform_stamped(
                        gate_frame, f'{gate_frame}/{corner_name}', local_pos, [0., 0., 0., 1.], ts_ns))
            else:
                for local_pos, corner_name in zip(inner_corners_local, _CORNER_NAMES):
                    transforms.append(create_transform_stamped(
                        gate_frame, f'{gate_frame}/{corner_name}', local_pos, [0., 0., 0., 1.], ts_ns))
                for local_pos, corner_name in zip(outer_corners_local, _OUTER_CORNER_NAMES):
                    transforms.append(create_transform_stamped(
                        gate_frame, f'{gate_frame}/{corner_name}', local_pos, [0., 0., 0., 1.], ts_ns))
        writer.write('/tf', serialize_message(create_tf_msg(transforms)), ts_ns)


def get_camera_extrinsics(flight_name):
    """Return (translation [x,y,z], quaternion [x,y,z,w]) for drone→camera transform."""
    extr_path = os.path.join("..", "camera_calibration", "drone_to_camera.json")
    with open(extr_path) as f:
        extr = json.load(f)
    t = extr["translation"]
    trans = [t["x"], t["y"], t["z"]]
    if "p-" in flight_name and "trackRATM" in flight_name:
        rot_key = "piloted_trackRATM"
    elif "trackRATM" in flight_name:
        rot_key = "trackRATM"
    elif "p-" in flight_name:
        rot_key = "piloted"
    elif "lemniscate" in flight_name:
        rot_key = "lemniscate"
    else:
        rot_key = "ellipse"
    print(f"Rot key for {flight_name}: {rot_key}")
    r = extr["rotation"][rot_key]
    quat = [r["x"], r["y"], r["z"], r["w"]]
    return trans, quat


def get_calibration_path(flight_name):
    calib_dir = os.path.join("..", "camera_calibration")
    if 'trackRATM' in flight_name:
        if 'p-' in flight_name:
            return os.path.join(calib_dir, 'calib_p-trackRATM.json')
        else:
            return os.path.join(calib_dir, 'calib_a-trackRATM.json')
    else:
        return os.path.join(calib_dir, 'calib_ap-ellipse-lemniscate.json')


def create_transform_stamped(parent, child, trans, quat, timestamp):
    ts = TransformStamped()
    ts.header.stamp.sec = timestamp // 1000000000
    ts.header.stamp.nanosec = timestamp % 1000000000
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = float(trans[0])
    ts.transform.translation.y = float(trans[1])
    ts.transform.translation.z = float(trans[2])
    ts.transform.rotation.x = float(quat[0])
    ts.transform.rotation.y = float(quat[1])
    ts.transform.rotation.z = float(quat[2])
    ts.transform.rotation.w = float(quat[3])
    return ts


def create_tf_msg(transforms):
    msg = TFMessage()
    msg.transforms = transforms
    return msg


# QoS profile string for topics that require TRANSIENT_LOCAL durability (e.g. /tf_static)
_TRANSIENT_LOCAL_QOS = (
    "- history: 3\n"
    "  depth: 0\n"
    "  reliability: 1\n"
    "  durability: 1\n"
    "  deadline:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  lifespan:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  liveliness: 1\n"
    "  liveliness_lease_duration:\n    sec: 2147483647\n    nsec: 4294967295\n"
    "  avoid_ros_namespace_conventions: false\n"
)


def create_topic(writer, topic_name, topic_type, serialization_format='cdr',
                 offered_qos_profiles=''):
    topic = rosbag2_py.TopicMetadata(name=topic_name, type=topic_type,
                                     serialization_format=serialization_format,
                                     offered_qos_profiles=offered_qos_profiles)
    writer.create_topic(topic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flight', required=True, help="Flight ID (e.g., flight-01p-ellipse)")
    parser.add_argument('--compressed', action='store_true',
                        help="Save images as CompressedImage (JPEG) instead of raw Image")
    parser.add_argument('--as2', action='store_true',
                        help="Use Aerostack2 standard topic names (/drone0/...)")
    parser.add_argument('--rectified', action='store_true',
                        help="Also publish rectified images on camera_rectified namespace")
    parser.add_argument('--fix-rotation', action='store_true',
                        help="Force gate TF orientations to yaw-only (roll=0, pitch=0)")
    parser.add_argument('--computed-corners', action='store_true',
                        help="Compute 8 corner TFs per gate from --interior-size / --exterior-size "
                             "instead of using raw CSV mocap corners")
    parser.add_argument('--static-map', action='store_true',
                        help="Publish the earth→drone0/map TF to /tf_static (once) instead of "
                             "per-frame to /tf.")
    parser.add_argument('--static-gates', action='store_true',
                        help="Publish gate center TFs to /tf_static (once, using flight-mean "
                             "positions) instead of per-frame to /tf. Corners and original_gate "
                             "frames are unaffected.")
    parser.add_argument('--interior-size', type=float, default=1.5,
                        help="Inner gate opening dimension in meters (default: 1.5)")
    parser.add_argument('--exterior-size', type=float, default=2.7,
                        help="Outer gate frame dimension in meters (default: 2.7)")
    parser.add_argument('--labels-from-3d', action='store_true',
                        help="Generate detected_gates_data by reprojecting 3D gate corners "
                             "instead of reading YOLO label files. Uses the same corners as "
                             "the TF tree (CSV corners, or computed corners if --computed-corners).")
    parser.add_argument('--rectified-points', action='store_true',
                        help="Project 3D points onto the rectified image (use with --labels-from-3d). "
                             "Uses the optimal rectified camera matrix with zero distortion, matching "
                             "the rectified image published with --rectified.")
    parser.add_argument('--min-visible-corners', type=int, default=3,
                        help="Minimum number of corners inside the image for a gate to be "
                             "included when using --labels-from-3d (default: 3)")
    parser.add_argument('--noise-std', type=float, default=0.0,
                        help="Standard deviation (pixels) of Gaussian noise added to projected "
                             "keypoints when using --labels-from-3d (default: 0.0, no noise)")
    parser.add_argument('--gate-depth', type=float, default=0.0,
                        help="Depth offset of the visible gate face from the gate center plane "
                             "in metres when using --labels-from-3d (default: 0.0). Sign is set "
                             "automatically based on which side of the gate the drone is on.")
    parser.add_argument('--start', type=float, default=None,
                        help="Discard data before this many seconds from bag start (default: keep all)")
    parser.add_argument('--end', type=float, default=None,
                        help="Discard data after this many seconds from bag start (default: keep all)")
    parser.add_argument('--arm', type=float, default=0.0,
                        help="Write an arm RC message (data[4]=2010) at bag_start + this many "
                             "seconds (default: 0.0)")
    parser.add_argument('--offboard', type=float, default=0.0,
                        help="Write an offboard RC message (data[6]=2010) at bag_start + this "
                             "many seconds (default: 0.0)")
    args = parser.parse_args()

    if args.offboard < args.arm:
        parser.error(f"--offboard ({args.offboard}) must be >= --arm ({args.arm})")

    flight_type = "piloted" if "p-" in args.flight else "autonomous"
    flight_dir = os.path.join("..", "data", flight_type, args.flight)
    bag_path = os.path.join(flight_dir, f"ros2bag_{args.flight}")
    image_path = os.path.join(flight_dir, "camera_" + args.flight + "/")

    imu_df = pd.DataFrame(columns=["timestamp", "accel_x", "accel_y",
                          "accel_z", "gyro_x", "gyro_y", "gyro_z"])
    qualisys_df = pd.DataFrame(columns=["timestamp", "pose", "twist"])

    # Read IMU data from rosbag
    storage_options, converter_options = get_rosbag_options(bag_path, 'sqlite3')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    storage_filter = rosbag2_py.StorageFilter(topics=['/sensors/imu', '/perception/drone_state'])
    reader.set_filter(storage_filter)

    imu_data_list = []
    qualisys_data_list = []

    while reader.has_next():
        (topic, data, _) = reader.read_next()
        msg_type = get_message(type_map[topic])
        if (msg_type.__name__ == 'SensorImu'):
            imu_data = deserialize_message(data, msg_type)
            imu_data_list.append({
                "timestamp": imu_data.timestamp,
                "accel_x": imu_data.accel_x,
                "accel_y": imu_data.accel_y,
                "accel_z": imu_data.accel_z,
                "gyro_x": imu_data.gyro_x,
                "gyro_y": imu_data.gyro_y,
                "gyro_z": imu_data.gyro_z,
            })
        if (msg_type.__name__ == 'DroneState'):
            qualisys_data = deserialize_message(data, msg_type)
            qualisys_data_list.append({
                "timestamp": qualisys_data.timestamp,
                "pose": qualisys_data.pose,
                "velocity": qualisys_data.velocity,
            })

    del reader

    imu_df = pd.DataFrame(imu_data_list)
    qualisys_df = pd.DataFrame(qualisys_data_list)

    # Compute time window bounds in microseconds relative to first IMU timestamp
    ref_ts_us = int(imu_df.iloc[0]["timestamp"])
    cut_start_us = ref_ts_us + int(args.start * 1e6) if args.start is not None else None
    cut_end_us = ref_ts_us + int(args.end * 1e6) if args.end is not None else None

    # Create a ROS2 bag for output
    output_bag_path = os.path.join(flight_dir, "imu_cam_bag")

    if os.path.exists(output_bag_path):
        answer = input(f"Output bag '{output_bag_path}' already exists. Overwrite? [Y|n] ")
        if answer.strip().lower() == 'n':
            print("Aborted.")
            return
        shutil.rmtree(output_bag_path)

    storage_options, converter_options = get_rosbag_options(output_bag_path, 'sqlite3')
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    if args.as2:
        imu_topic = '/drone0/sensor_measurements/imu'
        pose_topic = '/drone0/self_localization/pose'
        twist_topic = '/drone0/self_localization/twist'
        camera_ns = '/drone0/sensor_measurements/camera'
        camera_frame_id = 'drone0/camera/camera_link'
    else:
        imu_topic = '/sensors/imu'
        pose_topic = '/perception/drone_state'
        twist_topic = None
        camera_ns = '/camera'
        camera_frame_id = 'camera_link'

    if args.compressed:
        image_topic = camera_ns + '/image/compressed'
        image_type = 'sensor_msgs/msg/CompressedImage'
    else:
        image_topic = camera_ns + '/image'
        image_type = 'sensor_msgs/msg/Image'

    camera_info_topic = camera_ns + '/camera_info'

    rect_ns = camera_ns.replace('/camera', '/camera_rectified')
    rect_image_topic = rect_ns + ('/image/compressed' if args.compressed else '/image')
    rect_info_topic = rect_ns + '/camera_info'

    # create topics
    create_topic(writer, imu_topic, 'sensor_msgs/msg/Imu')
    create_topic(writer, image_topic, image_type)
    create_topic(writer, camera_info_topic, 'sensor_msgs/msg/CameraInfo')
    if args.rectified:
        create_topic(writer, rect_image_topic, image_type)
        create_topic(writer, rect_info_topic, 'sensor_msgs/msg/CameraInfo')
    create_topic(writer, pose_topic, 'geometry_msgs/PoseStamped')
    if twist_topic:
        create_topic(writer, twist_topic, 'geometry_msgs/msg/TwistStamped')
    if args.as2:
        create_topic(writer, '/drone0/debug/detected_gates_data',
                     'as2_gates_localization/msg/KeypointDetectionArray')
        create_topic(writer, '/tf_static', 'tf2_msgs/msg/TFMessage',
                     offered_qos_profiles=_TRANSIENT_LOCAL_QOS)
        create_topic(writer, '/tf', 'tf2_msgs/msg/TFMessage')
        create_topic(writer, '/drone0/debug/rc/read',
                     'as2_msgs/msg/UInt16MultiArrayStamped')

    print("Converting IMU...")
    for _, row in imu_df.iterrows():
        ts_us = int(row["timestamp"])
        if cut_start_us is not None and ts_us < cut_start_us:
            continue
        if cut_end_us is not None and ts_us > cut_end_us:
            continue
        imu_msg = create_imu_msg(
            convert_to_nanosec(int(row["timestamp"])),
            row["accel_x"], row["accel_y"], row["accel_z"],
            row["gyro_x"], row["gyro_y"], row["gyro_z"]
        )
        writer.write(imu_topic, serialize_message(imu_msg),
                     int(convert_to_nanosec(row["timestamp"])))

    # Compute bag start timestamp (nanoseconds) for RC message offsets
    bag_start_ns = convert_to_nanosec(cut_start_us if cut_start_us is not None else ref_ts_us)

    if args.as2:
        first_pose = qualisys_df.iloc[0]["pose"]
        p0 = np.array([first_pose.position.x, first_pose.position.y, first_pose.position.z])
        q0 = np.array([first_pose.orientation.x, first_pose.orientation.y,
                       first_pose.orientation.z, first_pose.orientation.w])
        R0_inv = Rotation.from_quat(q0).inv()

        first_ts = convert_to_nanosec(int(qualisys_df.iloc[0]["timestamp"]))
        cam_trans, cam_quat = get_camera_extrinsics(args.flight)
        ts_base_cam = create_transform_stamped(
            'drone0/base_link', 'drone0/camera/camera_link', cam_trans, cam_quat, first_ts)
        static_tfs = [ts_base_cam]
        if args.static_map:
            ts_earth_map = create_transform_stamped('earth', 'drone0/map', p0, q0, first_ts)
            static_tfs.append(ts_earth_map)
        writer.write('/tf_static', serialize_message(create_tf_msg(static_tfs)), first_ts)

        # Write a base RC message at bag start with default values (arm/offboard off)
        writer.write('/drone0/debug/rc/read',
                     serialize_message(create_rc_msg(bag_start_ns, _RC_DEFAULTS)), bag_start_ns)

        # Write arm and offboard RC messages at their designated timestamps.
        # If both land on the same nanosecond, arm is written before offboard.
        arm_ts_ns = bag_start_ns + int(args.arm * 1e9)
        offboard_ts_ns = bag_start_ns + int(args.offboard * 1e9)

        arm_data = list(_RC_DEFAULTS)
        arm_data[4] = 2010
        offboard_data = list(arm_data)
        offboard_data[6] = 2010

        if arm_ts_ns == offboard_ts_ns:
            writer.write('/drone0/debug/rc/read',
                         serialize_message(create_rc_msg(arm_ts_ns, arm_data)), arm_ts_ns)
            writer.write('/drone0/debug/rc/read',
                         serialize_message(create_rc_msg(offboard_ts_ns, offboard_data)), offboard_ts_ns)
        else:
            for ts_ns, data in sorted([(arm_ts_ns, arm_data), (offboard_ts_ns, offboard_data)]):
                writer.write('/drone0/debug/rc/read',
                             serialize_message(create_rc_msg(ts_ns, data)), ts_ns)

    print("Converting POSE...")
    for _, row in qualisys_df.iterrows():
        ts_us = int(row["timestamp"])
        if cut_start_us is not None and ts_us < cut_start_us:
            continue
        if cut_end_us is not None and ts_us > cut_end_us:
            continue
        ts_ns = convert_to_nanosec(int(row["timestamp"]))
        pose_msg = create_pose_msg(ts_ns, row["pose"])
        writer.write(pose_topic, serialize_message(pose_msg), ts_ns)
        if twist_topic:
            twist_msg = create_twist_msg(ts_ns, row["velocity"])
            writer.write(twist_topic, serialize_message(twist_msg), ts_ns)

        if args.as2:
            pose = row["pose"]
            pt = np.array([pose.position.x, pose.position.y, pose.position.z])
            qt = np.array([pose.orientation.x, pose.orientation.y,
                           pose.orientation.z, pose.orientation.w])
            trans = R0_inv.apply(pt - p0)
            rot = (R0_inv * Rotation.from_quat(qt)).as_quat()

            ts_map_odom = create_transform_stamped(
                'drone0/map', 'drone0/odom', [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], ts_ns)
            ts_odom_base = create_transform_stamped(
                'drone0/odom', 'drone0/base_link', trans, rot, ts_ns)
            dynamic_tfs = [ts_map_odom, ts_odom_base]
            if not args.static_map:
                ts_earth_map = create_transform_stamped('earth', 'drone0/map', p0, q0, ts_ns)
                dynamic_tfs.insert(0, ts_earth_map)
            writer.write('/tf', serialize_message(create_tf_msg(dynamic_tfs)), ts_ns)

    if args.as2:
        write_gate_tf_messages(writer, flight_dir, args.flight,
                               args.computed_corners, args.interior_size, args.exterior_size,
                               args.fix_rotation, args.static_gates)

    print("Converting IMAGE...(it may take several GBs)")
    images = sorted(glob(image_path + "*"))

    # Load calibration and image dimensions
    calib_path = get_calibration_path(args.flight)
    with open(calib_path) as f:
        calib = json.load(f)
    first_img = cv2.imread(images[0])
    img_height, img_width = first_img.shape[:2]

    # Precompute rectification maps (needed for --rectified image and/or --rectified-points)
    K_arr = np.array(calib['mtx'])
    dist_arr = np.array(calib['dist'][0])
    if args.rectified or args.rectified_points:
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K_arr, dist_arr, (img_width, img_height), alpha=0)
    if args.rectified:
        map1, map2 = cv2.initUndistortRectifyMap(
            K_arr, dist_arr, None, new_K, (img_width, img_height), cv2.CV_32FC1)

    # Build timestamp → label file map
    label_map = {}
    if args.as2 and not args.labels_from_3d:
        label_dir = os.path.join(flight_dir, "label_" + args.flight)
        for lf in glob(os.path.join(label_dir, "*.txt")):
            ts = os.path.basename(lf).split('_')[-1].split('.')[0]
            label_map[ts] = lf

    # Precompute data needed for 3D reprojection labels
    if args.as2 and args.labels_from_3d:
        corners_csv = os.path.join(flight_dir, 'csv_raw', f'gate_corners_{args.flight}.csv')
        gate_corners_df = pd.read_csv(corners_csv)
        gate_ids_3d = sorted({
            int(col.split('_')[0][4:])
            for col in gate_corners_df.columns
            if col.startswith('gate') and '_marker1_x' in col
        })
        if args.rectified_points:
            K_proj = new_K                  # optimal rectified camera matrix
            dist_proj = np.zeros(5)         # rectified image has no distortion
        else:
            K_proj = K_arr
            dist_proj = dist_arr
        cam_trans_3d, cam_quat_3d = get_camera_extrinsics(args.flight)
        t_cam_3d = np.array(cam_trans_3d)
        R_cam_inv_3d = Rotation.from_quat(cam_quat_3d).inv()
        pose_ts_arr = qualisys_df['timestamp'].values

        gate_quats_3d, _, _, _, gate_mean_centers_3d, gate_mean_corners_3d = \
            precompute_gate_orientations(gate_corners_df, gate_ids_3d, args.fix_rotation)

        if args.computed_corners:
            half_i = args.interior_size / 2.0
            half_o = args.exterior_size / 2.0
            inner_local_3d = [
                np.array([0.0, -half_i, half_i]),
                np.array([0.0, half_i, half_i]),
                np.array([0.0, half_i, -half_i]),
                np.array([0.0, -half_i, -half_i]),
            ]
            outer_local_3d = [
                np.array([0.0, -half_o, half_o]),
                np.array([0.0, half_o, half_o]),
                np.array([0.0, half_o, -half_o]),
                np.array([0.0, -half_o, -half_o]),
            ]
        else:
            inner_local_3d = outer_local_3d = None

    for image in images:
        timestamp = image.split('_')[-1].split('.')[0]
        ts_us = int(timestamp)
        if cut_start_us is not None and ts_us < cut_start_us:
            continue
        if cut_end_us is not None and ts_us > cut_end_us:
            continue
        ts_ns = convert_to_nanosec(ts_us)
        img_arr = cv2.imread(image)

        if args.compressed:
            img_msg = create_compressed_image_msg_from_array(img_arr, ts_ns, camera_frame_id)
        else:
            img_msg = create_image_msg_from_array(img_arr, ts_ns, camera_frame_id)

        camera_info_msg = create_camera_info_msg(
            calib, img_width, img_height, ts_ns, camera_frame_id)

        writer.write(image_topic, serialize_message(img_msg), ts_ns)
        writer.write(camera_info_topic, serialize_message(camera_info_msg), ts_ns)

        if args.rectified:
            rect_arr = cv2.remap(img_arr, map1, map2, cv2.INTER_LINEAR)
            if args.compressed:
                rect_msg = create_compressed_image_msg_from_array(rect_arr, ts_ns, camera_frame_id)
            else:
                rect_msg = create_image_msg_from_array(rect_arr, ts_ns, camera_frame_id)
            rect_info_msg = create_rectified_camera_info_msg(
                new_K, img_width, img_height, ts_ns, camera_frame_id)
            writer.write(rect_image_topic, serialize_message(rect_msg), ts_ns)
            writer.write(rect_info_topic, serialize_message(rect_info_msg), ts_ns)

        if args.as2:
            if args.labels_from_3d:
                pose_idx = _find_nearest_idx(pose_ts_arr, int(timestamp))
                det_msg = create_keypoint_detection_array_from_3d(
                    qualisys_df.iloc[pose_idx],
                    gate_ids_3d, gate_quats_3d,
                    gate_mean_centers_3d, gate_mean_corners_3d,
                    args.computed_corners, inner_local_3d, outer_local_3d,
                    K_proj, dist_proj, R_cam_inv_3d, t_cam_3d,
                    img_width, img_height, ts_ns,
                    args.min_visible_corners,
                    args.noise_std,
                    args.gate_depth,
                    frame_id=camera_frame_id,
                )
            else:
                det_msg = create_keypoint_detection_array_msg(
                    label_map.get(timestamp), img_width, img_height, ts_ns,
                    frame_id=camera_frame_id)
            writer.write('/drone0/debug/detected_gates_data', serialize_message(det_msg), ts_ns)

    del writer


if __name__ == "__main__":
    main()
