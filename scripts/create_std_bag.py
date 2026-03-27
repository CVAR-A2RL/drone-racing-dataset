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
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from as2_gates_localization.msg import KeypointDetectionArray, KeypointDetection
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


def create_keypoint_detection_array_msg(label_file, img_width, img_height, timestamp):
    assert len(str(timestamp)) == 19
    msg = KeypointDetectionArray()
    msg.header.stamp.sec = timestamp // 1000000000
    msg.header.stamp.nanosec = timestamp % 1000000000

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


_CORNER_NAMES = ['tli', 'tri', 'bri', 'bli']  # marker1..marker4, matches _KEYPOINT_NAMES order


def write_gate_tf_messages(writer, flight_dir, flight_name):
    """Write per-frame gate center and corner transforms to /tf."""
    csv_path = os.path.join(flight_dir, 'csv_raw', f'gate_corners_{flight_name}.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    gate_ids = sorted({
        int(col.split('_')[0][4:])
        for col in df.columns if col.startswith('gate') and '_marker1_x' in col
    })

    # Precompute orientation per gate from mean positions (stable, noise-free)
    gate_rots = {}
    gate_quats = {}
    for gate_id in gate_ids:
        mean_corners = np.array([
            [df[f'gate{gate_id}_marker{m}_{ax}'].mean() for ax in ('x', 'y', 'z')]
            for m in range(1, 5)
        ])
        mean_center = mean_corners.mean(axis=0)
        _, _, Vt = np.linalg.svd(mean_corners - mean_center)
        x_axis = Vt[0]
        normal = Vt[2]
        y_axis = np.cross(normal, x_axis)
        if np.dot(x_axis, mean_corners[1] - mean_corners[0]) < 0:
            x_axis = -x_axis
            y_axis = -y_axis
        normal = np.cross(x_axis, y_axis)
        rot_matrix = np.column_stack([x_axis, y_axis, normal])
        gate_rots[gate_id] = Rotation.from_matrix(rot_matrix).inv()
        gate_quats[gate_id] = Rotation.from_matrix(rot_matrix).as_quat()

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
            transforms.append(create_transform_stamped(
                'earth', gate_frame, center, gate_quats[gate_id], ts_ns))
            for corner_pos, corner_name in zip(corners, _CORNER_NAMES):
                local_pos = gate_rots[gate_id].apply(corner_pos - center)
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
    if "trackRATM" in flight_name:
        rot_key = "trackRATM"
    elif "p-" in flight_name:
        rot_key = "piloted"
    elif "lemniscate" in flight_name:
        rot_key = "lemniscate"
    else:
        rot_key = "ellipse"
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
    args = parser.parse_args()

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
        camera_ns = '/drone0/sensor_measurements/camera'
        camera_frame_id = 'drone0/camera/camera_link'
    else:
        imu_topic = '/sensors/imu'
        pose_topic = '/perception/drone_state'
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
    if args.as2:
        create_topic(writer, '/drone0/debug/detected_gates_data',
                     'as2_gates_localization/msg/KeypointDetectionArray')
        create_topic(writer, '/tf_static', 'tf2_msgs/msg/TFMessage',
                     offered_qos_profiles=_TRANSIENT_LOCAL_QOS)
        create_topic(writer, '/tf', 'tf2_msgs/msg/TFMessage')

    print("Converting IMU...")
    for _, row in imu_df.iterrows():
        imu_msg = create_imu_msg(
            convert_to_nanosec(int(row["timestamp"])),
            row["accel_x"], row["accel_y"], row["accel_z"],
            row["gyro_x"], row["gyro_y"], row["gyro_z"]
        )
        writer.write(imu_topic, serialize_message(imu_msg),
                     int(convert_to_nanosec(row["timestamp"])))

    if args.as2:
        first_pose = qualisys_df.iloc[0]["pose"]
        p0 = np.array([first_pose.position.x, first_pose.position.y, first_pose.position.z])
        q0 = np.array([first_pose.orientation.x, first_pose.orientation.y,
                        first_pose.orientation.z, first_pose.orientation.w])
        R0_inv = Rotation.from_quat(q0).inv()

        first_ts = convert_to_nanosec(int(qualisys_df.iloc[0]["timestamp"]))
        ts_earth_map = create_transform_stamped('earth', 'drone0/map', p0, q0, first_ts)
        cam_trans, cam_quat = get_camera_extrinsics(args.flight)
        ts_base_cam = create_transform_stamped(
            'drone0/base_link', 'drone0/camera/camera_link', cam_trans, cam_quat, first_ts)
        writer.write('/tf_static', serialize_message(
            create_tf_msg([ts_earth_map, ts_base_cam])), first_ts)

    print("Converting POSE...")
    for _, row in qualisys_df.iterrows():
        ts_ns = convert_to_nanosec(int(row["timestamp"]))
        pose_msg = create_pose_msg(ts_ns, row["pose"])
        writer.write(pose_topic, serialize_message(pose_msg), ts_ns)

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
            writer.write('/tf', serialize_message(create_tf_msg([ts_map_odom, ts_odom_base])), ts_ns)

    if args.as2:
        write_gate_tf_messages(writer, flight_dir, args.flight)

    print("Converting IMAGE...(it may take several GBs)")
    images = sorted(glob(image_path + "*"))

    # Load calibration and image dimensions
    calib_path = get_calibration_path(args.flight)
    with open(calib_path) as f:
        calib = json.load(f)
    first_img = cv2.imread(images[0])
    img_height, img_width = first_img.shape[:2]

    # Precompute rectification maps
    if args.rectified:
        K_arr = np.array(calib['mtx'])
        dist_arr = np.array(calib['dist'][0])
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K_arr, dist_arr, (img_width, img_height), alpha=0)
        map1, map2 = cv2.initUndistortRectifyMap(
            K_arr, dist_arr, None, new_K, (img_width, img_height), cv2.CV_32FC1)

    # Build timestamp → label file map
    label_map = {}
    if args.as2:
        label_dir = os.path.join(flight_dir, "label_" + args.flight)
        for lf in glob(os.path.join(label_dir, "*.txt")):
            ts = os.path.basename(lf).split('_')[-1].split('.')[0]
            label_map[ts] = lf

    for image in images:
        timestamp = image.split('_')[-1].split('.')[0]
        ts_ns = convert_to_nanosec(int(timestamp))
        img_arr = cv2.imread(image)

        if args.compressed:
            img_msg = create_compressed_image_msg_from_array(img_arr, ts_ns, camera_frame_id)
        else:
            img_msg = create_image_msg_from_array(img_arr, ts_ns, camera_frame_id)

        camera_info_msg = create_camera_info_msg(calib, img_width, img_height, ts_ns, camera_frame_id)

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
            det_msg = create_keypoint_detection_array_msg(
                label_map.get(timestamp), img_width, img_height, ts_ns)
            writer.write('/drone0/debug/detected_gates_data', serialize_message(det_msg), ts_ns)

    del writer


if __name__ == "__main__":
    main()
