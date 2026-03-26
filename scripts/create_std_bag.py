import argparse
import json
import os
import shutil
from glob import glob
import pandas as pd
import cv2
from sensor_msgs.msg import Imu, CameraInfo
from geometry_msgs.msg import PoseStamped
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


def create_image_msg(image_path, timestamp):
    assert len(str(timestamp)) == 19
    image = cv2.imread(image_path)
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
    image_msg.header.stamp.sec = timestamp // 1000000000
    image_msg.header.stamp.nanosec = timestamp % 1000000000
    return image_msg


def create_compressed_image_msg(image_path, timestamp):
    assert len(str(timestamp)) == 19
    image = cv2.imread(image_path)
    bridge = CvBridge()
    compressed_msg = bridge.cv2_to_compressed_imgmsg(image, dst_format='jpeg')
    compressed_msg.header.stamp.sec = timestamp // 1000000000
    compressed_msg.header.stamp.nanosec = timestamp % 1000000000
    return compressed_msg


def create_camera_info_msg(calib, width, height, timestamp):
    assert len(str(timestamp)) == 19
    msg = CameraInfo()
    msg.header.stamp.sec = timestamp // 1000000000
    msg.header.stamp.nanosec = timestamp % 1000000000
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


def get_calibration_path(flight_name):
    calib_dir = os.path.join("..", "camera_calibration")
    if 'trackRATM' in flight_name:
        if 'p-' in flight_name:
            return os.path.join(calib_dir, 'calib_p-trackRATM.json')
        else:
            return os.path.join(calib_dir, 'calib_a-trackRATM.json')
    else:
        return os.path.join(calib_dir, 'calib_ap-ellipse-lemniscate.json')


def create_topic(writer, topic_name, topic_type, serialization_format='cdr'):
    topic_name = topic_name
    topic = rosbag2_py.TopicMetadata(name=topic_name, type=topic_type,
                                     serialization_format=serialization_format)

    writer.create_topic(topic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flight', required=True, help="Flight ID (e.g., flight-01p-ellipse)")
    parser.add_argument('--compressed', action='store_true',
                        help="Save images as CompressedImage (JPEG) instead of raw Image")
    parser.add_argument('--as2', action='store_true',
                        help="Use Aerostack2 standard topic names (/drone0/...)")
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
    else:
        imu_topic = '/sensors/imu'
        pose_topic = '/perception/drone_state'
        camera_ns = '/camera'

    if args.compressed:
        image_topic = camera_ns + '/image/compressed'
        image_type = 'sensor_msgs/msg/CompressedImage'
    else:
        image_topic = camera_ns + '/image'
        image_type = 'sensor_msgs/msg/Image'

    camera_info_topic = camera_ns + '/camera_info'

    # create topics
    create_topic(writer, imu_topic, 'sensor_msgs/msg/Imu')
    create_topic(writer, image_topic, image_type)
    create_topic(writer, camera_info_topic, 'sensor_msgs/msg/CameraInfo')
    create_topic(writer, pose_topic, 'geometry_msgs/PoseStamped')
    if args.as2:
        create_topic(writer, '/drone0/debug/detected_gates_data',
                     'as2_gates_localization/msg/KeypointDetectionArray')

    print("Converting IMU...")
    for _, row in imu_df.iterrows():
        imu_msg = create_imu_msg(
            convert_to_nanosec(int(row["timestamp"])),
            row["accel_x"], row["accel_y"], row["accel_z"],
            row["gyro_x"], row["gyro_y"], row["gyro_z"]
        )
        writer.write(imu_topic, serialize_message(imu_msg),
                     int(convert_to_nanosec(row["timestamp"])))

    print("Converting POSE...")
    for _, row in qualisys_df.iterrows():
        pose_msg = create_pose_msg(
            convert_to_nanosec(int(row["timestamp"])),
            row["pose"]
        )
        writer.write(pose_topic, serialize_message(pose_msg),
                     int(convert_to_nanosec(row["timestamp"])))

    print("Converting IMAGE...(it may take several GBs)")
    images = sorted(glob(image_path + "*"))

    # Load calibration and image dimensions
    calib_path = get_calibration_path(args.flight)
    with open(calib_path) as f:
        calib = json.load(f)
    first_img = cv2.imread(images[0])
    img_height, img_width = first_img.shape[:2]

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

        if args.compressed:
            img_msg = create_compressed_image_msg(image, ts_ns)
        else:
            img_msg = create_image_msg(image, ts_ns)

        camera_info_msg = create_camera_info_msg(calib, img_width, img_height, ts_ns)

        writer.write(image_topic, serialize_message(img_msg), ts_ns)
        writer.write(camera_info_topic, serialize_message(camera_info_msg), ts_ns)

        if args.as2:
            det_msg = create_keypoint_detection_array_msg(
                label_map.get(timestamp), img_width, img_height, ts_ns)
            writer.write('/drone0/debug/detected_gates_data', serialize_message(det_msg), ts_ns)

    del writer


if __name__ == "__main__":
    main()
