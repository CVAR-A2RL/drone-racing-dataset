import argparse
import os
from glob import glob
import pandas as pd
import cv2
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py

from transforms3d.euler import euler2quat
import shutil
from tf2_msgs.msg import TFMessage

from gate_perception_msgs.msg import GateCornersDetection, GatesDetectionStamped

from utils import *

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
    imu_msg.header.stamp = nanosec_to_stamp(timestamp)
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
    pose_img.header.stamp = nanosec_to_stamp(timestamp)
    pose_img.pose = pose

    pose_img.header.frame_id = "earth"
    return pose_img

def create_image_msg(image_path, timestamp):
    assert len(str(timestamp)) == 19
    image = cv2.imread(image_path)
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
    image_msg.header.stamp = nanosec_to_stamp(timestamp)
    return image_msg

def create_topic(writer, topic_name, topic_type, serialization_format='cdr'):
    topic_name = topic_name
    topic = rosbag2_py.TopicMetadata(name=topic_name, type=topic_type,
                                     serialization_format=serialization_format)

    writer.create_topic(topic)


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--flight', required=True, help="Flight ID (e.g., flight-01p-ellipse)")
    parser.add_argument('--namespace', default='drone0', help="Namespace for the drone")
    parser.add_argument('-ow', '--overwrite', action='store_true', help='Overwrite output file if it exists', default=False)
    args = parser.parse_args()

    namespace = args.namespace

    flight = args.flight
    flight_type = "piloted" if "p-" in args.flight else "autonomous"
    flight_shape = flight.split("-")[-1]
    flight_dir = os.path.join("..", "data", flight_type, args.flight)
    bag_path = os.path.join(flight_dir, f"ros2bag_{args.flight}")
    image_path = os.path.join(flight_dir, "camera_" + args.flight + "/")
    label_path = os.path.join(flight_dir, "label_" + args.flight + "/")
    n_gates = 7 if flight_shape == "trackRATM" else 4

    # Create a ROS2 bag for output
    output_bag_path = os.path.join(flight_dir, "imu_cam_bag")

    # Remove output file if it exists
    if args.overwrite:
        try:
            shutil.rmtree(output_bag_path)
        except FileNotFoundError:
            pass

    # Create writer
    storage_options, converter_options = get_rosbag_options(output_bag_path, 'sqlite3')
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    # Define Topics
    IMU_TOPIC = f'{namespace}/sensor_measurements/imu'
    CAMERA_TOPIC = f'{namespace}/sensor_measurements/camera/image'
    POSE_TOPIC = f'{namespace}/self_localization/pose'
    DETECTIONS_TOPIC = f'ground_truth/gate_corner_detections'
    CAMERA_INFO_TOPIC = f'{namespace}/sensor_measurements/camera/camera_info'
    TF_TOPIC = '/tf'

    GATE_TOPICS = []
    for i in range(1, n_gates+1):
        GATE_TOPICS.append(f'ground_truth/gate{i}/pose')


    # Create Topics
    create_topic(writer, IMU_TOPIC, 'sensor_msgs/msg/Imu')
    create_topic(writer, CAMERA_TOPIC, 'sensor_msgs/msg/Image')
    create_topic(writer, POSE_TOPIC, 'geometry_msgs/msg/PoseStamped')
    create_topic(writer, DETECTIONS_TOPIC, 'gate_perception_msgs/msg/GatesDetectionStamped')
    create_topic(writer, CAMERA_INFO_TOPIC, 'sensor_msgs/msg/CameraInfo')
    create_topic(writer, TF_TOPIC, 'tf2_msgs/msg/TFMessage')

    for t in GATE_TOPICS:
        create_topic(writer, t, 'geometry_msgs/msg/PoseStamped')


    # Get camera position depending on the flight
    camera_pose = None
    with open(os.path.join("..", "camera_calibration", "drone_to_camera.json")) as f:
        camera_data = json.load(f)
        camera_pose = {}
        camera_pose["translation"] = camera_data["translation"]
        if flight_type == "piloted":
            camera_pose["rotation"] = camera_data["rotation"]["piloted"]
        else:
            camera_pose["rotation"] = camera_data["rotation"][flight_shape]   


    # Get camera calibration
    camera_calibration = None
    camera_calibration_file =  os.path.join("..", "camera_calibration", "calibration_results.json") if flight_shape in ["ellipse", "lemniscate"] else os.path.join("..", "camera_calibration", "calibration_results_trackRATM.json")
    with open(camera_calibration_file) as f:
        camera_calibration = json.load(f)


    # Load data from CSV
    data_df = pd.read_csv(f"{flight_dir}/{args.flight}_500hz_freq_sync.csv")


    imu_df = pd.DataFrame(columns=["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
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
        if(msg_type.__name__ == 'SensorImu'):
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
        if(msg_type.__name__ == 'DroneState'):
            qualisys_data = deserialize_message(data, msg_type)
            qualisys_data_list.append({
                "timestamp": qualisys_data.timestamp,
                "pose": qualisys_data.pose,
                "velocity": qualisys_data.velocity,
            })

    del reader

    imu_df = pd.DataFrame(imu_data_list)
    qualisys_df = pd.DataFrame(qualisys_data_list)


    print("Converting IMU...")
    for _, row in imu_df.iterrows():
        imu_msg = create_imu_msg(
            convert_to_nanosec(int(row["timestamp"])),
            row["accel_x"], row["accel_y"], row["accel_z"],
            row["gyro_x"], row["gyro_y"], row["gyro_z"]
        )
        writer.write(IMU_TOPIC, serialize_message(imu_msg), int(convert_to_nanosec(row["timestamp"])))
    

    print("Converting POSE...")
    for _, row in qualisys_df.iterrows():
        stamp_nanoseconds = int(convert_to_nanosec(row["timestamp"]))
        pose_msg = create_pose_msg(
            convert_to_nanosec(int(row["timestamp"])),
            row["pose"]
        )
        writer.write(POSE_TOPIC, serialize_message(pose_msg), int(convert_to_nanosec(row["timestamp"])))

        # TF
        tf_msg = TFMessage()
        tfs = static_tfs(stamp_nanoseconds, namespace, camera_pose)
        # /ns/odom -> /ns/base_link
        tf_base_link = identity_tf(stamp_nanoseconds, f"{namespace}/odom", f"{namespace}/base_link")
        tf_base_link.transform.translation.x = pose_msg.pose.position.x
        tf_base_link.transform.translation.y = pose_msg.pose.position.y
        tf_base_link.transform.translation.z = pose_msg.pose.position.z
        tf_base_link.transform.rotation = pose_msg.pose.orientation
        tfs.append(tf_base_link)
        # Publish TFMessage
        tf_msg.transforms = tfs
        writer.write(TF_TOPIC, serialize_message(tf_msg), int(stamp_nanoseconds))

    print("Converting gate poses...")
    for _, row in data_df.iterrows():
        stamp = int(convert_to_nanosec(row["timestamp"]))
        # TFs
        tf_msg = TFMessage()
        tfs = []
        for i in range(1, n_gates+1):
            gate_x = float(row[f'gate{i}_int_x'])
            gate_y = float(row[f'gate{i}_int_y'])
            gate_z = float(row[f'gate{i}_int_z'])
            gate_roll = float(row[f'gate{i}_int_roll'])
            gate_pitch = float(row[f'gate{i}_int_pitch'])
            gate_yaw = float(row[f'gate{i}_int_yaw'])
            quaternion = euler2quat(gate_roll, gate_pitch, gate_yaw, 'sxyz')
            gate_posestamped = PoseStamped()
            gate_posestamped.header.stamp = nanosec_to_stamp(stamp)
            gate_posestamped.header.frame_id = "earth"
            gate_posestamped.pose.position.x = gate_x
            gate_posestamped.pose.position.y = gate_y
            gate_posestamped.pose.position.z = gate_z
            gate_posestamped.pose.orientation.x = quaternion[1]
            gate_posestamped.pose.orientation.y = quaternion[2]
            gate_posestamped.pose.orientation.z = quaternion[3]
            gate_posestamped.pose.orientation.w = quaternion[0]
            writer.write(GATE_TOPICS[i-1], serialize_message(gate_posestamped), stamp)

            # /earth -> /gate{i}
            tf_gate = identity_tf(stamp, "earth", f"gate{i}")
            tf_gate.transform.translation.x = gate_x
            tf_gate.transform.translation.y = gate_y
            tf_gate.transform.translation.z = gate_z
            tf_gate.transform.rotation.w = quaternion[0]
            tf_gate.transform.rotation.x = quaternion[1]
            tf_gate.transform.rotation.y = quaternion[2]
            tf_gate.transform.rotation.z = quaternion[3]
            tfs.append(tf_gate)
        # Publish TFMessage
        tf_msg.transforms = tfs
        writer.write(TF_TOPIC, serialize_message(tf_msg), stamp)
    
    print("Converting IMAGE...(it may take several GBs)")
    images = sorted(glob(image_path + "*"))
    print(image_path)
    img_size = None
    for image in images:
        timestamp = image.split('_')[-1].split('.')[0]
        image_msg = create_image_msg(
            image,
            convert_to_nanosec(int(timestamp))
        )
        image_msg.header.frame_id = f"{namespace}/camera/optical_link"
        if img_size is None:
            img_size = (image_msg.width, image_msg.height)
        #print(image_msg.header.stamp.sec, image_msg.header.stamp.nanosec, 'IMAGE')
        writer.write(CAMERA_TOPIC, serialize_message(image_msg), convert_to_nanosec(int(timestamp)))

        camera_info_msg = get_camera_info(namespace, camera_calibration, img_size[0], img_size[1])

        # Publish camera_info
        camera_info_msg.header.stamp = image_msg.header.stamp
        writer.write(CAMERA_INFO_TOPIC, serialize_message(camera_info_msg), convert_to_nanosec(int(timestamp)))

    print("Converting gate detections...")
    labels = sorted(glob(label_path + "*"))
    for label in labels:
        timestamp = label.split('_')[-1].split('.')[0]
        with open(label, 'r') as f:
            lines = f.readlines()
            corner_detections = []
            for line in lines:
                line = line.strip()
                if line:
                    d = line.split(' ')
                    # print(d[7], d[10], d[13], d[16])
                    if float(d[7]) < 1 or float(d[10]) < 1 or float(d[13]) < 1 or float(d[16]) < 1:
                        continue
                    detection = GateCornersDetection()
                    detection.top_left.x = float(d[5])
                    detection.top_left.y = float(d[6])
                    detection.top_right.x = float(d[8])
                    detection.top_right.y = float(d[9])
                    detection.bottom_right.x = float(d[11])
                    detection.bottom_right.y = float(d[12])
                    detection.bottom_left.x = float(d[14])
                    detection.bottom_left.y = float(d[15])
                    corner_detections.append(detection)
            gates_detection_stamped = GatesDetectionStamped()
            gates_detection_stamped.header.stamp = nanosec_to_stamp(convert_to_nanosec(int(timestamp)))
            gates_detection_stamped.header.frame_id = f"{namespace}/camera/optical_link"
            gates_detection_stamped.corner_detections = corner_detections

            writer.write(DETECTIONS_TOPIC, serialize_message(gates_detection_stamped), convert_to_nanosec(int(timestamp)))
            
            
            

    del writer

if __name__ == "__main__":
    main()
