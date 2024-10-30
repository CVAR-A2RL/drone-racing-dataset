from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo
import json

def nanosec_to_stamp(nanosec):
    return Time(sec=nanosec // 1_000_000_000, nanosec=nanosec % 1_000_000_000)

def identity_tf(timestamp, frame_id, child_frame_id):
    assert len(str(timestamp)) == 19
    tf = TransformStamped()
    tf.header.stamp = nanosec_to_stamp(timestamp)
    tf.header.frame_id = frame_id
    tf.child_frame_id = child_frame_id
    tf.transform.translation.x = 0.0
    tf.transform.translation.y = 0.0
    tf.transform.translation.z = 0.0
    tf.transform.rotation.w = 1.0
    tf.transform.rotation.x = 0.0
    tf.transform.rotation.y = 0.0
    tf.transform.rotation.z = 0.0
    return tf

def static_tfs(stamp_nanoseconds, namespace, camera_pose):
    tfs = []
    # /earth -> /ns/map
    tf_map = identity_tf(stamp_nanoseconds, "earth", f"{namespace}/map")
    tfs.append(tf_map)
    # /ns/map -> /ns/odom
    tf_odom = identity_tf(stamp_nanoseconds, f"{namespace}/map", f"{namespace}/odom")
    tfs.append(tf_odom)
    # /ns/base_link -> /ns/camera/optical_link
    tf_camera = identity_tf(stamp_nanoseconds, f"{namespace}/base_link", f"{namespace}/camera/optical_link")
    if camera_pose is None:
        tf_camera.transform.translation.x = 0.091422
        tf_camera.transform.translation.y = 0.024722
        tf_camera.transform.translation.z = 0.055073
        tf_camera.transform.rotation.w = 0.664463
        tf_camera.transform.rotation.x = -0.2418448
        tf_camera.transform.rotation.y = 0.2418448
        tf_camera.transform.rotation.z = -0.664463
    else:
        tf_camera.transform.translation.x = camera_pose["translation"]["x"]
        tf_camera.transform.translation.y = camera_pose["translation"]["y"]
        tf_camera.transform.translation.z = camera_pose["translation"]["z"]
        tf_camera.transform.rotation.w = camera_pose["rotation"]["w"]
        tf_camera.transform.rotation.x = camera_pose["rotation"]["x"]
        tf_camera.transform.rotation.y = camera_pose["rotation"]["y"]
        tf_camera.transform.rotation.z = camera_pose["rotation"]["z"]
    tfs.append(tf_camera)
    return tfs

def get_camera_info(namespace, camera_calibration, width, height):
    camera_info = CameraInfo()
    camera_info.header.frame_id = f"{namespace}/camera/optical_link"
    camera_info.header.stamp = nanosec_to_stamp(0)
    camera_info.width = width
    camera_info.height = height
    camera_info.distortion_model = "plumb_bob"
    camera_info.d = camera_calibration["dist"][0]
    matrix = list(camera_calibration["mtx"][0])
    matrix.extend(camera_calibration["mtx"][1])
    matrix.extend(camera_calibration["mtx"][2])
    camera_info.k = matrix
    camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    camera_info.p = [matrix[0], matrix[1], matrix[2], 0.0, matrix[3], matrix[4], matrix[5], 0.0, matrix[6], matrix[7], matrix[8], 0.0]
    return camera_info