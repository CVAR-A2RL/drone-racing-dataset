import argparse
import sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import tf2_ros
from tf2_ros import TransformException
from sensor_msgs.msg import CameraInfo
from as2_gates_localization.msg import KeypointDetectionArray, KeypointDetection
from yolo_inference_cpp.msg import BoundingBox as YoloBoundingBox, Keypoint as YoloKeypoint


_KEYPOINT_NAMES = ["top_left_inner", "top_right_inner", "bottom_right_inner", "bottom_left_inner"]
_OUTER_KEYPOINT_NAMES = ["top_left_outer", "top_right_outer", "bottom_right_outer", "bottom_left_outer"]


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


def _tf_to_pos_rot(tf):
    t = tf.transform.translation
    r = tf.transform.rotation
    pos = np.array([t.x, t.y, t.z])
    rot = Rotation.from_quat([r.x, r.y, r.z, r.w])
    return pos, rot


class ProjectGatesNode(Node):
    def __init__(self, args):
        super().__init__('project_gates')

        self._gate_ids = args.gate_ids
        self._min_visible = args.min_visible_corners
        self._world_frame = args.world_frame
        self._drone_frame = args.drone_frame
        self._camera_frame = args.camera_frame

        half_i = args.interior_size / 2.0
        half_o = args.exterior_size / 2.0
        self._inner_local = [
            np.array([0.0, -half_i,  half_i]),  # tli
            np.array([0.0,  half_i,  half_i]),  # tri
            np.array([0.0,  half_i, -half_i]),  # bri
            np.array([0.0, -half_i, -half_i]),  # bli
        ]
        self._outer_local = [
            np.array([0.0, -half_o,  half_o]),  # tlo
            np.array([0.0,  half_o,  half_o]),  # tro
            np.array([0.0,  half_o, -half_o]),  # bro
            np.array([0.0, -half_o, -half_o]),  # blo
        ]

        self._rectified_points = args.rectified_points
        self._gate_depth = args.gate_depth

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        _best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._pub = self.create_publisher(KeypointDetectionArray, args.output_topic, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_cb,
                                 _best_effort)

    def _camera_info_cb(self, camera_info):
        stamp = camera_info.header.stamp

        K = np.array(camera_info.k).reshape(3, 3)
        dist = np.array(camera_info.d[:5])
        img_width = camera_info.width
        img_height = camera_info.height

        if self._rectified_points:
            K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (img_width, img_height), alpha=0)
            dist = np.zeros(5)

        try:
            drone_tf = self._tf_buffer.lookup_transform(
                self._world_frame, self._drone_frame, stamp)
            cam_tf = self._tf_buffer.lookup_transform(
                self._drone_frame, self._camera_frame, stamp)
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=1.0)
            return

        pos_drone, R_drone = _tf_to_pos_rot(drone_tf)
        R_drone_inv = R_drone.inv()
        t_cam, R_cam = _tf_to_pos_rot(cam_tf)
        R_cam_inv = R_cam.inv()

        kp_names = _KEYPOINT_NAMES + _OUTER_KEYPOINT_NAMES

        candidates = []
        for gate_id in self._gate_ids:
            try:
                gate_tf = self._tf_buffer.lookup_transform(
                    self._world_frame, f'gate{gate_id}', stamp)
            except TransformException as e:
                self.get_logger().warn(
                    f'TF lookup for gate{gate_id} failed: {e}', throttle_duration_sec=1.0)
                continue

            center, R_gate = _tf_to_pos_rot(gate_tf)
            gate_normal = R_gate.apply([1.0, 0.0, 0.0])
            dot = np.dot(gate_normal, pos_drone - center)

            corners_earth = np.array(
                [center + R_gate.apply(lp) for lp in self._inner_local + self._outer_local],
                dtype=np.float64,
            )

            if self._gate_depth != 0.0:
                corners_earth += gate_normal * (self._gate_depth * np.sign(dot))

            if dot < 0:
                swap = [1, 0, 3, 2]
                corners_earth = corners_earth[swap + [i + 4 for i in swap]]

            corners_body = R_drone_inv.apply(corners_earth - pos_drone)
            corners_cam = R_cam_inv.apply(corners_body - t_cam)

            projected, _ = cv2.projectPoints(
                corners_cam, np.zeros(3), np.zeros(3), K, dist)
            projected = projected.reshape(-1, 2)

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

            if np.count_nonzero(visible) < self._min_visible:
                continue

            depth = float(corners_cam[:, 2].mean())
            candidates.append([projected, visible, depth])

        candidates.sort(key=lambda c: c[2])
        _apply_occlusion(candidates)

        ts_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        msg = KeypointDetectionArray()
        msg.header.stamp = stamp
        msg.header.frame_id = camera_info.header.frame_id

        for projected, visible, _ in candidates:
            if np.count_nonzero(visible) < self._min_visible:
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

        self._pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description='Real-time gate projection node')
    parser.add_argument('--gate-ids', nargs='+', type=int, required=True,
                        help='Gate TF frame IDs to project, e.g. 0 1 2 3')
    parser.add_argument('--interior-size', type=float, default=1.5,
                        help='Gate inner aperture in metres (default: 1.5)')
    parser.add_argument('--exterior-size', type=float, default=2.7,
                        help='Gate outer frame size in metres (default: 2.7)')
    parser.add_argument('--min-visible-corners', type=int, default=3,
                        help='Min visible corners to include a gate (default: 3)')
    parser.add_argument('--world-frame', default='earth',
                        help='World TF frame (default: earth)')
    parser.add_argument('--drone-frame', default='drone0/base_link',
                        help='Drone body TF frame (default: drone0/base_link)')
    parser.add_argument('--camera-frame', default='drone0/camera/camera_link',
                        help='Camera TF frame (default: drone0/camera/camera_link)')
    parser.add_argument('--camera-info-topic',
                        default='/drone0/sensor_measurements/camera/camera_info',
                        help='CameraInfo input topic')
    parser.add_argument('--output-topic', default='/drone0/debug/detected_gates_data',
                        help='KeypointDetectionArray output topic')
    parser.add_argument('--gate-depth', type=float, default=0.0,
                        help='Depth offset of the visible gate face from the gate center plane '
                             'in metres (default: 0.0). Sign is set automatically based on '
                             'which side of the gate the drone is on.')
    parser.add_argument('--rectified-points', action='store_true',
                        help='Project 3D points onto the rectified image. Computes the optimal '
                             'rectified camera matrix (alpha=0) and uses zero distortion, matching '
                             'create_std_bag --rectified-points.')

    # Split ROS args from script args
    ros_args = rclpy.utilities.remove_ros_args(sys.argv)
    args = parser.parse_args(ros_args[1:])

    rclpy.init(args=sys.argv)
    node = ProjectGatesNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
