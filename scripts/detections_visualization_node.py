#!/usr/bin/env python3
import argparse

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from as2_gates_localization.msg import KeypointDetectionArray
import message_filters


# One BGR color per gate index
_GATE_COLORS = [
    (0,   0,   255),   # red
    (0,   255,  0),    # green
    (255,  0,   0),    # blue
    (0,   255, 255),   # yellow
    (255,  0,  255),   # magenta
    (255, 255,   0),   # cyan
    (128,  0,  255),   # purple
]

# Short labels for the four inner corners
_KP_SHORT = {
    "top_left_inner":     "TL",
    "top_right_inner":    "TR",
    "bottom_right_inner": "BR",
    "bottom_left_inner":  "BL",
}


class DetectionsVisualizerNode(Node):
    def __init__(self, compressed: bool, pub_comp: bool):
        super().__init__('detections_visualizer')
        self._bridge = CvBridge()
        self._compressed = compressed
        self._pub_comp = pub_comp

        if compressed:
            img_sub = message_filters.Subscriber(
                self, CompressedImage,
                '/drone0/sensor_measurements/camera/image/compressed')
        else:
            img_sub = message_filters.Subscriber(
                self, Image,
                '/drone0/sensor_measurements/camera/image')

        det_sub = message_filters.Subscriber(
            self, KeypointDetectionArray,
            '/drone0/debug/detected_gates_data')

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, det_sub], queue_size=20, slop=0.05)
        self._sync.registerCallback(self._callback)

        self._pub = self.create_publisher(
            Image,
            '/drone0/debug/detected_gates_visualization',
            10)

        self._pub_compressed = None
        if pub_comp:
            self._pub_compressed = self.create_publisher(
                CompressedImage,
                '/drone0/debug/detected_gates_visualization/compressed',
                10)

        topics = '/drone0/debug/detected_gates_visualization'
        if pub_comp:
            topics += ' + /compressed'
        self.get_logger().info(
            f"Subscribed to {'compressed ' if compressed else ''}image + detections. "
            f"Publishing visualization on {topics}")

    def _callback(self, img_msg, det_msg):
        if self._compressed:
            img = self._bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        else:
            img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        img = self._draw_detections(img, det_msg)

        out = self._bridge.cv2_to_imgmsg(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), encoding='rgb8')
        out.header = img_msg.header
        self._pub.publish(out)

        if self._pub_compressed is not None:
            out_comp = self._bridge.cv2_to_compressed_imgmsg(img, dst_format='jpeg')
            out_comp.header = img_msg.header
            self._pub_compressed.publish(out_comp)

    def _draw_detections(self, img: np.ndarray, det_msg: KeypointDetectionArray) -> np.ndarray:
        for gate_idx, det in enumerate(det_msg.detections):
            color = _GATE_COLORS[gate_idx % len(_GATE_COLORS)]
            bb = det.bounding_box

            # Bounding box
            x1, y1, x2, y2 = int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Gate label (class id) above the bounding box
            gate_label = f"gate {det.class_id}"
            cv2.putText(img, gate_label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Keypoints
            for kp in det.keypoints:
                kx, ky = int(round(kp.x)), int(round(kp.y))
                short = _KP_SHORT.get(kp.name, kp.name)

                if kp.visible:
                    cv2.circle(img, (kx, ky), 5, color, -1)
                    # label to the right-and-up of the point, with a small background
                    # so it reads cleanly on any background
                    (tw, th), _ = cv2.getTextSize(
                        short, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    tx, ty = kx + 7, ky - 5
                    cv2.rectangle(img, (tx - 1, ty - th - 1),
                                  (tx + tw + 1, ty + 2), (0, 0, 0), -1)
                    cv2.putText(img, short, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                else:
                    # hollow circle for non-visible corners
                    cv2.circle(img, (kx, ky), 5, color, 1)

        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compressed', action='store_true',
                        help="Subscribe to CompressedImage instead of Image")
    parser.add_argument('--pub_comp', action='store_true',
                        help="Also publish CompressedImage on .../detected_gates_visualization/compressed")
    # rclpy args are passed after '--'
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = DetectionsVisualizerNode(compressed=args.compressed, pub_comp=args.pub_comp)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
