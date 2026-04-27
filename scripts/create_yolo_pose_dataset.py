import argparse
import json
import os
import shutil
from glob import glob

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import rosbag2_py

# Reorder from internal [tli, tri, bri, bli, tlo, tro, bro, blo]
#              to YOLO  [bro, blo, tro, tlo, bri, bli, tri, tli]
_YOLO_KP_ORDER = [6, 7, 5, 4, 2, 3, 1, 0]

_DATASET_YAML = """\
path: {dataset_name}
train: images
val: images

kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]

nc: 1
names:
  0: gate

task: pose

keypoint_names:
  0: bottom_right_outer
  1: bottom_left_outer
  2: top_right_outer
  3: top_left_outer
  4: bottom_right_inner
  5: bottom_left_inner
  6: top_right_inner
  7: top_left_inner
"""


# ── Helpers (duplicated from create_std_bag.py) ───────────────────────────

def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)
    return storage_options, converter_options


def _find_nearest_idx(timestamps_arr, target):
    idx = np.searchsorted(timestamps_arr, target)
    if idx == 0:
        return 0
    if idx >= len(timestamps_arr):
        return len(timestamps_arr) - 1
    if abs(timestamps_arr[idx] - target) < abs(timestamps_arr[idx - 1] - target):
        return idx
    return idx - 1


def get_calibration_path(flight_name):
    calib_dir = os.path.join("..", "camera_calibration")
    if 'trackRATM' in flight_name:
        if 'p-' in flight_name:
            return os.path.join(calib_dir, 'calib_p-trackRATM.json')
        return os.path.join(calib_dir, 'calib_a-trackRATM.json')
    return os.path.join(calib_dir, 'calib_ap-ellipse-lemniscate.json')


def get_camera_extrinsics(flight_name):
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
    return trans, [r["x"], r["y"], r["z"], r["w"]]


def precompute_gate_orientations(df, gate_ids, fix_rotation):
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
        normal = Vt[2]
        y_axis_orig = np.cross(normal, x_axis_3d)
        z_axis_orig = np.cross(x_axis_3d, y_axis_orig)
        orig_rot = np.column_stack([z_axis_orig, x_axis_3d, y_axis_orig])
        orig_gate_rots[gate_id] = Rotation.from_matrix(orig_rot).inv()
        orig_gate_quats[gate_id] = Rotation.from_matrix(orig_rot).as_quat()
        if fix_rotation:
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


# ── Projection ────────────────────────────────────────────────────────────

def _apply_occlusion(candidates):
    """Mark keypoints of far gates as not visible when they fall inside the frame
    region (outer polygon minus inner polygon) of any closer gate.

    candidates: list of [projected(8,2), visible(8,), depth], sorted closest-first.
    Modifies visible arrays in place.

    Internal corner order: [tli, tri, bri, bli, tlo, tro, bro, blo]
    Inner polygon: indices 0-3  (tli→tri→bri→bli, clockwise in image space)
    Outer polygon: indices 4-7  (tlo→tro→bro→blo, clockwise in image space)
    """
    for i in range(1, len(candidates)):
        projected_i, visible_i, _ = candidates[i]
        for j in range(i):  # gate j is closer than gate i
            projected_j = candidates[j][0]
            outer = projected_j[4:8].astype(np.float32)  # tlo, tro, bro, blo
            inner = projected_j[0:4].astype(np.float32)  # tli, tri, bri, bli
            for k in range(8):
                if not visible_i[k]:
                    continue
                pt = (float(projected_i[k, 0]), float(projected_i[k, 1]))
                if (cv2.pointPolygonTest(outer, pt, False) >= 0 and
                        cv2.pointPolygonTest(inner, pt, False) < 0):
                    visible_i[k] = False


def project_gates(
    pose_row, gate_ids,
    gate_quats, gate_mean_centers,
    inner_corners_local, outer_corners_local,
    K, dist, R_cam_inv, t_cam,
    img_width, img_height,
    min_visible_corners=3,
    noise_std=0.0,
):
    """Return list of (projected_8pts, visible_8mask) for each gate that passes the threshold.

    projected_8pts shape: (8, 2) in internal order [tli, tri, bri, bli, tlo, tro, bro, blo].
    visible_8mask  shape: (8,) bool.
    Apply _YOLO_KP_ORDER to both before writing labels.
    """
    pose = pose_row['pose']
    pos_drone = np.array([pose.position.x, pose.position.y, pose.position.z])
    R_drone_inv = Rotation.from_quat([
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w,
    ]).inv()

    # Collect (projected, visible, camera_depth) for all gates that pass initial threshold
    candidates = []
    for gate_id in gate_ids:
        center = gate_mean_centers[gate_id]
        R_gate = Rotation.from_quat(gate_quats[gate_id])

        corners_earth = np.array(
            [center + R_gate.apply(lp) for lp in inner_corners_local + outer_corners_local],
            dtype=np.float64,
        )

        # Reverse-gate check: if drone is on the negative-normal side, swap left↔right
        gate_normal = R_gate.apply([1.0, 0.0, 0.0])
        if np.dot(gate_normal, pos_drone - center) < 0:
            swap = [1, 0, 3, 2]
            corners_earth = corners_earth[swap + [i + 4 for i in swap]]

        corners_body = R_drone_inv.apply(corners_earth - pos_drone)
        corners_cam = R_cam_inv.apply(corners_body - t_cam)

        projected, _ = cv2.projectPoints(
            corners_cam, np.zeros(3), np.zeros(3), K, dist)
        projected = projected.reshape(-1, 2)

        if noise_std > 0.0:
            projected = projected + np.random.normal(0.0, noise_std, projected.shape)

        # Barrel-distortion wrap-around guard
        z = corners_cam[:, 2]
        k1 = float(dist[0])
        if k1 < 0:
            r_u = np.sqrt((corners_cam[:, 0] / z) ** 2 + (corners_cam[:, 1] / z) ** 2)
            r_crit = np.sqrt(1.0 / (-3.0 * k1))
            no_wraparound = r_u < r_crit
        else:
            no_wraparound = np.ones(8, dtype=bool)

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

    # Sort closest-first, then mark keypoints of far gates occluded by closer gate frames
    candidates.sort(key=lambda c: c[2])
    _apply_occlusion(candidates)

    # Re-filter after occlusion (some gates may fall below the threshold)
    return [(p, v) for p, v, _ in candidates if np.count_nonzero(v) >= min_visible_corners]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flight', required=True, help="Flight ID (e.g. flight-07p-lemniscate)")
    parser.add_argument('--fix-rotation', action='store_true',
                        help="Force gate orientations to yaw-only")
    parser.add_argument('--interior-size', type=float, default=1.5,
                        help="Inner gate opening dimension in meters (default: 1.5)")
    parser.add_argument('--exterior-size', type=float, default=2.7,
                        help="Outer gate frame dimension in meters (default: 2.7)")
    parser.add_argument('--rectified', action='store_true',
                        help="Project onto the rectified image (zero distortion)")
    parser.add_argument('--min-visible-corners', type=int, default=3,
                        help="Minimum corners in frame for a gate to be included (default: 3)")
    parser.add_argument('--noise-std', type=float, default=0.0,
                        help="Gaussian noise std dev in pixels added to projections (default: 0.0)")
    parser.add_argument('--start', type=float, default=None,
                        help="Discard data before this many seconds from bag start")
    parser.add_argument('--end', type=float, default=None,
                        help="Discard data after this many seconds from bag start")
    args = parser.parse_args()

    flight_type = "piloted" if "p-" in args.flight else "autonomous"
    flight_dir = os.path.join("..", "data", flight_type, args.flight)
    bag_path = os.path.join(flight_dir, f"ros2bag_{args.flight}")
    image_path = os.path.join(flight_dir, f"camera_{args.flight}/")
    dataset_name = f"yolo_pose_dataset_{args.flight}"
    out_dir = os.path.join(flight_dir, dataset_name)

    if os.path.exists(out_dir):
        answer = input(f"Output '{out_dir}' already exists. Overwrite? [Y|n] ")
        if answer.strip().lower() == 'n':
            print("Aborted.")
            return
        shutil.rmtree(out_dir)

    os.makedirs(os.path.join(out_dir, "images"))
    os.makedirs(os.path.join(out_dir, "labels"))

    # Read drone poses from rosbag
    storage_options, converter_options = get_rosbag_options(bag_path, 'sqlite3')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/perception/drone_state']))

    pose_list = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg_type = get_message(type_map[topic])
        if msg_type.__name__ == 'DroneState':
            msg = deserialize_message(data, msg_type)
            pose_list.append({"timestamp": msg.timestamp, "pose": msg.pose})
    del reader

    qualisys_df = pd.DataFrame(pose_list)
    ref_ts_us = int(qualisys_df.iloc[0]["timestamp"])
    cut_start_us = ref_ts_us + int(args.start * 1e6) if args.start is not None else None
    cut_end_us = ref_ts_us + int(args.end * 1e6) if args.end is not None else None
    pose_ts_arr = qualisys_df['timestamp'].values

    # Gate orientations from corner CSV
    corners_csv = os.path.join(flight_dir, 'csv_raw', f'gate_corners_{args.flight}.csv')
    gate_corners_df = pd.read_csv(corners_csv)
    gate_ids = sorted({
        int(col.split('_')[0][4:])
        for col in gate_corners_df.columns
        if col.startswith('gate') and '_marker1_x' in col
    })
    gate_quats, _, _, _, gate_mean_centers, _ = precompute_gate_orientations(
        gate_corners_df, gate_ids, args.fix_rotation)

    # Camera intrinsics
    calib_path = get_calibration_path(args.flight)
    with open(calib_path) as f:
        calib = json.load(f)
    K_arr = np.array(calib['mtx'])
    dist_arr = np.array(calib['dist'][0])

    first_img = cv2.imread(sorted(glob(image_path + "*"))[0])
    img_height, img_width = first_img.shape[:2]

    if args.rectified:
        K_proj, _ = cv2.getOptimalNewCameraMatrix(K_arr, dist_arr, (img_width, img_height), alpha=0)
        dist_proj = np.zeros(5)
        map1, map2 = cv2.initUndistortRectifyMap(
            K_arr, dist_arr, None, K_proj, (img_width, img_height), cv2.CV_32FC1)
    else:
        K_proj, dist_proj = K_arr, dist_arr

    # Camera extrinsics
    cam_trans, cam_quat = get_camera_extrinsics(args.flight)
    t_cam = np.array(cam_trans)
    R_cam_inv = Rotation.from_quat(cam_quat).inv()

    # Local corner positions in gate frame (x=normal, y=width, z=up)
    half_i = args.interior_size / 2.0
    half_o = args.exterior_size / 2.0
    inner_corners_local = [
        np.array([0.0, -half_i,  half_i]),  # tli
        np.array([0.0,  half_i,  half_i]),  # tri
        np.array([0.0,  half_i, -half_i]),  # bri
        np.array([0.0, -half_i, -half_i]),  # bli
    ]
    outer_corners_local = [
        np.array([0.0, -half_o,  half_o]),  # tlo
        np.array([0.0,  half_o,  half_o]),  # tro
        np.array([0.0,  half_o, -half_o]),  # bro
        np.array([0.0, -half_o, -half_o]),  # blo
    ]

    images = sorted(glob(image_path + "*"))
    n_images = 0

    print(f"Generating YOLO pose dataset for {args.flight}...")
    for image_file in images:
        timestamp = image_file.split('_')[-1].split('.')[0]
        ts_us = int(timestamp)
        if cut_start_us is not None and ts_us < cut_start_us:
            continue
        if cut_end_us is not None and ts_us > cut_end_us:
            continue

        pose_idx = _find_nearest_idx(pose_ts_arr, ts_us)
        detections = project_gates(
            qualisys_df.iloc[pose_idx], gate_ids, gate_quats, gate_mean_centers,
            inner_corners_local, outer_corners_local,
            K_proj, dist_proj, R_cam_inv, t_cam,
            img_width, img_height,
            args.min_visible_corners, args.noise_std,
        )

        stem = os.path.splitext(os.path.basename(image_file))[0]
        label_path = os.path.join(out_dir, "labels", stem + ".txt")

        lines = []
        for projected, visible in detections:
            proj_out = projected[_YOLO_KP_ORDER]
            vis_out = visible[_YOLO_KP_ORDER]

            vis_pts = projected[visible]
            x1, y1 = vis_pts[:, 0].min(), vis_pts[:, 1].min()
            x2, y2 = vis_pts[:, 0].max(), vis_pts[:, 1].max()
            cx = ((x1 + x2) / 2) / img_width
            cy = ((y1 + y2) / 2) / img_height
            bw = (x2 - x1) / img_width
            bh = (y2 - y1) / img_height

            kp_str = " ".join(
                f"{p[0] / img_width:.6f} {p[1] / img_height:.6f} {2 if v else 0}"
                for p, v in zip(proj_out, vis_out)
            )
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kp_str}")

        with open(label_path, 'w') as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        img_arr = cv2.imread(image_file)
        if args.rectified:
            img_arr = cv2.remap(img_arr, map1, map2, cv2.INTER_LINEAR)
        out_img_path = os.path.join(out_dir, "images", os.path.basename(image_file))
        cv2.imwrite(out_img_path, img_arr)
        n_images += 1

    # Write dataset.yaml
    with open(os.path.join(out_dir, "dataset.yaml"), 'w') as f:
        f.write(_DATASET_YAML.format(dataset_name=dataset_name))

    print(f"Done. {n_images} images written to {out_dir}/")


if __name__ == "__main__":
    main()
