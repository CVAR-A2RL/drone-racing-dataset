import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

IMG_WIDTH, IMG_HEIGHT = 1640, 1232


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
    trans = np.array([t["x"], t["y"], t["z"]])
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
    r = extr["rotation"][rot_key]
    return trans, [r["x"], r["y"], r["z"], r["w"]]


def precompute_gate_orientations(df, gate_ids, fix_rotation):
    gate_quats, gate_mean_centers = {}, {}
    for gate_id in gate_ids:
        mean_corners = np.array([
            [df[f'gate{gate_id}_marker{m}_{ax}'].mean() for ax in ('x', 'y', 'z')]
            for m in range(1, 5)
        ])
        mean_center = mean_corners.mean(axis=0)
        gate_mean_centers[gate_id] = mean_center
        _, _, Vt = np.linalg.svd(mean_corners - mean_center)
        x_axis_3d = Vt[0]
        if np.dot(x_axis_3d, mean_corners[1] - mean_corners[0]) < 0:
            x_axis_3d = -x_axis_3d
        normal = Vt[2]
        if fix_rotation:
            z_axis = np.array([normal[0], normal[1], 0.0])
            z_axis /= np.linalg.norm(z_axis)
            y_axis = np.array([0.0, 0.0, 1.0])
            x_axis = np.cross(y_axis, z_axis)
            rot = np.column_stack([z_axis, x_axis, y_axis])
        else:
            y_axis = np.cross(normal, x_axis_3d)
            z_axis = np.cross(x_axis_3d, y_axis)
            rot = np.column_stack([z_axis, x_axis_3d, y_axis])
        gate_quats[gate_id] = Rotation.from_matrix(rot).as_quat()
    return gate_quats, gate_mean_centers


def _compute_gate_bodies(ts_us, delay_us, mocap_df, ts_arr, gate_ids,
                         gate_quats, gate_mean_centers, all_corners_local):
    row = mocap_df.iloc[_find_nearest_idx(ts_arr, ts_us + delay_us)]
    pos_drone = np.array([row['pose_position_x'], row['pose_position_y'], row['pose_position_z']])
    R_drone_inv = Rotation.from_quat([
        row['pose_orientation_x'], row['pose_orientation_y'],
        row['pose_orientation_z'], row['pose_orientation_w'],
    ]).inv()

    gate_bodies = []
    for gate_id in gate_ids:
        center = gate_mean_centers[gate_id]
        R_gate = Rotation.from_quat(gate_quats[gate_id])
        corners_earth = np.array(
            [center + R_gate.apply(lp) for lp in all_corners_local], dtype=np.float64)
        gate_normal = R_gate.apply([1.0, 0.0, 0.0])
        if np.dot(gate_normal, pos_drone - center) < 0:
            swap = [1, 0, 3, 2]
            corners_earth = corners_earth[swap + [i + 4 for i in swap]]
        gate_bodies.append(R_drone_inv.apply(corners_earth - pos_drone))
    return gate_bodies


def _apply_occlusion(candidates):
    for i in range(1, len(candidates)):
        projected_i, visible_i, _ = candidates[i]
        for j in range(i):
            projected_j = candidates[j][0]
            outer = projected_j[4:8].astype(np.float32)
            inner = projected_j[0:4].astype(np.float32)
            for k in range(8):
                if not visible_i[k]:
                    continue
                pt = (float(projected_i[k, 0]), float(projected_i[k, 1]))
                if (cv2.pointPolygonTest(outer, pt, False) >= 0 and
                        cv2.pointPolygonTest(inner, pt, False) < 0):
                    visible_i[k] = False


def _project_gates(gate_bodies, R_cam_inv, t_cam, K, dist):
    k1 = float(dist[0])
    r_crit = np.sqrt(1.0 / (-3.0 * k1)) if k1 < 0 else None

    candidates = []
    for g_idx, corners_body in enumerate(gate_bodies):
        corners_cam = R_cam_inv.apply(corners_body - t_cam)
        projected, _ = cv2.projectPoints(
            corners_cam.astype(np.float64), np.zeros(3), np.zeros(3), K, dist)
        projected = projected.reshape(-1, 2)

        z = corners_cam[:, 2]
        if r_crit is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                r_u = np.sqrt((corners_cam[:, 0] / z) ** 2 +
                              (corners_cam[:, 1] / z) ** 2)
            r_u = np.where(np.isfinite(r_u), r_u, np.inf)
            no_wraparound = r_u < r_crit
        else:
            no_wraparound = np.ones(len(corners_body), dtype=bool)

        in_front = z > 0
        visible = (
            in_front & no_wraparound
            & (projected[:, 0] >= 0) & (projected[:, 0] < IMG_WIDTH)
            & (projected[:, 1] >= 0) & (projected[:, 1] < IMG_HEIGHT)
        )

        if not np.any(visible[:4]):
            continue

        depth = float(z.mean())
        candidates.append([g_idx, projected, visible, depth])

    candidates.sort(key=lambda c: c[3])
    occ_input = [[proj, vis, depth] for _, proj, vis, depth in candidates]
    _apply_occlusion(occ_input)

    result_map = {}
    for (g_idx, _, _, _), (proj, vis, _) in zip(candidates, occ_input):
        if not np.any(vis[:4]):
            continue
        result_map[g_idx] = (proj, vis)

    return [result_map.get(i) for i in range(len(gate_bodies))]


def _draw_gate(img, proj, visible):
    inner_color = (0, 100, 255)   # orange
    outer_color = (0, 220, 255)   # yellow

    order = [i for i in range(4) if visible[i]]
    for a, b in zip(order, order[1:] + order[:1]):
        p1 = (int(round(float(proj[a][0]))), int(round(float(proj[a][1]))))
        p2 = (int(round(float(proj[b][0]))), int(round(float(proj[b][1]))))
        cv2.line(img, p1, p2, inner_color, 2, cv2.LINE_AA)

    order = [i + 4 for i in range(4) if visible[i + 4]]
    for a, b in zip(order, order[1:] + order[:1]):
        p1 = (int(round(float(proj[a][0]))), int(round(float(proj[a][1]))))
        p2 = (int(round(float(proj[b][0]))), int(round(float(proj[b][1]))))
        cv2.line(img, p1, p2, outer_color, 2, cv2.LINE_AA)

    for i in range(4):
        if visible[i] and visible[i + 4]:
            p1 = (int(round(float(proj[i][0]))), int(round(float(proj[i][1]))))
            p2 = (int(round(float(proj[i + 4][0]))), int(round(float(proj[i + 4][1]))))
            cv2.line(img, p1, p2, outer_color, 1, cv2.LINE_AA)

    for i in range(4):
        if visible[i]:
            cv2.circle(img, (int(round(float(proj[i][0]))), int(round(float(proj[i][1])))),
                       5, inner_color, -1, cv2.LINE_AA)
    for i in range(4, 8):
        if visible[i]:
            cv2.circle(img, (int(round(float(proj[i][0]))), int(round(float(proj[i][1])))),
                       4, outer_color, -1, cv2.LINE_AA)


def _put_text(img, text, y, color):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _visualize(frame_data, mocap_df, ts_arr, gate_ids, gate_quats, gate_mean_centers,
               all_corners_local, R_cam_inv, t_cam, K, dist, rect_maps=None):
    n = len(frame_data)
    idx = 0
    delay_us = 0          # current delay in microseconds (image ts + delay → pose lookup)
    step_s = 0.1          # current step size in seconds
    cv2.namedWindow('visualize_flight', cv2.WINDOW_NORMAL)

    while True:
        img_path, ts_us = frame_data[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        if rect_maps is not None:
            img = cv2.remap(img, rect_maps[0], rect_maps[1], cv2.INTER_LINEAR)

        gate_bodies = _compute_gate_bodies(
            ts_us, delay_us, mocap_df, ts_arr, gate_ids,
            gate_quats, gate_mean_centers, all_corners_local)
        gate_results = _project_gates(gate_bodies, R_cam_inv, t_cam, K, dist)
        n_visible = sum(1 for r in gate_results if r is not None)

        for result in gate_results:
            if result is None:
                continue
            proj, visible = result
            _draw_gate(img, proj, visible)

        delay_s = delay_us / 1e6
        _put_text(img, (f"[{idx + 1}/{n}]  {os.path.basename(img_path)}"
                        f"  gates: {n_visible}  |  n/p=navigate  q=quit"),
                  22, (0, 0, 0))
        _put_text(img, (f"delay: {delay_s:+.4f} s  |  step: {step_s:.4f} s"
                        f"  |  up/down=+/-step  w/s=double/half step"),
                  46, (0, 0, 0))
        cv2.imshow('visualize_flight', img)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), 83, 32):     # right arrow / space
            idx = min(idx + 1, n - 1)
        elif key in (ord('p'), 81):         # left arrow
            idx = max(idx - 1, 0)
        elif key == 82:                     # up arrow — add delay
            delay_us += int(step_s * 1e6)
        elif key == 84:                     # down arrow — reduce delay
            delay_us -= int(step_s * 1e6)
        elif key == ord('w'):               # double step
            step_s *= 2.0
        elif key == ord('s'):               # halve step
            step_s /= 2.0

    cv2.destroyAllWindows()
    print(f"Final delay: {delay_us / 1e6:+.4f} s")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize gate projections (8 corners) on flight images.")
    parser.add_argument('--flight', required=True)
    parser.add_argument('--fix-rotation', action='store_true')
    parser.add_argument('--interior-size', type=float, default=1.5)
    parser.add_argument('--exterior-size', type=float, default=2.7)
    parser.add_argument('--rectified', action='store_true')
    parser.add_argument('--start', type=float, default=None)
    parser.add_argument('--end', type=float, default=None)
    args = parser.parse_args()

    flight_type = "piloted" if "p-" in args.flight else "autonomous"
    flight_dir = os.path.join("..", "data", flight_type, args.flight)

    with open(get_calibration_path(args.flight)) as f:
        calib = json.load(f)
    K_arr = np.array(calib['mtx'])
    dist_arr = np.array(calib['dist'][0])

    t_cam, cam_quat = get_camera_extrinsics(args.flight)
    R_cam_inv = Rotation.from_quat(cam_quat).inv()

    csv_dir = os.path.join(flight_dir, 'csv_raw')
    mocap_df = pd.read_csv(glob(os.path.join(csv_dir, 'ros2bag_dump', 'drone_state_*.csv'))[0])
    gate_corners_df = pd.read_csv(glob(os.path.join(csv_dir, 'gate_corners_*.csv'))[0])

    ref_ts_us = int(mocap_df['timestamp'].iloc[0])
    cut_start_us = ref_ts_us + int(args.start * 1e6) if args.start is not None else None
    cut_end_us = ref_ts_us + int(args.end * 1e6) if args.end is not None else None
    ts_arr = mocap_df['timestamp'].values

    gate_ids = sorted({
        int(col.split('_')[0][4:])
        for col in gate_corners_df.columns if col.startswith('gate') and '_marker1_x' in col
    })
    gate_quats, gate_mean_centers = precompute_gate_orientations(gate_corners_df, gate_ids, args.fix_rotation)

    img_dir = os.path.join(flight_dir, f'camera_{args.flight}')
    images = sorted(glob(os.path.join(img_dir, '*.jpg')))

    sample = cv2.imread(images[0])
    global IMG_HEIGHT, IMG_WIDTH
    IMG_HEIGHT, IMG_WIDTH = sample.shape[:2]
    print(f"Image size: {IMG_WIDTH}×{IMG_HEIGHT}")

    if args.rectified:
        K_proj, _ = cv2.getOptimalNewCameraMatrix(
            K_arr, dist_arr, (IMG_WIDTH, IMG_HEIGHT), alpha=0)
        dist_proj = np.zeros(5)
        map1, map2 = cv2.initUndistortRectifyMap(
            K_arr, dist_arr, None, K_proj, (IMG_WIDTH, IMG_HEIGHT), cv2.CV_32FC1)
        rect_maps = (map1, map2)
    else:
        K_proj, dist_proj = K_arr, dist_arr
        rect_maps = None

    half_i = args.interior_size / 2.0
    half_o = args.exterior_size / 2.0
    all_corners_local = [
        np.array([0.0, -half_i,  half_i]),  # tli
        np.array([0.0,  half_i,  half_i]),  # tri
        np.array([0.0,  half_i, -half_i]),  # bri
        np.array([0.0, -half_i, -half_i]),  # bli
        np.array([0.0, -half_o,  half_o]),  # tlo
        np.array([0.0,  half_o,  half_o]),  # tro
        np.array([0.0,  half_o, -half_o]),  # bro
        np.array([0.0, -half_o, -half_o]),  # blo
    ]

    frame_data = []
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        ts_us = int(stem.split('_')[-1])
        if cut_start_us is not None and ts_us < cut_start_us:
            continue
        if cut_end_us is not None and ts_us > cut_end_us:
            continue
        frame_data.append((img_path, ts_us))

    print(f"Frames: {len(frame_data)}")
    if not frame_data:
        print("No frames found — check flight name and time range.")
        return

    _visualize(frame_data, mocap_df, ts_arr, gate_ids, gate_quats, gate_mean_centers,
               all_corners_local, R_cam_inv, t_cam, K_proj, dist_proj, rect_maps)


if __name__ == '__main__':
    main()
