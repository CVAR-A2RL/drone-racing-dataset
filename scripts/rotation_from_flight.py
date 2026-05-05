import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

IMG_WIDTH, IMG_HEIGHT = 1640, 1232  # overridden at runtime from actual images


# ── Helpers (duplicated from create_std_bag.py) ──────────────────────────────

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
    return trans, [r["x"], r["y"], r["z"], r["w"]], rot_key


def precompute_gate_orientations(df, gate_ids, fix_rotation):
    gate_quats, gate_mean_centers, gate_mean_corners = {}, {}, {}
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
    return gate_quats, gate_mean_centers, gate_mean_corners


# ── Projection (copied exactly from create_std_bag.py, inner 4 pts only) ─────

def _apply_occlusion(candidates):
    """Copied exactly from create_std_bag.py / create_yolo_pose_dataset.py.

    candidates: list of [projected(8,2), visible(8,bool), depth], sorted closest-first.
    Inner polygon: indices 0-3 (tli,tri,bri,bli).
    Outer polygon: indices 4-7 (tlo,tro,bro,blo).
    Modifies visible arrays in place.
    """
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
    """Project pre-computed body-frame corners (8 per gate: inner[0:4] + outer[4:8]).

    Copied from create_std_bag.py: barrel-distortion guard, z>0, image bounds,
    depth sort, occlusion test.  Returns list (one entry per gate, preserving order):
      (proj(8,2), visible(8,bool), centroid_of_visible_inner(2,))  or  None.
    Error computation uses only inner corners [0:4]; outer [4:8] are for occlusion only.
    """
    k1 = float(dist[0])
    r_crit = np.sqrt(1.0 / (-3.0 * k1)) if k1 < 0 else None

    # First pass: project + visibility, collect candidates with depth
    candidates = []   # [gate_idx, projected(8,2), visible(8,bool), depth]
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

        # Gate included if at least 1 inner corner is visible
        if not np.any(visible[:4]):
            continue

        depth = float(z.mean())
        candidates.append([g_idx, projected, visible, depth])

    # Sort closest-first, apply occlusion (identical to create_std_bag.py)
    candidates.sort(key=lambda c: c[3])
    occ_input = [[proj, vis, depth] for _, proj, vis, depth in candidates]
    _apply_occlusion(occ_input)

    # Rebuild results in original gate order (None for excluded gates)
    result_map = {}
    for (g_idx, _, _, _), (proj, vis, _) in zip(candidates, occ_input):
        if not np.any(vis[:4]):
            continue
        centroid = proj[:4][vis[:4]].mean(axis=0)
        result_map[g_idx] = (proj, vis, centroid)

    return [result_map.get(i) for i in range(len(gate_bodies))]


def _match_gates(gate_results, label_gates):
    """One-to-one greedy matching: each projected gate gets the closest unassigned GT label.

    gate_results: output of _project_gates (list, may contain None).
    Returns list of (gate_idx, proj(4,2), visible(4,bool), label_gate) for matched pairs.
    """
    valid = [(i, r) for i, r in enumerate(gate_results) if r is not None]
    if not valid or not label_gates:
        return []

    # Build distance matrix: rows=projected gates, cols=GT labels
    proj_centroids = np.array([r[2] for _, r in valid])   # (n_proj, 2)
    gt_centroids   = np.array([lg['centroid'] for lg in label_gates])  # (n_gt, 2)
    dists = np.linalg.norm(
        proj_centroids[:, None, :] - gt_centroids[None, :, :], axis=2)  # (n_proj, n_gt)

    matched = []
    used_gt = set()
    # Greedy: repeatedly pick the closest remaining (proj, gt) pair
    flat = np.argsort(dists, axis=None)
    for idx in flat:
        pi, gi = divmod(int(idx), len(label_gates))
        if gi in used_gt:
            continue
        # Check this projected gate hasn't been matched yet
        if any(m[0] == valid[pi][0] for m in matched):
            continue
        used_gt.add(gi)
        gate_i, (proj, visible, _) = valid[pi]
        matched.append((gate_i, proj, visible, label_gates[gi]))

    return matched


# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_label(label_path):
    """Parse 4-keypoint YOLO label file (distorted pixel coordinates)."""
    gates = []
    if not os.path.exists(label_path):
        return gates
    with open(label_path) as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            cx = float(vals[1]) * IMG_WIDTH
            cy = float(vals[2]) * IMG_HEIGHT
            kps, vis = [], []
            for i in range(4):
                kps.append([float(vals[5 + i * 3]) * IMG_WIDTH,
                             float(vals[6 + i * 3]) * IMG_HEIGHT])
                vis.append(int(vals[7 + i * 3]) == 2)
            if any(vis):
                gates.append({'centroid': np.array([cx, cy]),
                              'kps': np.array(kps),
                              'vis': np.array(vis)})
    return gates


def _undistort_label_gates(label_gates, K_arr, dist_arr, K_proj):
    """Transform GT keypoints from distorted to rectified pixel coordinates."""
    for lg in label_gates:
        pts = lg['kps'].reshape(-1, 1, 2).astype(np.float32)
        pts_rect = cv2.undistortPoints(pts, K_arr, dist_arr, P=K_proj).reshape(-1, 2)
        lg['kps'] = pts_rect
        visible_pts = pts_rect[lg['vis']]
        if len(visible_pts):
            lg['centroid'] = visible_pts.mean(axis=0)


# ── Objective ─────────────────────────────────────────────────────────────────

def _compute_mse(frame_data, rotvec_cam, t_cam, K, dist, max_rmse=None):
    R_cam_inv = Rotation.from_rotvec(rotvec_cam).inv()
    total_sq, total_n = 0.0, 0

    for _, gate_bodies, label_gates in frame_data:
        gate_results = _project_gates(gate_bodies, R_cam_inv, t_cam, K, dist)
        frame_sq, frame_n = 0.0, 0
        for _, proj, visible, lg in _match_gates(gate_results, label_gates):
            for i, v_gt in enumerate(lg['vis']):
                if v_gt and visible[:4][i]:
                    d = proj[i] - lg['kps'][i]
                    frame_sq += float(d @ d)
                    frame_n += 1
        if frame_n == 0:
            continue
        if max_rmse is not None and (frame_sq / frame_n) ** 0.5 > max_rmse:
            continue
        total_sq += frame_sq
        total_n  += frame_n

    return total_sq / max(total_n, 1)


# ── Visualization ─────────────────────────────────────────────────────────────

_KP_NAMES  = ['tl', 'tr', 'br', 'bl']
_GT_COLOR  = (0, 220, 0)    # green  — ground truth
_PRJ_COLOR = (0, 100, 255)  # orange — projection
_ERR_COLOR = (60, 60, 255)  # red    — error line


def _draw_keypoints(img, kps, vis, color, prefix):
    for i, (kp, v) in enumerate(zip(kps, vis)):
        x, y = int(round(float(kp[0]))), int(round(float(kp[1])))
        if v:
            cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
        cv2.putText(img, f"{prefix}{_KP_NAMES[i]}", (x + 6, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def _visualize(frame_data, rotvec_cam, t_cam, K, dist, rect_maps=None):
    n = len(frame_data)
    idx = 0
    cv2.namedWindow('rotation_from_flight', cv2.WINDOW_NORMAL)

    while True:
        img_path, gate_bodies, label_gates = frame_data[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        if rect_maps is not None:
            img = cv2.remap(img, rect_maps[0], rect_maps[1], cv2.INTER_LINEAR)

        R_cam_inv = Rotation.from_rotvec(rotvec_cam).inv()
        gate_results = _project_gates(gate_bodies, R_cam_inv, t_cam, K, dist)

        total_err, total_n = 0.0, 0

        for _, proj, visible, lg in _match_gates(gate_results, label_gates):
            _draw_keypoints(img, lg['kps'], lg['vis'], _GT_COLOR, 'gt_')
            _draw_keypoints(img, proj[:4], visible[:4], _PRJ_COLOR, 'prj_')

            for i, v_gt in enumerate(lg['vis']):
                if v_gt and visible[:4][i]:
                    pt_gt  = (int(round(float(lg['kps'][i][0]))),
                              int(round(float(lg['kps'][i][1]))))
                    pt_prj = (int(round(float(proj[i][0]))),
                              int(round(float(proj[i][1]))))
                    cv2.line(img, pt_gt, pt_prj, _ERR_COLOR, 1, cv2.LINE_AA)
                    d = proj[i] - lg['kps'][i]
                    total_err += float(d @ d)
                    total_n += 1

        rmse = (total_err / max(total_n, 1)) ** 0.5
        info = (f"[{idx + 1}/{n}]  {os.path.basename(img_path)}"
                f"  RMSE: {rmse:.1f} px  |  n/p=navigate  q=quit")
        cv2.putText(img, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('rotation_from_flight', img)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), 83, 32):
            idx = min(idx + 1, n - 1)
        elif key in (ord('p'), 81):
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optimize camera rotation extrinsic from flight labels.")
    parser.add_argument('--flight', required=True)
    parser.add_argument('--fix-rotation', action='store_true',
                        help="Yaw-only gate orientation (same flag as create_std_bag.py)")
    parser.add_argument('--interior-size', type=float, default=1.5,
                        help="Inner gate side length in metres (default: 1.5)")
    parser.add_argument('--exterior-size', type=float, default=2.7,
                        help="Outer gate side length in metres (default: 2.7)")
    parser.add_argument('--rectified', action='store_true',
                        help="Project onto rectified image (zero distortion); "
                             "also undistorts GT labels and shows rectified frames")
    parser.add_argument('--max-rmse', type=float, default=None,
                        help="Exclude frames with per-frame RMSE above this threshold (default: use all frames)")
    parser.add_argument('--pitch-opt', type=float, default=5.0,
                        help="Initial step for pitch optimization in degrees (default: 5.0)")
    parser.add_argument('--yaw-opt', type=float, default=5.0,
                        help="Initial step for yaw optimization in degrees (default: 5.0)")
    parser.add_argument('--roll-opt', type=float, default=5.0,
                        help="Initial step for roll optimization in degrees (default: 5.0)")
    parser.add_argument('--iterations', type=int, default=3,
                        help="Consecutive middle wins required to finish each axis (default: 3)")
    parser.add_argument('--start', type=float, default=None)
    parser.add_argument('--end',   type=float, default=None)
    parser.add_argument('--visualize', action='store_true',
                        help="Show frames with GT + projected keypoints; skip optimization")
    args = parser.parse_args()

    flight_type = "piloted" if "p-" in args.flight else "autonomous"
    flight_dir  = os.path.join("..", "data", flight_type, args.flight)

    # Camera calibration
    with open(get_calibration_path(args.flight)) as f:
        calib = json.load(f)
    K_arr    = np.array(calib['mtx'])
    dist_arr = np.array(calib['dist'][0])

    # Prior extrinsics
    t_cam, prior_quat, rot_key = get_camera_extrinsics(args.flight)
    print(f"Rot key: {rot_key}")
    rotvec_init = Rotation.from_quat(prior_quat).as_rotvec()

    # Raw CSVs (same source as create_std_bag.py — no interpolation to camera timestamps)
    csv_dir = os.path.join(flight_dir, 'csv_raw')
    df = pd.read_csv(glob(os.path.join(csv_dir, 'ros2bag_dump', 'drone_state_*.csv'))[0])
    gate_corners_df = pd.read_csv(glob(os.path.join(csv_dir, 'gate_corners_*.csv'))[0])

    ref_ts_us    = int(df['timestamp'].iloc[0])
    cut_start_us = ref_ts_us + int(args.start * 1e6) if args.start is not None else None
    cut_end_us   = ref_ts_us + int(args.end   * 1e6) if args.end   is not None else None
    if cut_start_us is not None:
        df = df[df['timestamp'] >= cut_start_us]
    if cut_end_us is not None:
        df = df[df['timestamp'] <= cut_end_us]
    df = df.reset_index(drop=True)
    ts_arr = df['timestamp'].values

    gate_ids = sorted({
        int(col.split('_')[0][4:])
        for col in gate_corners_df.columns if col.startswith('gate') and '_marker1_x' in col
    })
    gate_quats, gate_mean_centers, _ = precompute_gate_orientations(
        gate_corners_df, gate_ids, args.fix_rotation)

    # Image / label directories
    img_dir   = os.path.join(flight_dir, f'camera_{args.flight}')
    label_dir = os.path.join(flight_dir, f'label_{args.flight}')
    images    = sorted(glob(os.path.join(img_dir, '*.jpg')))

    # Actual image dimensions
    sample = cv2.imread(images[0])
    global IMG_HEIGHT, IMG_WIDTH
    IMG_HEIGHT, IMG_WIDTH = sample.shape[:2]
    print(f"Image size: {IMG_WIDTH}×{IMG_HEIGHT}")

    # Projection camera matrix (rectified or original)
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

    # Corner local positions in gate frame — identical to create_std_bag.py
    # Gate frame: x=normal, y=width (left→right), z=up
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
    all_corners_local = inner_corners_local + outer_corners_local

    # ── Build frame_data ──────────────────────────────────────────────────────
    # Per frame: pre-compute body-frame inner corners for every gate.
    # Only the body→camera step depends on R_cam, so this is safe to cache.
    frame_data = []   # (img_path, gate_bodies: list[(4,3)], label_gates)

    for img_path in images:
        stem  = os.path.splitext(os.path.basename(img_path))[0]
        ts_us = int(stem.split('_')[-1])
        if cut_start_us is not None and ts_us < cut_start_us:
            continue
        if cut_end_us is not None and ts_us > cut_end_us:
            continue

        label_gates = parse_label(os.path.join(label_dir, stem + '.txt'))
        if not label_gates:
            continue

        # Drone pose from CSV
        # drone_rot[0..8] are stored column-major → transpose to get actual rotation matrix
        row = df.iloc[_find_nearest_idx(ts_arr, ts_us)]
        pos_drone = np.array([row['pose_position_x'], row['pose_position_y'], row['pose_position_z']])
        R_drone_inv = Rotation.from_quat([
            row['pose_orientation_x'], row['pose_orientation_y'],
            row['pose_orientation_z'], row['pose_orientation_w'],
        ]).inv()

        # Inner corners in earth frame → body frame (identical to create_std_bag.py)
        gate_bodies = []
        for gate_id in gate_ids:
            center = gate_mean_centers[gate_id]
            R_gate = Rotation.from_quat(gate_quats[gate_id])

            # 8 corners: inner[0:4] + outer[4:8] — identical to create_std_bag.py
            corners_earth = np.array(
                [center + R_gate.apply(lp) for lp in all_corners_local],
                dtype=np.float64)

            # Reverse-gate check: swap left↔right on both inner and outer halves
            gate_normal = R_gate.apply([1.0, 0.0, 0.0])
            if np.dot(gate_normal, pos_drone - center) < 0:
                swap = [1, 0, 3, 2]
                corners_earth = corners_earth[swap + [i + 4 for i in swap]]

            gate_bodies.append(R_drone_inv.apply(corners_earth - pos_drone))

        # GT labels: undistort to rectified space if needed
        if args.rectified:
            _undistort_label_gates(label_gates, K_arr, dist_arr, K_proj)

        frame_data.append((img_path, gate_bodies, label_gates))

    n_labeled = sum(sum(lg['vis']) for _, _, lgs in frame_data for lg in lgs)
    print(f"Frames with labels: {len(frame_data)},  visible keypoints: {n_labeled}")
    if n_labeled == 0:
        print("No visible keypoints — check flight name and label directory.")
        return

    if args.visualize:
        _visualize(frame_data, rotvec_init, t_cam, K_proj, dist_proj, rect_maps)
        return

    # ── Coordinate-wise hill climbing ─────────────────────────────────────────
    def evaluate(rotvec):
        return _compute_mse(frame_data, rotvec, t_cam, K_proj, dist_proj,
                            max_rmse=args.max_rmse) ** 0.5

    def qstr(rotvec):
        q = Rotation.from_rotvec(rotvec).as_quat()  # [x,y,z,w]
        return f"w={q[3]:+.6f}  x={q[0]:+.6f}  y={q[1]:+.6f}  z={q[2]:+.6f}"

    prior_rmse = evaluate(rotvec_init)
    print(f"Prior RMSE:  {prior_rmse:.2f} px\n")

    # Euler angles 'xyz': index 0=roll, 1=pitch, 2=yaw
    # Optimize pitch → yaw → roll
    euler = Rotation.from_rotvec(rotvec_init).as_euler('xyz', degrees=True)
    axes  = [('Roll',  0, args.roll_opt),
             ('Pitch', 1, args.pitch_opt),
             ('Yaw',   2, args.yaw_opt)]

    for axis_name, axis_idx, init_step in axes:
        step = init_step
        consecutive_middle = 0

        while consecutive_middle < args.iterations:
            candidates = {}
            for label, delta in [('-', -step), ('mid', 0.0), ('+', +step)]:
                e = euler.copy()
                e[axis_idx] += delta
                rv = Rotation.from_euler('xyz', e, degrees=True).as_rotvec()
                rmse = evaluate(rv)
                candidates[label] = (delta, rv, rmse)
                sign = f"{delta:+.2f}°" if label != 'mid' else '  mid  '
                print(f"  [{axis_name} {sign}]  {qstr(rv)}  RMSE: {rmse:.2f} px")

            rmse_minus = candidates['-'][2]
            rmse_mid   = candidates['mid'][2]
            rmse_plus  = candidates['+'][2]

            if rmse_plus < rmse_mid and rmse_plus < rmse_minus:
                print(f"  → moving +{step:.2f}°, step reset to {init_step:.2f}°\n")
                euler[axis_idx] += step
                step = init_step
                consecutive_middle = 0
            elif rmse_minus < rmse_mid and rmse_minus <= rmse_plus:
                print(f"  → moving -{step:.2f}°, step reset to {init_step:.2f}°\n")
                euler[axis_idx] -= step
                step = init_step
                consecutive_middle = 0
            else:
                consecutive_middle += 1
                step /= 2
                print(f"  → middle is best ({consecutive_middle}/{args.iterations}), step → {step:.3f}°\n")

        best_rmse = evaluate(Rotation.from_euler('xyz', euler, degrees=True).as_rotvec())
        print(f"  ✓ Best {axis_name}: {euler[axis_idx]:.4f}°  RMSE: {best_rmse:.2f} px\n")

    final_rotvec = Rotation.from_euler('xyz', euler, degrees=True).as_rotvec()
    final_rmse   = evaluate(final_rotvec)
    opt_quat     = Rotation.from_rotvec(final_rotvec).as_quat()  # [x,y,z,w]

    print("══ Optimized rotation ══")
    print(f"Prior RMSE:  {prior_rmse:.2f} px")
    print(f"Final RMSE:  {final_rmse:.2f} px")
    print(f'\n    "{rot_key}": {{"w": {opt_quat[3]:.10f}, "x": {opt_quat[0]:.10f}, '
          f'"y": {opt_quat[1]:.10f}, "z": {opt_quat[2]:.10f}}}')


if __name__ == '__main__':
    main()
