import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
import scipy.optimize
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

def _compute_mse(frame_data, rotvec_cam, t_cam, K, dist, max_rmse=None, include_paths=None):
    R_cam_inv = Rotation.from_rotvec(rotvec_cam).inv()
    total_sq, total_n = 0.0, 0

    for img_path, gate_bodies, label_gates in frame_data:
        if include_paths is not None and img_path not in include_paths:
            continue
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


def _rmse_fixed_matching(object_points, image_points, K, dist):
    """RMSE using fixed pre-matched correspondences (camera-frame 3D, zero rvec/tvec).
    Avoids gate-remapping artifacts from dynamic matching in _compute_mse."""
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    total_sq, total_n = 0.0, 0
    for obj, img in zip(object_points, image_points):
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, dist)
        diff = proj.reshape(-1, 2) - img
        total_sq += float((diff * diff).sum())
        total_n  += len(obj)
    return (total_sq / max(total_n, 1)) ** 0.5


# ── Direct optimisation helpers ───────────────────────────────────────────────

def _pack_x0(K, dist, calibrate):
    parts = []
    if calibrate in ('both', 'K'):
        parts += [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    if calibrate in ('both', 'dist'):
        parts += list(dist.flatten()[:5])
    return np.array(parts, dtype=np.float64)


def _unpack_K_dist(params, K0, dist0, calibrate):
    K = K0.copy()
    dist = dist0.copy()
    i = 0
    if calibrate in ('both', 'K'):
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = params[0], params[1], params[2], params[3]
        i = 4
    if calibrate in ('both', 'dist'):
        dist.flat[:5] = params[i:i + 5]
    return K, dist


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


def _visualize_planar_calib(calib_obj, calib_img, frame_paths, K_prior, dist_prior,
                            K_ref, dist_ref, rect_maps=None):
    """Show each gate observation used for planar calibration: GT (green), prior (orange), refined (cyan)."""
    n = len(calib_obj)
    idx = 0
    cv2.namedWindow('planar_calib_gates', cv2.WINDOW_NORMAL)

    while True:
        img = cv2.imread(frame_paths[idx])
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        if rect_maps is not None:
            img = cv2.remap(img, rect_maps[0], rect_maps[1], cv2.INTER_LINEAR)

        obj = calib_obj[idx]
        gt  = calib_img[idx]

        # Per-view pose estimated from GT labels using each K
        _, rvec_p, tvec_p = cv2.solvePnP(obj, gt.reshape(-1, 1, 2), K_prior, dist_prior)
        _, rvec_r, tvec_r = cv2.solvePnP(obj, gt.reshape(-1, 1, 2), K_ref,   dist_ref)
        pts_prior,   _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec_p, tvec_p, K_prior, dist_prior)
        pts_refined, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec_r, tvec_r, K_ref,   dist_ref)
        pts_prior   = pts_prior.reshape(-1, 2)
        pts_refined = pts_refined.reshape(-1, 2)

        prior_rmse   = float(np.sqrt(np.mean((pts_prior   - gt) ** 2)))
        refined_rmse = float(np.sqrt(np.mean((pts_refined - gt) ** 2)))

        # GT green, prior orange, refined cyan
        for i, name in enumerate(_KP_NAMES):
            gx, gy = int(round(gt[i, 0])),          int(round(gt[i, 1]))
            px, py = int(round(pts_prior[i, 0])),   int(round(pts_prior[i, 1]))
            rx, ry = int(round(pts_refined[i, 0])), int(round(pts_refined[i, 1]))
            cv2.circle(img, (gx, gy), 5, _GT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 5, _PRJ_COLOR,  2, cv2.LINE_AA)
            cv2.circle(img, (rx, ry), 5, (255, 220, 0), 2, cv2.LINE_AA)
            cv2.line(img, (gx, gy), (px, py), _ERR_COLOR, 1, cv2.LINE_AA)
            cv2.putText(img, name, (gx + 6, gy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, _GT_COLOR, 1, cv2.LINE_AA)

        info = (f"[{idx+1}/{n}]  {os.path.basename(frame_paths[idx])}"
                f"  prior={prior_rmse:.1f}px  refined={refined_rmse:.1f}px"
                f"  |  n/p=navigate  q=quit")
        cv2.putText(img, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('planar_calib_gates', img)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), 83, 32):
            idx = min(idx + 1, n - 1)
        elif key in (ord('p'), 81):
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()


def _visualize(frame_data, rotvec_cam, t_cam, K, dist, rect_maps=None):
    n = len(frame_data)
    idx = 0
    cv2.namedWindow('intrinsics_from_flight', cv2.WINDOW_NORMAL)

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
        cv2.imshow('intrinsics_from_flight', img)

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
        description="Calibrate camera intrinsics from flight labels.")
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
    parser.add_argument('--max-rmse', type=float, default=30.0,
                        help="Only use frames with per-frame RMSE below this threshold (default: 30)")
    parser.add_argument('--max-calib-frames', type=int, default=500,
                        help="Max frames fed to cv2.calibrateCamera (uniform stride if exceeded, default: 500)")
    parser.add_argument('--calibrate', choices=['both', 'K', 'dist'], default='both',
                        help="What to refine: 'both' (default), 'K' only (fix distortion), 'dist' only (fix K)")
    parser.add_argument('--start', type=float, default=None)
    parser.add_argument('--end',   type=float, default=None)
    parser.add_argument('--method', choices=['normal', 'planar', 'direct'], default='normal',
                        help="Calibration method: "
                             "'normal' = calibrateCamera, one view per frame; "
                             "'planar' = calibrateCamera, one view per gate (all 4 visible, gate-local z=0); "
                             "'direct' = scipy LM, fixed extrinsics, minimises pipeline RMSE directly")
    parser.add_argument('--visualize', action='store_true',
                        help="Show frames with GT + projected keypoints; skip calibration")
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

    # Synchronized CSV (drone poses + gate marker positions)
    csv_files = glob(os.path.join(flight_dir, '*_cam_ts_sync.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No *_cam_ts_sync.csv in {flight_dir}")
    df = pd.read_csv(csv_files[0])

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
        for col in df.columns if col.startswith('gate') and '_marker1_x' in col
    })
    gate_quats, gate_mean_centers, _ = precompute_gate_orientations(
        df, gate_ids, args.fix_rotation)

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
    frame_data = []   # (img_path, gate_bodies: list[(8,3)], label_gates)

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
        pos_drone = np.array([row['drone_x'], row['drone_y'], row['drone_z']])
        rot_mat   = np.array([row[f'drone_rot[{i}]'] for i in range(9)]).reshape(3, 3).T
        R_drone_inv = Rotation.from_matrix(rot_mat).inv()

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

    # ── Intrinsics calibration ────────────────────────────────────────────────
    R_cam_inv = Rotation.from_rotvec(rotvec_init).inv()

    prior_rmse = _compute_mse(frame_data, rotvec_init, t_cam, K_proj, dist_proj) ** 0.5
    print(f"Prior RMSE (all frames):  {prior_rmse:.2f} px")

    # Collect 3D-2D correspondences for frames passing the RMSE filter
    object_points   = []   # list of (n, 3) float32
    image_points    = []   # list of (n, 2) float32
    gate_frame_info = []   # (planar only) img_path per gate observation
    filtered_paths  = set()  # img_paths of frames that passed the RMSE filter
    n_skipped_rmse  = 0
    n_skipped_pts   = 0

    for img_path, gate_bodies, label_gates in frame_data:
        gate_results = _project_gates(gate_bodies, R_cam_inv, t_cam, K_proj, dist_proj)
        matches = _match_gates(gate_results, label_gates)

        # Per-frame RMSE with prior intrinsics
        frame_sq, frame_n = 0.0, 0
        for _, proj, visible, lg in matches:
            for i, v_gt in enumerate(lg['vis']):
                if v_gt and visible[:4][i]:
                    d = proj[i] - lg['kps'][i]
                    frame_sq += float(d @ d)
                    frame_n += 1
        if frame_n == 0 or (frame_sq / frame_n) ** 0.5 > args.max_rmse:
            n_skipped_rmse += 1
            continue

        filtered_paths.add(img_path)

        if args.method in ('normal', 'direct'):
            # One element per frame: collect all visible corners across all gates
            frame_obj, frame_img = [], []
            for gate_idx, _, visible, lg in matches:
                corners_cam = R_cam_inv.apply(gate_bodies[gate_idx][:4] - t_cam)
                for i, v_gt in enumerate(lg['vis']):
                    if v_gt and visible[:4][i]:
                        frame_obj.append(corners_cam[i])
                        frame_img.append(lg['kps'][i])
            if len(frame_obj) < 6:
                n_skipped_pts += 1
                continue
            object_points.append(np.array(frame_obj, dtype=np.float32))
            image_points.append(np.array(frame_img, dtype=np.float32))
        else:  # planar
            # One element per gate: require all 4 corners visible
            for gate_idx, _, visible, lg in matches:
                if not (all(visible[:4]) and all(lg['vis'])):
                    continue
                # Gate-local object coordinates (z=0 plane, tl/tr/br/bl order)
                gate_obj = np.array([
                    [-half_i,  half_i, 0.0],
                    [ half_i,  half_i, 0.0],
                    [ half_i, -half_i, 0.0],
                    [-half_i, -half_i, 0.0],
                ], dtype=np.float32)
                object_points.append(gate_obj)
                image_points.append(lg['kps'].astype(np.float32))
                gate_frame_info.append(img_path)

    n_kept = len(object_points)
    unit = 'gates' if args.method == 'planar' else 'frames'
    print(f"Frame filter (max_rmse={args.max_rmse}):  "
          f"kept={n_kept} {unit}  skipped_rmse={n_skipped_rmse}  skipped_few_pts={n_skipped_pts}")

    if n_kept < 3:
        print("Too few frames after filtering. Try increasing --max-rmse.")
        return

    # Downsample if needed (applies to all methods)
    if n_kept > args.max_calib_frames:
        stride = int(np.ceil(n_kept / args.max_calib_frames))
        calib_obj   = object_points[::stride]
        calib_img   = image_points[::stride]
        calib_paths = gate_frame_info[::stride]
    else:
        stride = 1
        calib_obj   = object_points
        calib_img   = image_points
        calib_paths = gate_frame_info

    n_calib = len(calib_obj)
    n_pts   = sum(len(p) for p in calib_obj)

    if args.method in ('normal', 'planar'):
        calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS
        if args.calibrate == 'K':
            calib_flags |= (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
                            | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
                            | cv2.CALIB_ZERO_TANGENT_DIST)
        elif args.calibrate == 'dist':
            calib_flags |= cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
        print(f"Running cv2.calibrateCamera on {n_calib} {unit} "
              f"({n_pts} pts, stride={stride}), calibrate={args.calibrate}...")
        _, K_ref, dist_ref, _, _ = cv2.calibrateCamera(
            calib_obj, calib_img, (IMG_WIDTH, IMG_HEIGHT),
            K_proj.copy(), dist_proj.copy(),
            flags=calib_flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
        )

    else:  # direct — scipy LM with fixed extrinsics
        all_obj = np.vstack(calib_obj).reshape(-1, 1, 3).astype(np.float32)
        all_img = np.vstack(calib_img).astype(np.float64)
        rvec0, tvec0 = np.zeros(3), np.zeros(3)

        def _res(params):
            K_c, dist_c = _unpack_K_dist(params, K_proj.copy(), dist_proj.copy(), args.calibrate)
            proj, _ = cv2.projectPoints(all_obj, rvec0, tvec0, K_c, dist_c)
            return (proj.reshape(-1, 2) - all_img).flatten()

        x0 = _pack_x0(K_proj, dist_proj, args.calibrate)
        print(f"Running scipy LM on {n_calib} {unit} "
              f"({n_pts} pts, stride={stride}), calibrate={args.calibrate}...")
        opt = scipy.optimize.least_squares(_res, x0, method='lm', max_nfev=2000)
        K_ref, dist_ref = _unpack_K_dist(opt.x, K_proj.copy(), dist_proj.copy(), args.calibrate)

    # ── Evaluation ────────────────────────────────────────────────────────────
    rotvec_ref = Rotation.from_matrix(R_cam_inv.inv().as_matrix()).as_rotvec()
    n_filtered_frames = len(filtered_paths)
    prior_rmse_filt   = _compute_mse(frame_data, rotvec_init, t_cam, K_proj, dist_proj,
                                     include_paths=filtered_paths) ** 0.5
    refined_rmse_filt = _compute_mse(frame_data, rotvec_ref,  t_cam, K_ref,  dist_ref,
                                     include_paths=filtered_paths) ** 0.5
    refined_rmse_all  = _compute_mse(frame_data, rotvec_ref,  t_cam, K_ref,  dist_ref) ** 0.5

    lbl_filt = f"pipeline RMSE — filtered frames ({n_filtered_frames})"
    print(f"\n{'':42s}  {'prior':>8}  {'refined':>8}")
    print(f"{lbl_filt:42s}  {prior_rmse_filt:8.2f}  {refined_rmse_filt:8.2f}  px")
    print(f"{'pipeline RMSE — all frames':42s}  {prior_rmse:8.2f}  {refined_rmse_all:8.2f}  px")

    if args.method == 'direct':
        # Fixed-matching RMSE: same objective the optimizer minimised — no gate-remapping artifacts
        prior_fixed   = _rmse_fixed_matching(object_points, image_points, K_proj, dist_proj)
        refined_fixed = _rmse_fixed_matching(object_points, image_points, K_ref,  dist_ref)
        lbl_fixed = f"fixed-match RMSE — calib frames ({n_calib})"
        print(f"{lbl_fixed:42s}  {prior_fixed:8.2f}  {refined_fixed:8.2f}  px")

    result = {
        "mtx": K_ref.tolist(),
        "dist": [dist_ref.flatten().tolist()],
    }
    print("\n" + json.dumps(result, indent=4))

    if args.method == 'planar':
        _visualize_planar_calib(calib_obj, calib_img, calib_paths,
                                K_proj, dist_proj, K_ref, dist_ref, rect_maps)


if __name__ == '__main__':
    main()
