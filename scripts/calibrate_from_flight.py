import argparse
import json
import os
import zipfile
from glob import glob
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


def ensure_extracted(folder_path, zip_path):
    if os.path.isdir(folder_path):
        return True
    if not os.path.isfile(zip_path):
        return False

    print(f"Extracting: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(folder_path))

    return os.path.isdir(folder_path)


def get_flight_dir(flight):
    flight_type = "piloted" if "p-" in flight else "autonomous"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "data", flight_type, flight)


def find_nearest_idx(sorted_values, target):
    idx = np.searchsorted(sorted_values, target)
    if idx <= 0:
        return 0
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    if abs(sorted_values[idx] - target) < abs(sorted_values[idx - 1] - target):
        return idx
    return idx - 1


def parse_label_file(label_file):
    detections = []
    with open(label_file) as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue

            keypoints = []
            kp_values = [float(v) for v in values[5:]]
            if len(kp_values) < 12:
                continue

            for i in range(4):
                x = kp_values[i * 3]
                y = kp_values[i * 3 + 1]
                vis = int(float(kp_values[i * 3 + 2]))
                keypoints.append((x, y, vis))

            visible_xy = [(x, y) for x, y, vis in keypoints if vis ==
                          2 and not (x == 0.0 and y == 0.0)]
            if visible_xy:
                center_x = float(np.mean([p[0] for p in visible_xy]))
                center_y = float(np.mean([p[1] for p in visible_xy]))
            else:
                center_x = float(values[1])
                center_y = float(values[2])

            detections.append({
                "keypoints": keypoints,
                "center_norm": (center_x, center_y),
            })
    return detections


def get_default_calibration_path(flight):
    calib_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "camera_calibration"))
    if "trackRATM" in flight:
        if "p-" in flight:
            return os.path.join(calib_dir, "calib_p-trackRATM.json")
        return os.path.join(calib_dir, "calib_a-trackRATM.json")
    return os.path.join(calib_dir, "calib_ap-ellipse-lemniscate.json")


def get_camera_extrinsics(flight):
    extr_path = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "..", "camera_calibration", "drone_to_camera.json"))
    with open(extr_path) as f:
        extr = json.load(f)
    t = extr["translation"]
    trans = np.array([t["x"], t["y"], t["z"]], dtype=np.float64)
    if "trackRATM" in flight:
        rot_key = "trackRATM"
    elif "p-" in flight:
        rot_key = "piloted"
    elif "lemniscate" in flight:
        rot_key = "lemniscate"
    else:
        rot_key = "ellipse"
    r = extr["rotation"][rot_key]
    quat = np.array([r["x"], r["y"], r["z"], r["w"]], dtype=np.float64)
    return trans, quat


def assign_detections_to_gates(det_centers_px, gate_centers_px, max_center_dist_px):
    if not det_centers_px or not gate_centers_px:
        return {}
    gate_ids = list(gate_centers_px.keys())
    n_det = len(det_centers_px)
    n_gate = len(gate_ids)
    cost = np.full((n_det, n_gate), max_center_dist_px * 2.0)
    for i, det in enumerate(det_centers_px):
        for j, gid in enumerate(gate_ids):
            d = float(np.hypot(det[0] - gate_centers_px[gid][0], det[1] - gate_centers_px[gid][1]))
            if d <= max_center_dist_px:
                cost[i, j] = d
    row_ind, col_ind = linear_sum_assignment(cost)
    return {
        int(row_ind[k]): gate_ids[col_ind[k]]
        for k in range(len(row_ind))
        if cost[row_ind[k], col_ind[k]] <= max_center_dist_px
    }


def infer_image_size(image_dir):
    images = sorted(glob(os.path.join(image_dir, "*.jpg")))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    img = cv2.imread(images[0])
    if img is None:
        raise RuntimeError(f"Cannot read image {images[0]}")

    height, width = img.shape[:2]
    return width, height


def build_correspondences(
    gate_df,
    gate_ids,
    label_files,
    width,
    height,
    max_time_delta_us,
    pose_df,
    init_mtx,
    init_dist,
    cam_trans,
    cam_quat,
    max_center_dist_px,
):
    timestamps = gate_df["timestamp"].astype(np.int64).values
    pose_timestamps = pose_df["timestamp"].astype(np.int64).values
    R_cam_inv = Rotation.from_quat(cam_quat).inv()

    object_points = []
    image_points = []
    frame_gate_ids = []

    frame_stats = {
        "total_label_files": len(label_files),
        "used_frames": 0,
        "skipped_no_points": 0,
        "skipped_time_delta": 0,
        "skipped_unmatched": 0,
    }

    progress_every = max(1, len(label_files) // 20)

    for index, label_file in enumerate(label_files, start=1):
        timestamp = int(os.path.basename(label_file).split("_")[-1].split(".")[0])
        idx = find_nearest_idx(timestamps, timestamp)
        matched_ts = int(timestamps[idx])
        if abs(matched_ts - timestamp) > max_time_delta_us:
            frame_stats["skipped_time_delta"] += 1
            continue

        row = gate_df.iloc[idx]
        detections = parse_label_file(label_file)

        if not detections:
            frame_stats["skipped_no_points"] += 1
            continue

        pose_idx = find_nearest_idx(pose_timestamps, timestamp)
        pose_row = pose_df.iloc[pose_idx]
        pos_drone = np.array([pose_row["drone_x"], pose_row["drone_y"],
                             pose_row["drone_z"]], dtype=np.float64)
        drone_rot = np.array([
            pose_row["drone_rot[0]"], pose_row["drone_rot[1]"], pose_row["drone_rot[2]"],
            pose_row["drone_rot[3]"], pose_row["drone_rot[4]"], pose_row["drone_rot[5]"],
            pose_row["drone_rot[6]"], pose_row["drone_rot[7]"], pose_row["drone_rot[8]"],
        ], dtype=np.float64).reshape(3, 3)
        R_drone_inv = Rotation.from_matrix(drone_rot).inv()

        gate_centers_px = {}
        gate_corners_cam = {}
        gate_corners_px = {}
        for gate_id in gate_ids:
            corners_earth = np.array([
                [
                    row[f"gate{gate_id}_marker{m}_x"],
                    row[f"gate{gate_id}_marker{m}_y"],
                    row[f"gate{gate_id}_marker{m}_z"],
                ]
                for m in range(1, 5)
            ], dtype=np.float64)
            if np.isnan(corners_earth).any():
                continue
            corners_body = R_drone_inv.apply(corners_earth - pos_drone)
            corners_cam = R_cam_inv.apply(corners_body - cam_trans)
            if np.any(corners_cam[:, 2] <= 0):
                continue
            projected, _ = cv2.projectPoints(
                corners_cam, np.zeros(3), np.zeros(3), init_mtx, init_dist)
            projected = projected.reshape(-1, 2)
            center_px = projected.mean(axis=0)
            gate_centers_px[gate_id] = (float(center_px[0]), float(center_px[1]))
            gate_corners_cam[gate_id] = corners_cam   # (4, 3) camera-frame 3D positions
            gate_corners_px[gate_id] = projected       # (4, 2) projected pixel positions

        det_centers_px = [
            (det["center_norm"][0] * width, det["center_norm"][1] * height)
            for det in detections
        ]
        det_to_gate = assign_detections_to_gates(
            det_centers_px, gate_centers_px, max_center_dist_px)
        if not det_to_gate:
            frame_stats["skipped_unmatched"] += 1
            continue

        frame_obj = []
        frame_img = []
        frame_gates = []

        for det_idx, det in enumerate(detections):
            gate_id = det_to_gate.get(det_idx)
            if gate_id is None or gate_id not in gate_corners_cam:
                continue

            corners_cam = gate_corners_cam[gate_id]   # (4, 3)
            proj_markers = gate_corners_px[gate_id]    # (4, 2)

            keypoints = det["keypoints"]
            vis_kps = [
                (kp_idx, np.array([x * width, y * height]))
                for kp_idx, (x, y, vis) in enumerate(keypoints)
                if vis == 2 and not (x == 0.0 and y == 0.0)
            ]
            if not vis_kps:
                continue

            kp_pxs = [kp for _, kp in vis_kps]
            kp_pxs_arr = np.array(kp_pxs)  # (n_vis, 2)

            # Hungarian matching: each visible keypoint → nearest projected marker
            cost = np.linalg.norm(kp_pxs_arr[:, None, :] - proj_markers[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)

            # Reject gate if any matched pair is too far (> half gate bounding-box diagonal)
            gate_diag = float(np.linalg.norm(proj_markers.max(axis=0) - proj_markers.min(axis=0)))
            if gate_diag > 0 and cost[row_ind, col_ind].max() > 0.5 * gate_diag:
                continue

            for r, c in zip(row_ind, col_ind):
                frame_obj.append(corners_cam[c].tolist())    # camera-frame 3D point
                frame_img.append(kp_pxs[r].tolist())
                frame_gates.append(gate_id)

        if len(frame_obj) < 6:
            frame_stats["skipped_no_points"] += 1
            continue

        object_points.append(np.array(frame_obj, dtype=np.float32))
        image_points.append(np.array(frame_img, dtype=np.float32))
        frame_gate_ids.append(np.array(frame_gates, dtype=np.int32))
        frame_stats["used_frames"] += 1

        if index % progress_every == 0 or index == len(label_files):
            print(
                f"  [build] {index}/{len(label_files)} labels | "
                f"used={frame_stats['used_frames']} "
                f"skip_time={frame_stats['skipped_time_delta']} "
                f"skip_points={frame_stats['skipped_no_points']} "
                f"skip_unmatched={frame_stats['skipped_unmatched']}"
            )

    return object_points, image_points, frame_gate_ids, frame_stats


def downsample_frames(object_points, image_points, frame_gate_ids, max_frames):
    if max_frames <= 0 or len(object_points) <= max_frames:
        return object_points, image_points, frame_gate_ids, 1

    stride = int(np.ceil(len(object_points) / max_frames))
    return (
        object_points[::stride],
        image_points[::stride],
        frame_gate_ids[::stride],
        stride,
    )


def compute_diagnostics(object_points, image_points, frame_gate_ids, rvecs, tvecs, mtx, dist):
    all_errors = []
    frame_mean_errors = []
    gate_errors = {}

    for i, (obj, img) in enumerate(zip(object_points, image_points)):
        projected, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], mtx, dist)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - img, axis=1)

        all_errors.extend(errors.tolist())
        frame_mean_errors.append(float(np.mean(errors)))

        for error, gate_id in zip(errors, frame_gate_ids[i]):
            gate_errors.setdefault(int(gate_id), []).append(float(error))

    all_errors = np.array(all_errors, dtype=np.float64)
    mse = float(np.mean(np.square(all_errors)))

    per_gate = {
        f"gate{gate_id}": {
            "mean_error_px": float(np.mean(vals)),
            "median_error_px": float(np.median(vals)),
            "num_points": int(len(vals)),
        }
        for gate_id, vals in sorted(gate_errors.items())
    }

    return {
        "rmse_px": float(np.sqrt(mse)),
        "mean_error_px": float(np.mean(all_errors)),
        "median_error_px": float(np.median(all_errors)),
        "max_error_px": float(np.max(all_errors)),
        "num_points": int(len(all_errors)),
        "num_frames": int(len(object_points)),
        "frame_mean_error_px": {
            "mean": float(np.mean(frame_mean_errors)),
            "median": float(np.median(frame_mean_errors)),
            "max": float(np.max(frame_mean_errors)),
        },
        "per_gate": per_gate,
    }


def load_initial_calibration(calib_path):
    with open(calib_path) as f:
        data = json.load(f)

    if "mtx" not in data or "dist" not in data:
        raise ValueError(f"Initial calibration file must contain 'mtx' and 'dist': {calib_path}")

    mtx = np.array(data["mtx"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64)

    if mtx.shape != (3, 3):
        raise ValueError(f"Initial 'mtx' must have shape (3, 3), got {mtx.shape} in {calib_path}")

    dist = dist.reshape(-1)
    if dist.size < 5:
        raise ValueError(
            f"Initial 'dist' must contain at least 5 coefficients, got {dist.size} in {calib_path}")

    return mtx, dist


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from a flight using 3D gate corners in csv_raw and 2D keypoints in labels."
    )
    parser.add_argument("--flight", required=True, help="Flight ID (e.g., flight-01p-ellipse)")
    parser.add_argument("--output", required=True,
                        help="Output JSON path for calibration parameters")
    parser.add_argument(
        "--camera-model",
        default="plumb_bob",
        help="Camera model. Only 'plumb_bob' is currently supported.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting the output file if it already exists.",
    )
    parser.add_argument(
        "--max-time-delta-us",
        type=int,
        default=5000,
        help="Maximum allowed timestamp mismatch (us) between label frame and nearest csv_raw sample.",
    )
    parser.add_argument(
        "--diagnostics-output",
        default="",
        help="Optional path to write calibration diagnostics JSON.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2500,
        help="Maximum number of frames used for calibration (automatic even downsampling if exceeded).",
    )
    parser.add_argument(
        "--max-calib-frames",
        type=int,
        default=800,
        help="Maximum frames used by the optimizer itself (second downsampling stage for speed).",
    )
    parser.add_argument(
        "--initial-calib",
        default="",
        help="Optional path to an initial calibration JSON (with mtx/dist) used as optimization seed.",
    )
    parser.add_argument(
        "--optimize-mode",
        choices=["full", "fast"],
        default="full",
        help="Calibration optimization mode: full (all standard params) or fast (fewer free params).",
    )
    parser.add_argument(
        "--max-calib-iters",
        type=int,
        default=25,
        help="Maximum optimizer iterations for cv2.calibrateCamera.",
    )
    parser.add_argument(
        "--calib-eps",
        type=float,
        default=1e-6,
        help="Optimizer epsilon threshold for cv2.calibrateCamera termination.",
    )
    parser.add_argument(
        "--max-center-dist-px",
        type=float,
        default=220.0,
        help="Max pixel distance for matching detections to projected gate centers.",
    )
    args = parser.parse_args()

    if args.camera_model != "plumb_bob":
        parser.error("Only --camera-model plumb_bob is supported for now.")

    initial_calib_path = ""
    if args.initial_calib:
        initial_calib_path = os.path.abspath(args.initial_calib)
        if not os.path.isfile(initial_calib_path):
            raise FileNotFoundError(f"Initial calibration file not found: {initial_calib_path}")

    if initial_calib_path:
        match_mtx, match_dist = load_initial_calibration(initial_calib_path)
    else:
        default_calib_path = get_default_calibration_path(args.flight)
        if not os.path.isfile(default_calib_path):
            raise FileNotFoundError(
                "No initial calibration provided and default calibration file not found: "
                f"{default_calib_path}. Provide --initial-calib."
            )
        match_mtx, match_dist = load_initial_calibration(default_calib_path)

    output_path = os.path.abspath(args.output)
    if os.path.exists(output_path) and not args.force:
        parser.error(f"Output file already exists: {output_path}. Use --force to overwrite.")

    flight_dir = get_flight_dir(args.flight)
    if not os.path.isdir(flight_dir):
        raise FileNotFoundError(f"Flight folder not found: {flight_dir}")

    print(f"[1/6] Preparing calibration for flight: {args.flight}")

    gate_csv = os.path.join(flight_dir, "csv_raw", f"gate_corners_{args.flight}.csv")
    mocap_csv = os.path.join(flight_dir, "csv_raw", f"mocap_{args.flight}.csv")
    label_dir = os.path.join(flight_dir, f"label_{args.flight}")
    image_dir = os.path.join(flight_dir, f"camera_{args.flight}")

    ensure_extracted(label_dir, label_dir + ".zip")
    ensure_extracted(image_dir, image_dir + ".zip")

    if not os.path.isfile(gate_csv):
        raise FileNotFoundError(f"Missing file: {gate_csv}")
    if not os.path.isfile(mocap_csv):
        raise FileNotFoundError(f"Missing file: {mocap_csv}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Missing folder: {label_dir}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing folder: {image_dir}")

    gate_df = pd.read_csv(gate_csv)
    pose_df = pd.read_csv(mocap_csv)
    cam_trans, cam_quat = get_camera_extrinsics(args.flight)
    gate_ids = sorted({
        int(col.split('_')[0][4:])
        for col in gate_df.columns
        if col.startswith('gate') and '_marker1_x' in col
    })
    if not gate_ids:
        raise RuntimeError(f"No gate markers found in {gate_csv}")

    print(f"  gate CSV rows: {len(gate_df)} | detected gates: {gate_ids}")

    width, height = infer_image_size(image_dir)
    all_label_files = sorted(glob(os.path.join(label_dir, "*.txt")))
    if not all_label_files:
        raise RuntimeError(f"No label files found in {label_dir}")
    downsample_stride = max(1, len(all_label_files) // args.max_frames)
    label_files = all_label_files[::downsample_stride]

    print(
        f"  image size: {width}x{height} | label files: {len(all_label_files)} "
        f"| sampling stride: {downsample_stride} → {len(label_files)} to process"
    )
    print(f"[2/6] Building 3D-2D correspondences (max_time_delta_us={args.max_time_delta_us})")

    object_points, image_points, frame_gate_ids, frame_stats = build_correspondences(
        gate_df,
        gate_ids,
        label_files,
        width,
        height,
        args.max_time_delta_us,
        pose_df,
        match_mtx,
        match_dist,
        cam_trans,
        cam_quat,
        args.max_center_dist_px,
    )

    opt_object_points, opt_image_points, opt_frame_gate_ids, calib_stride = downsample_frames(
        object_points,
        image_points,
        frame_gate_ids,
        args.max_calib_frames,
    )

    print(
        f"[3/6] Correspondences ready: frames={len(object_points)} (pre-sampled stride={downsample_stride})")
    print(
        f"  optimizer frames: {len(opt_object_points)} (optimizer stride={calib_stride}, max_calib_frames={args.max_calib_frames})")

    if len(opt_object_points) < 3:
        raise RuntimeError(
            "Too few frames with valid correspondences for calibration. "
            "Try increasing --max-time-delta-us or check labels/csv consistency."
        )

    if initial_calib_path:
        init_mtx, init_dist = load_initial_calibration(initial_calib_path)
        init_dist = init_dist.astype(np.float64)
        print(f"  using initial calibration seed: {initial_calib_path}")
    else:
        # Use the matching calibration as optimizer seed; initCameraMatrix2D is designed
        # for flat patterns and gives poor estimates for camera-frame 3D points.
        init_mtx = match_mtx.copy()
        init_dist = match_dist.copy().astype(np.float64)
        print("  using matching calibration as optimizer seed")

    calib_flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if args.optimize_mode == "fast":
        calib_flags |= (
            cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_FIX_PRINCIPAL_POINT
        )

    calib_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        args.max_calib_iters,
        args.calib_eps,
    )

    print(
        f"  optimize_mode={args.optimize_mode} | max_calib_iters={args.max_calib_iters} | calib_eps={args.calib_eps}")

    print("[4/6] Running OpenCV calibration...")
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        opt_object_points,
        opt_image_points,
        (width, height),
        init_mtx,
        init_dist,
        flags=calib_flags,
        criteria=calib_criteria,
    )

    print("[4b/6] Running outlier rejection...")
    frames_removed_total = 0
    for _iter in range(3):
        per_frame_err = np.array([
            float(np.mean(np.linalg.norm(
                cv2.projectPoints(obj, rv, tv, mtx, dist)[0].reshape(-1, 2) - img,
                axis=1,
            )))
            for obj, img, rv, tv in zip(opt_object_points, opt_image_points, rvecs, tvecs)
        ])
        q25, q75 = np.percentile(per_frame_err, [25, 75])
        threshold = float(np.median(per_frame_err) + 2.0 * (q75 - q25))
        mask = per_frame_err < threshold
        n_removed = int((~mask).sum())
        if n_removed == 0:
            break
        frames_removed_total += n_removed
        print(
            f"  iter {_iter + 1}: removing {n_removed} frames "
            f"(threshold={threshold:.2f}px, median={float(np.median(per_frame_err)):.2f}px)"
        )
        opt_object_points = [p for p, m in zip(opt_object_points, mask) if m]
        opt_image_points = [p for p, m in zip(opt_image_points, mask) if m]
        opt_frame_gate_ids = [p for p, m in zip(opt_frame_gate_ids, mask) if m]
        if len(opt_object_points) < 3:
            print("  too few frames after outlier removal, stopping")
            break
        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            opt_object_points, opt_image_points, (width, height),
            mtx.copy(), dist.copy(), flags=calib_flags, criteria=calib_criteria,
        )
    print(f"  outlier rejection complete: removed {frames_removed_total} frames total, {len(opt_object_points)} remaining")

    output = {
        "mtx": np.array(mtx).tolist(),
        "dist": np.array(dist).tolist(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("[5/6] Writing calibration JSON...")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    diagnostics = {
        "flight": args.flight,
        "camera_model": args.camera_model,
        "image_width": width,
        "image_height": height,
        "frame_stats": frame_stats,
        "sampling": {
            "stride": int(downsample_stride),
            "max_frames": int(args.max_frames),
            "frames_after_sampling": int(len(object_points)),
        },
        "optimization": {
            "mode": args.optimize_mode,
            "max_calib_frames": int(args.max_calib_frames),
            "stride": int(calib_stride),
            "frames_before_outlier_rejection": int(len(opt_object_points) + frames_removed_total),
            "frames_removed_by_outlier_rejection": int(frames_removed_total),
            "used_frames": int(len(opt_object_points)),
            "max_calib_iters": int(args.max_calib_iters),
            "calib_eps": float(args.calib_eps),
        },
        "calibration": compute_diagnostics(opt_object_points, opt_image_points, opt_frame_gate_ids, rvecs, tvecs, mtx, dist),
    }

    print("[6/6] Calibration diagnostics computed")

    print("Calibration complete")
    print(f"  output: {output_path}")
    print(f"  used_frames: {frame_stats['used_frames']} / {frame_stats['total_label_files']}")
    if downsample_stride > 1:
        print(f"  sampling_stride: {downsample_stride}")
    print(f"  used_points: {diagnostics['calibration']['num_points']}")
    print(f"  rmse_px: {diagnostics['calibration']['rmse_px']:.4f}")
    print(f"  mean_error_px: {diagnostics['calibration']['mean_error_px']:.4f}")
    print(f"  median_error_px: {diagnostics['calibration']['median_error_px']:.4f}")

    if args.diagnostics_output:
        diagnostics_path = os.path.abspath(args.diagnostics_output)
        os.makedirs(os.path.dirname(diagnostics_path), exist_ok=True)
        with open(diagnostics_path, "w") as f:
            json.dump(diagnostics, f, indent=4)
        print(f"  diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
