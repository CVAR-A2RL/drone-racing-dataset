import argparse
import json
import os
import zipfile
from glob import glob

import cv2
import numpy as np
import pandas as pd


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

            detections.append(keypoints)
    return detections


def infer_image_size(image_dir):
    images = sorted(glob(os.path.join(image_dir, "*.jpg")))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    img = cv2.imread(images[0])
    if img is None:
        raise RuntimeError(f"Cannot read image {images[0]}")

    height, width = img.shape[:2]
    return width, height


def build_correspondences(gate_df, gate_ids, label_files, width, height, max_time_delta_us):
    timestamps = gate_df["timestamp"].astype(np.int64).values

    object_points = []
    image_points = []
    frame_gate_ids = []

    frame_stats = {
        "total_label_files": len(label_files),
        "used_frames": 0,
        "skipped_no_points": 0,
        "skipped_time_delta": 0,
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

        num_pairs = min(len(detections), len(gate_ids))
        frame_obj = []
        frame_img = []
        frame_gates = []

        for det_idx in range(num_pairs):
            gate_id = gate_ids[det_idx]
            keypoints = detections[det_idx]

            for kp_idx, (x_norm, y_norm, vis) in enumerate(keypoints):
                if vis != 2:
                    continue
                if x_norm == 0.0 and y_norm == 0.0:
                    continue

                marker = kp_idx + 1
                gx = row[f"gate{gate_id}_marker{marker}_x"]
                gy = row[f"gate{gate_id}_marker{marker}_y"]
                gz = row[f"gate{gate_id}_marker{marker}_z"]

                if np.isnan(gx) or np.isnan(gy) or np.isnan(gz):
                    continue

                frame_obj.append([float(gx), float(gy), float(gz)])
                frame_img.append([float(x_norm * width), float(y_norm * height)])
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
                f"skip_points={frame_stats['skipped_no_points']}"
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


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from a flight using 3D gate corners in csv_raw and 2D keypoints in labels."
    )
    parser.add_argument("--flight", required=True, help="Flight ID (e.g., flight-01p-ellipse)")
    parser.add_argument("--output", required=True, help="Output JSON path for calibration parameters")
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
    args = parser.parse_args()

    if args.camera_model != "plumb_bob":
        parser.error("Only --camera-model plumb_bob is supported for now.")

    output_path = os.path.abspath(args.output)
    if os.path.exists(output_path) and not args.force:
        parser.error(f"Output file already exists: {output_path}. Use --force to overwrite.")

    flight_dir = get_flight_dir(args.flight)
    if not os.path.isdir(flight_dir):
        raise FileNotFoundError(f"Flight folder not found: {flight_dir}")

    print(f"[1/6] Preparing calibration for flight: {args.flight}")

    gate_csv = os.path.join(flight_dir, "csv_raw", f"gate_corners_{args.flight}.csv")
    label_dir = os.path.join(flight_dir, f"label_{args.flight}")
    image_dir = os.path.join(flight_dir, f"camera_{args.flight}")

    ensure_extracted(label_dir, label_dir + ".zip")
    ensure_extracted(image_dir, image_dir + ".zip")

    if not os.path.isfile(gate_csv):
        raise FileNotFoundError(f"Missing file: {gate_csv}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Missing folder: {label_dir}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing folder: {image_dir}")

    gate_df = pd.read_csv(gate_csv)
    gate_ids = sorted({
        int(col.split('_')[0][4:])
        for col in gate_df.columns
        if col.startswith('gate') and '_marker1_x' in col
    })
    if not gate_ids:
        raise RuntimeError(f"No gate markers found in {gate_csv}")

    print(f"  gate CSV rows: {len(gate_df)} | detected gates: {gate_ids}")

    width, height = infer_image_size(image_dir)
    label_files = sorted(glob(os.path.join(label_dir, "*.txt")))
    if not label_files:
        raise RuntimeError(f"No label files found in {label_dir}")

    print(f"  image size: {width}x{height} | label files: {len(label_files)}")
    print(f"[2/6] Building 3D-2D correspondences (max_time_delta_us={args.max_time_delta_us})")

    object_points, image_points, frame_gate_ids, frame_stats = build_correspondences(
        gate_df,
        gate_ids,
        label_files,
        width,
        height,
        args.max_time_delta_us,
    )

    object_points, image_points, frame_gate_ids, downsample_stride = downsample_frames(
        object_points,
        image_points,
        frame_gate_ids,
        args.max_frames,
    )

    print(f"[3/6] Correspondences ready: frames={len(object_points)} after downsample stride={downsample_stride}")

    if len(object_points) < 3:
        raise RuntimeError(
            "Too few frames with valid correspondences for calibration. "
            "Try increasing --max-time-delta-us or check labels/csv consistency."
        )

    init_mtx = cv2.initCameraMatrix2D(object_points, image_points, (width, height), 0)
    print("[4/6] Running OpenCV calibration...")
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        (width, height),
        init_mtx,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )

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
        "downsample": {
            "stride": int(downsample_stride),
            "max_frames": int(args.max_frames),
            "used_frames_after_downsample": int(len(object_points)),
        },
        "calibration": compute_diagnostics(object_points, image_points, frame_gate_ids, rvecs, tvecs, mtx, dist),
    }

    print("[6/6] Calibration diagnostics computed")

    print("Calibration complete")
    print(f"  output: {output_path}")
    print(f"  used_frames: {frame_stats['used_frames']} / {frame_stats['total_label_files']}")
    if downsample_stride > 1:
        print(f"  downsample_stride: {downsample_stride}")
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
