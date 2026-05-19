import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


class TRTYoloPose:
    """Pure TensorRT YOLO-pose inference — no ultralytics required.

    Preprocessing matches the C++ pipeline:
      letterbox → 640×640 (pad=114), BGR→RGB, HWC→CHW, /255, float32.
    Postprocessing decodes [anchors, 5+num_kp*3], removes letterbox offset,
    applies cv2 NMS, and returns keypoints in original image coordinates.
    """
    INPUT_SIZE = 640
    PAD_VAL = 114

    def __init__(self, engine_path, conf=0.1, nms_iou=0.45):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initialises CUDA context
        self._cuda = cuda
        self.conf = conf
        self.nms_iou = nms_iou

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Support both TRT 9 (tensor API) and TRT 8 (binding API)
        self._tensor_api = hasattr(self.engine, 'num_io_tensors')
        if self._tensor_api:
            names = [self.engine.get_tensor_name(i)
                     for i in range(self.engine.num_io_tensors)]
            self._in_name = next(n for n in names
                                 if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
            self._out_name = next(n for n in names
                                  if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
            in_shape = tuple(abs(d) for d in self.engine.get_tensor_shape(self._in_name))
            out_shape = tuple(abs(d) for d in self.engine.get_tensor_shape(self._out_name))
        else:
            self._in_name = self.engine.get_binding_name(0)
            self._out_name = self.engine.get_binding_name(1)
            in_shape = tuple(abs(d) for d in self.engine.get_binding_shape(0))
            out_shape = tuple(abs(d) for d in self.engine.get_binding_shape(1))

        print(f"  TRT input:  {self._in_name}  {in_shape}")
        print(f"  TRT output: {self._out_name} {out_shape}")
        self._out_shape = out_shape

        in_n = int(np.prod(in_shape))
        out_n = int(np.prod(out_shape))
        self._in_host = cuda.pagelocked_empty(in_n, np.float32)
        self._out_host = cuda.pagelocked_empty(out_n, np.float32)
        self._in_dev = cuda.mem_alloc(in_n * 4)
        self._out_dev = cuda.mem_alloc(out_n * 4)

    # ------------------------------------------------------------------
    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.INPUT_SIZE - nw) // 2
        pad_y = (self.INPUT_SIZE - nh) // 2
        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), self.PAD_VAL, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        return chw[np.newaxis], scale, pad_x, pad_y

    def _decode(self, flat, scale, pad_x, pad_y):
        out = flat.reshape(self._out_shape)
        if out.ndim == 3:
            out = out[0]                    # drop batch dim → [f, a] or [a, f]
        if out.shape[0] < out.shape[1]:     # transposed [features, anchors]
            out = out.T                     # → [anchors, features]
        num_kp = (out.shape[1] - 5) // 3

        scores = out[:, 4]
        mask = scores >= self.conf
        if not np.any(mask):
            return []
        det = out[mask]
        scores = scores[mask]

        cx = (det[:, 0] - pad_x) / scale
        cy = (det[:, 1] - pad_y) / scale
        bw = det[:, 2] / scale
        bh = det[:, 3] / scale
        boxes = [[float(x), float(y), float(w), float(h)]
                 for x, y, w, h in zip((cx - bw / 2).tolist(),
                                       (cy - bh / 2).tolist(),
                                       bw.tolist(), bh.tolist())]
        idxs = cv2.dnn.NMSBoxes(boxes, scores.tolist(), self.conf, self.nms_iou)
        if len(idxs) == 0:
            return []

        results = []
        for i in np.array(idxs).flatten():
            kps = []
            for k in range(num_kp):
                kx = (det[i, 5 + k * 3] - pad_x) / scale
                ky = (det[i, 5 + k * 3 + 1] - pad_y) / scale
                kc = float(det[i, 5 + k * 3 + 2])
                kps.append((float(kx), float(ky), kc))
            results.append(kps)
        return results

    def __call__(self, img_bgr):
        inp, scale, pad_x, pad_y = self._preprocess(img_bgr)
        np.copyto(self._in_host, inp.ravel())
        self._cuda.memcpy_htod_async(self._in_dev, self._in_host, self.stream)
        if self._tensor_api:
            self.context.set_tensor_address(self._in_name, int(self._in_dev))
            self.context.set_tensor_address(self._out_name, int(self._out_dev))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(
                bindings=[int(self._in_dev), int(self._out_dev)],
                stream_handle=self.stream.handle)
        self._cuda.memcpy_dtoh_async(self._out_host, self._out_dev, self.stream)
        self.stream.synchronize()
        return self._decode(self._out_host.copy(), scale, pad_x, pad_y)


class UltralyticsYoloPose:
    """Ultralytics YOLO pose wrapper with the same interface as TRTYoloPose.
    Returns list of detections, each a list of (kx, ky, kc) in input image coords.
    """
    def __init__(self, model_path, conf=0.75):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf

    def __call__(self, img_bgr):
        results = self.model(img_bgr, verbose=False, conf=self.conf)
        detections = []
        for r in results:
            if r.keypoints is None:
                continue
            xy = r.keypoints.xy.cpu().numpy()        # (N_det, N_kp, 2)
            conf = r.keypoints.conf                   # (N_det, N_kp) or None
            if conf is not None:
                conf = conf.cpu().numpy()
            for di in range(xy.shape[0]):
                kps = []
                for ki in range(xy.shape[1]):
                    kx, ky = float(xy[di, ki, 0]), float(xy[di, ki, 1])
                    kc = float(conf[di, ki]) if conf is not None else 1.0
                    kps.append((kx, ky, kc))
                detections.append(kps)
        return detections


_KEYPOINT_NAMES = ["top_left_inner", "top_right_inner", "bottom_right_inner", "bottom_left_inner"]
_OUTER_KEYPOINT_NAMES = ["top_left_outer", "top_right_outer", "bottom_right_outer", "bottom_left_outer"]

# Model keypoint label order (from tensorrt_backend.cpp keypoint_names_)
# 4-kp models: label file order (tli, tri, bri, bli)
# 8-kp models: C++ order (bro, blo, tro, tlo, bri, bli, tri, tli)
_MODEL_KP_LABELS_4 = ["tli", "tri", "bri", "bli"]
_MODEL_KP_LABELS_8 = ["bro", "blo", "tro", "tlo", "bri", "bli", "tri", "tli"]

_ROT_COLS = [f'drone_rot[{i}]' for i in range(9)]


def get_calibration_path(flight_name):
    calib_dir = os.path.join("..", "camera_calibration")
    if 'trackRATM' in flight_name:
        if 'p-' in flight_name:
            return os.path.join(calib_dir, 'calib_p-trackRATM.json')
        else:
            return os.path.join(calib_dir, 'calib_a-trackRATM.json')
    else:
        return os.path.join(calib_dir, 'calib_ap-ellipse-lemniscate.json')


def get_camera_extrinsics(flight_name):
    extr_path = os.path.join("..", "camera_calibration", "drone_to_camera.json")
    with open(extr_path) as f:
        extr = json.load(f)
    t = extr["translation"]
    trans = [t["x"], t["y"], t["z"]]
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
    quat = [r["x"], r["y"], r["z"], r["w"]]
    return trans, quat


def precompute_gate_orientations(df, gate_ids, fix_rotation):
    gate_rots, gate_quats = {}, {}
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
        if fix_rotation:
            z_axis = np.array([normal[0], normal[1], 0.0])
            z_axis /= np.linalg.norm(z_axis)
            y_axis = np.array([0.0, 0.0, 1.0])
            x_axis = np.cross(y_axis, z_axis)
            fixed_rot = np.column_stack([z_axis, x_axis, y_axis])
            gate_rots[gate_id] = Rotation.from_matrix(fixed_rot).inv()
            gate_quats[gate_id] = Rotation.from_matrix(fixed_rot).as_quat()
        else:
            y_axis_orig = np.cross(normal, x_axis_3d)
            z_axis_orig = np.cross(x_axis_3d, y_axis_orig)
            orig_rot = np.column_stack([z_axis_orig, x_axis_3d, y_axis_orig])
            gate_rots[gate_id] = Rotation.from_matrix(orig_rot).inv()
            gate_quats[gate_id] = Rotation.from_matrix(orig_rot).as_quat()
    return gate_quats, gate_mean_centers, gate_mean_corners


_OUTER_OPP = {0: 2, 1: 3, 2: 0, 3: 1}  # opposite corner index in the outer quad


def _apply_occlusion(candidates):
    for i in range(1, len(candidates)):
        projected_i, visible_i = candidates[i][0], candidates[i][1]
        for j in range(i):
            projected_j, visible_j = candidates[j][0], candidates[j][1]

            outer_vis = visible_j[4:8]
            n_missing = int(np.count_nonzero(~outer_vis))

            if n_missing >= 2:
                continue  # too unreliable, skip

            outer = projected_j[4:8].copy().astype(np.float32)
            if n_missing == 1:
                # Reconstruct missing corner via parallelogram: missing = adj0 + adj1 - opposite
                m = int(np.where(~outer_vis)[0][0])
                opp = _OUTER_OPP[m]
                adj = [k for k in range(4) if k != m and k != opp]
                outer[m] = outer[adj[0]] + outer[adj[1]] - outer[opp]

            inner = projected_j[0:4].astype(np.float32)
            for k in range(len(visible_i)):
                if not visible_i[k]:
                    continue
                pt = (float(projected_i[k, 0]), float(projected_i[k, 1]))
                if (cv2.pointPolygonTest(outer, pt, False) >= 0 and
                        cv2.pointPolygonTest(inner, pt, False) < 0):
                    visible_i[k] = False


def _find_nearest_idx(timestamps_arr, target):
    idx = np.searchsorted(timestamps_arr, target)
    if idx == 0:
        return 0
    if idx >= len(timestamps_arr):
        return len(timestamps_arr) - 1
    if abs(timestamps_arr[idx] - target) < abs(timestamps_arr[idx - 1] - target):
        return idx
    return idx - 1


def _compute_r_crit(dist):
    """Smallest r>0 where f(r)=r*(1+k1*r^2+k2*r^4+k3*r^6) folds back (f'(r)=0).
    Uses the full polynomial so that k2/k3 terms are accounted for.
    Returns inf when the polynomial is monotonically increasing (no wraparound)."""
    k1 = float(dist[0])
    if k1 >= 0:
        return np.inf
    k2 = float(dist[1]) if len(dist) > 1 else 0.0
    k3 = float(dist[4]) if len(dist) > 4 else 0.0
    # f'(r)=0 → substitute u=r²: 7k3·u³ + 5k2·u² + 3k1·u + 1 = 0
    poly = [7*k3, 5*k2, 3*k1, 1.0]
    if abs(poly[0]) < 1e-12:
        poly = poly[1:]
    roots = np.roots(poly)
    pos_real = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    if not pos_real:
        return np.inf
    return float(np.sqrt(min(pos_real)))


def draw_x(img, pt, color, size=8, thickness=2):
    fx, fy = float(pt[0]), float(pt[1])
    if not (np.isfinite(fx) and np.isfinite(fy)):
        return
    x, y = int(round(fx)), int(round(fy))
    h, w = img.shape[:2]
    if x < -2000 or x > w + 2000 or y < -2000 or y > h + 2000:
        return
    cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(img, (x + size, y - size), (x - size, y + size), color, thickness)


def project_gates(pose_row, gate_ids, gate_quats, gate_mean_centers, gate_mean_corners,
                  computed_corners, inner_corners_local, outer_corners_local,
                  K, dist, R_cam_inv, t_cam,
                  img_width, img_height,
                  min_visible_corners, fisheye, gate_depth):
    """Return list of (projected (N,2), visible_final (N,), in_front (N,), kp_names) per gate."""
    pos_drone = np.array([pose_row['drone_x'], pose_row['drone_y'], pose_row['drone_z']])
    # drone_rot columns store R.T (= R_inv for orthogonal matrices), so reshape directly gives R_drone_inv
    rot_mat = pose_row[_ROT_COLS].values.astype(np.float64).reshape(3, 3)
    R_drone_inv = Rotation.from_matrix(rot_mat)

    kp_names = _KEYPOINT_NAMES + _OUTER_KEYPOINT_NAMES if computed_corners else _KEYPOINT_NAMES

    candidates = []
    for gate_id in gate_ids:
        center = gate_mean_centers[gate_id]
        R_gate = Rotation.from_quat(gate_quats[gate_id])
        gate_normal = R_gate.apply([1.0, 0.0, 0.0])
        dot = np.dot(gate_normal, pos_drone - center)

        if computed_corners:
            corners_earth = np.array(
                [center + R_gate.apply(lp) for lp in inner_corners_local + outer_corners_local],
                dtype=np.float64,
            )
        else:
            corners_earth = gate_mean_corners[gate_id].astype(np.float64)

        if gate_depth != 0.0:
            corners_earth += gate_normal * (gate_depth * np.sign(dot))

        if dot < 0:
            swap = [1, 0, 3, 2]
            full_swap = swap + [i + 4 for i in swap] if computed_corners else swap
            corners_earth = corners_earth[full_swap]

        corners_body = R_drone_inv.apply(corners_earth - pos_drone)
        corners_cam = R_cam_inv.apply(corners_body - t_cam)

        if fisheye:
            projected, _ = cv2.fisheye.projectPoints(
                corners_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist.reshape(4, 1))
        else:
            projected, _ = cv2.projectPoints(
                corners_cam, np.zeros(3), np.zeros(3), K, dist)
        projected = projected.reshape(-1, 2)

        z = corners_cam[:, 2]
        in_front = z > 0
        if not fisheye:
            r_crit = _compute_r_crit(dist)
            if np.isfinite(r_crit):
                safe_z = np.where(z != 0, z, 1e-9)
                r_u = np.sqrt((corners_cam[:, 0] / safe_z) ** 2 +
                              (corners_cam[:, 1] / safe_z) ** 2)
                no_wraparound = r_u < r_crit
            else:
                no_wraparound = np.ones(len(corners_cam), dtype=bool)
        else:
            no_wraparound = np.ones(len(corners_cam), dtype=bool)

        visible = (
            in_front & no_wraparound
            & (projected[:, 0] >= 0) & (projected[:, 0] < img_width)
            & (projected[:, 1] >= 0) & (projected[:, 1] < img_height)
        )

        if np.count_nonzero(visible) < min_visible_corners:
            continue

        depth = float(corners_cam[:, 2].mean())
        candidates.append([projected, visible.copy(), depth, in_front.copy(), no_wraparound.copy(), kp_names])

    candidates.sort(key=lambda c: c[2])
    if computed_corners:
        _apply_occlusion(candidates)

    result = []
    for projected, visible, _, in_front, no_wrap, names in candidates:
        if np.count_nonzero(visible) < min_visible_corners:
            continue
        result.append((projected, visible, in_front & no_wrap, names))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D gate reprojection on flight images (no ROS2 needed).")
    parser.add_argument('--flight', required=True, help="Flight ID, e.g. flight-01p-ellipse")
    parser.add_argument('--computed-corners', action='store_true',
                        help="Compute 8 corners from gate size instead of CSV mocap corners")
    parser.add_argument('--interior-size', type=float, default=1.5)
    parser.add_argument('--exterior-size', type=float, default=2.7)
    parser.add_argument('--gate-depth', type=float, default=0.0)
    parser.add_argument('--min-visible-corners', type=int, default=3)
    parser.add_argument('--fix-rotation', action='store_true')
    parser.add_argument('--fisheye', action='store_true',
                        help="Use Kannala-Brandt fisheye projection model")
    parser.add_argument('--labels', action='store_true',
                        help="Draw corner name labels next to each X mark")
    parser.add_argument('--v-only', action='store_true',
                        help="Only draw visible (green) corners, skip non-visible red marks")
    parser.add_argument('--start', type=float, default=None,
                        help="Skip frames before N seconds from first image")
    parser.add_argument('--end', type=float, default=None,
                        help="Skip frames after N seconds from first image")
    parser.add_argument('--model', default=None,
                        help="Path to TensorRT .engine file for YOLO pose inference "
                             "(draws blue X for each detected keypoint)")
    parser.add_argument('--conf', type=float, default=0.75,
                        help="Detection confidence threshold for --model inference (default: 0.75)")
    parser.add_argument('--show-rectified', action='store_true',
                        help="Open a second window showing the rectified image with inference drawn")
    parser.add_argument('--match-dist', type=float, default=50.0,
                        help="Max pixel distance for Hungarian corner matching (default: 50)")
    args = parser.parse_args()

    flight_type = "piloted" if "p-" in args.flight else "autonomous"
    flight_dir = os.path.join("..", "data", flight_type, args.flight)

    # Load calibration
    calib_path = get_calibration_path(args.flight)
    with open(calib_path) as f:
        calib = json.load(f)
    K = np.array(calib['mtx'])
    dist = np.array(calib['dist'][0])
    if args.fisheye and len(dist) > 4:
        dist = dist[:4]

    # Load extrinsics
    cam_trans, cam_quat = get_camera_extrinsics(args.flight)
    t_cam = np.array(cam_trans)
    R_cam_inv = Rotation.from_quat(cam_quat).inv()

    # Load gate corners
    corners_csv = os.path.join(flight_dir, 'csv_raw', f'gate_corners_{args.flight}.csv')
    gate_corners_df = pd.read_csv(corners_csv)
    gate_ids = sorted({
        int(col.split('_')[0][4:])
        for col in gate_corners_df.columns
        if col.startswith('gate') and '_marker1_x' in col
    })
    gate_quats, gate_mean_centers, gate_mean_corners = precompute_gate_orientations(
        gate_corners_df, gate_ids, args.fix_rotation)

    if args.computed_corners:
        half_i = args.interior_size / 2.0
        half_o = args.exterior_size / 2.0
        inner_corners_local = [
            np.array([0.0, -half_i,  half_i]),
            np.array([0.0,  half_i,  half_i]),
            np.array([0.0,  half_i, -half_i]),
            np.array([0.0, -half_i, -half_i]),
        ]
        outer_corners_local = [
            np.array([0.0, -half_o,  half_o]),
            np.array([0.0,  half_o,  half_o]),
            np.array([0.0,  half_o, -half_o]),
            np.array([0.0, -half_o, -half_o]),
        ]
    else:
        inner_corners_local = outer_corners_local = None

    # Load synced CSV for drone pose (one row per camera frame)
    synced_csv = os.path.join(flight_dir, f'{args.flight}_cam_ts_sync.csv')
    pose_df = pd.read_csv(synced_csv)
    pose_ts_arr = pose_df['timestamp'].values

    # Load images
    image_dir = os.path.join(flight_dir, f'camera_{args.flight}')
    images = sorted(glob(os.path.join(image_dir, '*.jpg')))
    if not images:
        print(f"No images found in {image_dir}")
        return

    first_img = cv2.imread(images[0])
    img_height, img_width = first_img.shape[:2]

    first_ts = int(images[0].split('_')[-1].split('.')[0])
    cut_start_us = first_ts + int(args.start * 1e6) if args.start is not None else None
    cut_end_us = first_ts + int(args.end * 1e6) if args.end is not None else None

    # Filter image list to time window once
    if cut_start_us is not None or cut_end_us is not None:
        images = [
            p for p in images
            if (cut_start_us is None or int(p.split('_')[-1].split('.')[0]) >= cut_start_us)
            and (cut_end_us is None or int(p.split('_')[-1].split('.')[0]) <= cut_end_us)
        ]

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    ORANGE = (0, 165, 255)

    # Rectification maps for model inference (inference is done on rectified image)
    rect_map1 = rect_map2 = new_K = None
    if args.model:
        if args.fisheye:
            D4 = dist.reshape(4, 1)
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D4, (img_width, img_height), np.eye(3), balance=0.0)
            rect_map1, rect_map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D4, np.eye(3), new_K, (img_width, img_height), cv2.CV_32FC1)
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                K, dist, (img_width, img_height), alpha=0)
            rect_map1, rect_map2 = cv2.initUndistortRectifyMap(
                K, dist, None, new_K, (img_width, img_height), cv2.CV_32FC1)
        if args.model.endswith('.pt'):
            yolo_model = UltralyticsYoloPose(args.model, conf=args.conf)
        else:
            yolo_model = TRTYoloPose(args.model, conf=args.conf)
    else:
        yolo_model = None

    cv2.namedWindow('reprojection', cv2.WINDOW_NORMAL)
    if args.show_rectified and args.model:
        cv2.namedWindow('rectified', cv2.WINDOW_NORMAL)
    print("Controls: SPACE/→ next  ←  prev  ↑ +10  ↓ -10  q/ESC quit")

    idx = 0
    while 0 <= idx < len(images):
        image_path = images[idx]
        ts_us = int(image_path.split('_')[-1].split('.')[0])
        pose_idx = _find_nearest_idx(pose_ts_arr, ts_us)
        pose_row = pose_df.iloc[pose_idx]

        img = cv2.imread(image_path)

        gate_projections = project_gates(
            pose_row, gate_ids, gate_quats, gate_mean_centers, gate_mean_corners,
            args.computed_corners, inner_corners_local, outer_corners_local,
            K, dist, R_cam_inv, t_cam,
            img_width, img_height,
            args.min_visible_corners, args.fisheye, args.gate_depth,
        )

        # Build per-gate corner dicts {label: (x,y)} for visible corners
        gate_corners = []
        for projected, visible, drawable, kp_names in gate_projections:
            gate = {}
            for i, (pt, vis, draw) in enumerate(zip(projected, visible, drawable)):
                if vis:
                    color = GREEN
                elif draw and not args.v_only:
                    color = RED
                else:
                    continue
                draw_x(img, pt, color)
                if args.labels:
                    parts = kp_names[i].split('_')
                    label = parts[0][0] + parts[1][0] + parts[2][0]
                    lx, ly = int(round(float(pt[0]))) + 10, int(round(float(pt[1]))) - 5
                    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                if vis:
                    parts = kp_names[i].split('_')
                    lbl = parts[0][0] + parts[1][0] + parts[2][0]
                    gate[lbl] = (float(pt[0]), float(pt[1]))
            gate_corners.append(gate)

        if yolo_model is not None:
            rect_img = cv2.remap(img, rect_map1, rect_map2, cv2.INTER_LINEAR)
            detections = yolo_model(rect_img)
            print(f"  model: {len(detections)} detections (conf≥{args.conf})")

            # Back-project all detections and build per-detection corner dicts
            det_corners = []   # [{label: (x,y)}, ...]  in distorted space
            det_rect_pts = []  # [{label: (kx,ky)}, ...]  in rectified space (for show_rectified)
            for kps in detections:
                pts_rect = np.array([[kx, ky] for kx, ky, _ in kps], dtype=np.float64)
                x_n = (pts_rect[:, 0] - new_K[0, 2]) / new_K[0, 0]
                y_n = (pts_rect[:, 1] - new_K[1, 2]) / new_K[1, 1]
                pts3d = np.stack([x_n, y_n, np.ones_like(x_n)], axis=1).reshape(-1, 1, 3)
                if args.fisheye:
                    proj, _ = cv2.fisheye.projectPoints(
                        pts3d, np.zeros(3), np.zeros(3), K, dist.reshape(4, 1))
                else:
                    proj, _ = cv2.projectPoints(
                        pts3d, np.zeros(3), np.zeros(3), K, dist)
                proj = proj.reshape(-1, 2)
                kp_labels = _MODEL_KP_LABELS_8 if len(kps) == 8 else _MODEL_KP_LABELS_4
                det = {}
                det_r = {}
                for i, (kx, ky, kc) in enumerate(kps):
                    if kc < 0.5:
                        continue
                    label = kp_labels[i] if i < len(kp_labels) else str(i)
                    det[label] = (float(proj[i, 0]), float(proj[i, 1]))
                    det_r[label] = (float(kx), float(ky))
                det_corners.append(det)
                det_rect_pts.append(det_r)

            # Draw blue X for all detected keypoints
            for det, det_r in zip(det_corners, det_rect_pts):
                for label, (dx, dy) in det.items():
                    draw_x(img, (dx, dy), BLUE)
                    if args.labels:
                        cv2.putText(img, label,
                                    (int(round(dx)) + 10, int(round(dy)) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1)
                    if args.show_rectified:
                        rx, ry = det_r[label]
                        draw_x(rect_img, (rx, ry), BLUE)
                        if args.labels:
                            cv2.putText(rect_img, label,
                                        (int(round(rx)) + 10, int(round(ry)) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1)

            # --- Gate-level Hungarian: assign each detection to one gate ---
            candidates = []
            if gate_corners and det_corners:
                n_g, n_d = len(gate_corners), len(det_corners)
                cost_mat = np.full((n_g, n_d), np.inf)
                for gi, gate in enumerate(gate_corners):
                    for di, det in enumerate(det_corners):
                        common = set(gate) & set(det)
                        if not common:
                            continue
                        cost_mat[gi, di] = np.mean([
                            np.linalg.norm(np.array(gate[l]) - np.array(det[l]))
                            for l in common])
                # Replace inf so linear_sum_assignment can run
                big = np.nanmax(cost_mat[np.isfinite(cost_mat)]) * 10 + 1000 \
                    if np.any(np.isfinite(cost_mat)) else 1000.0
                row_ind, col_ind = linear_sum_assignment(
                    np.where(np.isfinite(cost_mat), cost_mat, big))
                for gi, di in zip(row_ind, col_ind):
                    if not np.isfinite(cost_mat[gi, di]):
                        continue
                    gate = gate_corners[gi]
                    det = det_corners[di]
                    for lbl in set(gate) & set(det):
                        gpt = gate[lbl]
                        dpt = det[lbl]
                        d = float(np.linalg.norm(np.array(gpt) - np.array(dpt)))
                        if d <= args.match_dist:
                            candidates.append((d,
                                               (int(round(gpt[0])), int(round(gpt[1]))),
                                               (int(round(dpt[0])), int(round(dpt[1])))))
            # IQR outlier rejection across all matched corner pairs
            if len(candidates) >= 4:
                dists = np.array([d for d, _, _ in candidates])
                q1, q3 = np.percentile(dists, [25, 75])
                iqr = q3 - q1
                upper = q3 + 1.5 * iqr
                candidates = [(d, pg, pm) for d, pg, pm in candidates if d <= upper]
            for _, pt_g, pt_m in candidates:
                cv2.line(img, pt_g, pt_m, ORANGE, 1)

            if args.show_rectified:
                cv2.imshow('rectified', rect_img)

        print(f"[{idx + 1}/{len(images)}] ts={ts_us}  gates_visible={len(gate_projections)}")
        cv2.imshow('reprojection', img)

        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            break
        elif key == 82 or key == ord('w'):   # up arrow or w → +10
            idx = min(idx + 10, len(images) - 1)
        elif key == 84 or key == ord('s'):   # down arrow or s → -10
            idx = max(idx - 10, 0)
        elif key == 81 or key == ord('a'):   # left arrow or a → -1
            idx = max(idx - 1, 0)
        else:                                 # anything else (space, right arrow, etc.) → +1
            idx += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
