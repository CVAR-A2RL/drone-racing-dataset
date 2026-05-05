#!/usr/bin/env python3
"""
Robust fisheye camera calibration from drone-gate corner labels.

Designed for the setup: ~5000 frames, 4 corner labels per gate, multiple gates
visible per frame, horizontal FoV ~2.30 rad (~132 deg) -> Kannala-Brandt model.

Why a custom bundle adjustment (not cv2.fisheye.calibrate)
----------------------------------------------------------
Two hard limits make OpenCV's calibrator unusable here:
  1. cv2.fisheye.calibrate REQUIRES coplanar object points per view (the
     internal homography init in InitExtrinsics asserts on it). Points across
     several gates in a shared world frame are NOT coplanar.
  2. cv2.fisheye.calibrate has an undocumented floor of >=5 points per view.
     A single 4-corner gate is rejected outright (Mat rowRange assertion).

Both limits go away if we run our own optimisation. We use:
  - cv2.fisheye.projectPoints for the forward model (Kannala-Brandt).
  - scipy.optimize.least_squares with a sparse Jacobian, Cauchy robust loss,
    and trust-region method.

Pipeline
--------
1. Load 2D pixel labels and 3D gate corner ground truth (shared world frame).
2. Build a FrameView per frame (all gates' corners stacked, in world coords).
3. Coverage-based greedy frame selection: pick a small (~150) subset that
   covers image space well, with a bonus for multi-gate frames (richer
   non-coplanar constraints break the focal-vs-depth ambiguity).
4. Initial K from declared HFoV (Kannala-Brandt: f ~= W / HFoV near axis),
   D = zeros, principal point at image centre.
5. Per-frame initial extrinsics via fisheye-aware PnP (undistortPoints +
   solvePnP on identity intrinsics).
6. Joint bundle adjustment over [fx, fy, cx, cy, k1..k4, all (rvec, tvec)]
   with sparse Jacobian and Cauchy loss. Iterative outlier rejection: drop
   frames whose per-view RMS exceeds k * median, refit.
7. Save calibration as YAML + JSON, plus diagnostic PNGs.

Input file formats
------------------
gates_3d.json:
  { "<gate_id>": [[X,Y,Z], [X,Y,Z], [X,Y,Z], [X,Y,Z]], ... }
Coordinates are in a shared world frame (metres). Corner order MUST be
consistent across files (e.g. always TL, TR, BR, BL).

labels.json:
  { "image_size": [W, H],
    "frames": [
      { "frame_id": "0001",
        "gates": [
          {"gate_id": "<gate_id>", "corners_2d": [[u,v],[u,v],[u,v],[u,v]]},
          ...
        ] },
      ... ] }

Usage
-----
Real data:
  python calibrate_fisheye.py \
      --labels labels.json --gates-3d gates_3d.json \
      --image-size 1280 720 --hfov-rad 2.30 \
      --output-dir ./calib_out

Synthetic self-test (returns non-zero on bad recovery):
  python calibrate_fisheye.py --demo --output-dir ./calib_out
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FrameView:
    """All 2D-3D correspondences for one frame, in the shared world frame."""
    frame_id: str
    object_points: np.ndarray  # (N, 3) float64, world coords
    image_points: np.ndarray   # (N, 2) float64
    gate_ids: list[str] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return self.object_points.shape[0]


@dataclass
class CalibrationResult:
    K: np.ndarray
    D: np.ndarray
    image_size: tuple[int, int]
    rms: float
    per_frame_rms: np.ndarray
    used_frame_ids: list[str]
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_gates_3d(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        raw = json.load(f)
    out: dict[str, np.ndarray] = {}
    for gid, corners in raw.items():
        arr = np.asarray(corners, dtype=np.float64)
        if arr.shape != (4, 3):
            raise ValueError(f"Gate {gid}: expected (4,3) corners, got {arr.shape}")
        out[gid] = arr
    return out


def load_labels(path: Path) -> tuple[list[dict], Optional[tuple[int, int]]]:
    with open(path) as f:
        raw = json.load(f)
    image_size: Optional[tuple[int, int]] = None
    if "image_size" in raw:
        image_size = (int(raw["image_size"][0]), int(raw["image_size"][1]))
    return raw["frames"], image_size


def build_frame_views(
    frames_raw: list[dict],
    gates_3d: dict[str, np.ndarray],
    min_gates: int,
) -> list[FrameView]:
    out: list[FrameView] = []
    for fr in frames_raw:
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        gids: list[str] = []
        for g in fr.get("gates", []):
            gid = g["gate_id"]
            if gid not in gates_3d:
                continue
            c2d = np.asarray(g["corners_2d"], dtype=np.float64)
            if c2d.shape != (4, 2) or not np.isfinite(c2d).all():
                continue
            obj_pts.append(gates_3d[gid])
            img_pts.append(c2d)
            gids.append(gid)
        if len(obj_pts) < min_gates:
            continue
        out.append(FrameView(
            frame_id=str(fr.get("frame_id", f"frame_{len(out)}")),
            object_points=np.vstack(obj_pts),
            image_points=np.vstack(img_pts),
            gate_ids=gids,
        ))
    return out


# ---------------------------------------------------------------------------
# Coverage-greedy frame selection
# ---------------------------------------------------------------------------

def select_frames_by_coverage(
    frame_views: list[FrameView],
    image_size: tuple[int, int],
    n_target: int,
    grid: int = 8,
) -> list[FrameView]:
    """
    Greedy: at each step pick the frame that adds the most uncovered grid cells,
    weighted to prefer multi-gate frames (richer non-coplanar constraints).
    """
    if len(frame_views) <= n_target:
        return list(frame_views)

    W, H = image_size
    cell_sets: list[set[tuple[int, int]]] = []
    for v in frame_views:
        gx = np.clip((v.image_points[:, 0] / W * grid).astype(int), 0, grid - 1)
        gy = np.clip((v.image_points[:, 1] / H * grid).astype(int), 0, grid - 1)
        cell_sets.append(set(zip(gx.tolist(), gy.tolist())))

    coverage_count = np.zeros((grid, grid), dtype=np.int64)
    chosen_mask = np.zeros(len(frame_views), dtype=bool)
    chosen: list[int] = []

    for _ in range(n_target):
        best_idx, best_score = -1, -1.0
        for i, cells in enumerate(cell_sets):
            if chosen_mask[i]:
                continue
            score = sum(1.0 / (1.0 + coverage_count[cy, cx]) for (cx, cy) in cells)
            score += 0.05 * len(frame_views[i].gate_ids)  # multi-gate bonus
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        chosen.append(best_idx)
        chosen_mask[best_idx] = True
        for (cx, cy) in cell_sets[best_idx]:
            coverage_count[cy, cx] += 1

    return [frame_views[i] for i in chosen]


# ---------------------------------------------------------------------------
# Initial intrinsic guess
# ---------------------------------------------------------------------------

def initial_K_from_hfov(image_size: tuple[int, int], hfov_rad: float) -> np.ndarray:
    """Kannala-Brandt: r = f * theta near optical axis. f ~= W / HFoV."""
    W, H = image_size
    f = W / hfov_rad
    return np.array(
        [[f, 0.0, W / 2.0],
         [0.0, f, H / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Per-frame initial extrinsics via fisheye PnP
# ---------------------------------------------------------------------------

def initial_extrinsics_pnp(
    frame_views: list[FrameView], K: np.ndarray, D: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[bool]]:
    """
    Per-frame initial (rvec, tvec) in world frame.
    Approach: undistort image points to normalized (z=1) coords via the current
    fisheye model, then solvePnP with K=I, D=0. This is the standard fisheye
    PnP recipe.
    """
    rvecs, tvecs, ok_flags = [], [], []
    for v in frame_views:
        und = cv2.fisheye.undistortPoints(
            v.image_points.reshape(-1, 1, 2).astype(np.float64), K, D
        ).reshape(-1, 2)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                v.object_points.astype(np.float64),
                und.astype(np.float64),
                np.eye(3), np.zeros(4),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            ok = False
        if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            rvecs.append(np.zeros(3))
            tvecs.append(np.array([0.0, 0.0, 1.0]))
            ok_flags.append(False)
        else:
            rvecs.append(rvec.reshape(3))
            tvecs.append(tvec.reshape(3))
            ok_flags.append(True)
    return rvecs, tvecs, ok_flags


# ---------------------------------------------------------------------------
# Bundle adjustment core
# ---------------------------------------------------------------------------

def _pack(K: np.ndarray, D: np.ndarray, rvecs: list[np.ndarray], tvecs: list[np.ndarray]) -> np.ndarray:
    parts = [np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]), D.reshape(-1)]
    for r, t in zip(rvecs, tvecs):
        parts.append(r.reshape(-1))
        parts.append(t.reshape(-1))
    return np.concatenate(parts)


def _unpack(theta: np.ndarray, n_frames: int):
    fx, fy, cx, cy = theta[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = theta[4:8].copy()
    rvecs, tvecs = [], []
    off = 8
    for _ in range(n_frames):
        rvecs.append(theta[off: off + 3].copy())
        tvecs.append(theta[off + 3: off + 6].copy())
        off += 6
    return K, D, rvecs, tvecs


def _residuals(theta: np.ndarray, frame_views: list[FrameView]) -> np.ndarray:
    K, D, rvecs, tvecs = _unpack(theta, len(frame_views))
    res: list[np.ndarray] = []
    for v, r, t in zip(frame_views, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(
            v.object_points.reshape(-1, 1, 3),
            r.reshape(3, 1), t.reshape(3, 1), K, D,
        )
        proj = proj.reshape(-1, 2)
        res.append((proj - v.image_points).ravel())
    return np.concatenate(res)


def _build_jac_sparsity(frame_views: list[FrameView]) -> lil_matrix:
    n_frames = len(frame_views)
    n_res = sum(2 * v.n_points for v in frame_views)
    n_par = 8 + 6 * n_frames
    S = lil_matrix((n_res, n_par), dtype=np.uint8)
    row = 0
    for i, v in enumerate(frame_views):
        rows = 2 * v.n_points
        S[row: row + rows, 0:8] = 1                         # K + D
        S[row: row + rows, 8 + 6 * i: 8 + 6 * i + 6] = 1    # this frame's (rvec, tvec)
        row += rows
    return S


def bundle_adjust(
    frame_views: list[FrameView],
    K_init: np.ndarray,
    D_init: np.ndarray,
    fix_principal_point: bool = False,
    fix_k4: bool = False,
    loss: str = "cauchy",
    f_scale: float = 1.5,
    max_nfev: int = 200,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[FrameView], float]:
    """One BA pass. Drops frames whose PnP init fails."""
    rvecs0, tvecs0, ok = initial_extrinsics_pnp(frame_views, K_init, D_init)
    fv = [v for v, k in zip(frame_views, ok) if k]
    rvecs0 = [r for r, k in zip(rvecs0, ok) if k]
    tvecs0 = [t for t, k in zip(tvecs0, ok) if k]
    if len(fv) < 8:
        raise RuntimeError(f"Only {len(fv)} frames passed PnP init.")

    theta0 = _pack(K_init, D_init, rvecs0, tvecs0)
    sparsity = _build_jac_sparsity(fv)

    if fix_principal_point or fix_k4:
        # Use bounds to fix specific parameters; otherwise unconstrained.
        EPS = 1e-6
        lb = np.full_like(theta0, -np.inf)
        ub = np.full_like(theta0, np.inf)
        if fix_principal_point:
            lb[2] = theta0[2] - EPS; ub[2] = theta0[2] + EPS
            lb[3] = theta0[3] - EPS; ub[3] = theta0[3] + EPS
        if fix_k4:
            theta0[7] = 0.0
            lb[7] = -EPS; ub[7] = EPS
        bounds_arg = (lb, ub)
        method = "trf"  # bounds require trf
    else:
        bounds_arg = (-np.inf, np.inf)
        method = "trf"  # trf still supports sparse jac and robust loss

    if verbose:
        r0 = _residuals(theta0, fv)
        rms0 = float(np.sqrt(np.mean(r0 * r0)))
        print(f"  [BA init] frames={len(fv)}  RMS={rms0:.4f}  params={len(theta0)}  residuals={len(r0)}")

    result = least_squares(
        _residuals, theta0, args=(fv,),
        jac_sparsity=sparsity, method=method,
        loss=loss, f_scale=f_scale, x_scale="jac",
        bounds=bounds_arg,
        max_nfev=max_nfev, verbose=2 if verbose else 0,
    )
    K, D, rvecs, tvecs = _unpack(result.x, len(fv))
    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    if verbose:
        print(f"  [BA done] RMS={rms:.4f}  cost={result.cost:.3f}  "
              f"nfev={result.nfev}  status={result.status}")
    return K, D, rvecs, tvecs, fv, rms


# ---------------------------------------------------------------------------
# Iterative calibration with outlier rejection
# ---------------------------------------------------------------------------

def per_frame_rms(
    frame_views: list[FrameView],
    K: np.ndarray, D: np.ndarray,
    rvecs: list[np.ndarray], tvecs: list[np.ndarray],
) -> np.ndarray:
    out = np.zeros(len(frame_views))
    for i, v in enumerate(frame_views):
        proj, _ = cv2.fisheye.projectPoints(
            v.object_points.reshape(-1, 1, 3),
            np.asarray(rvecs[i]).reshape(3, 1), np.asarray(tvecs[i]).reshape(3, 1),
            K, D,
        )
        proj = proj.reshape(-1, 2)
        diff = proj - v.image_points
        out[i] = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return out


def calibrate_iterative(
    frame_views: list[FrameView],
    image_size: tuple[int, int],
    hfov_rad: float,
    max_outer_iter: int = 3,
    outlier_k: float = 3.0,
    fix_k4: bool = False,
    loss: str = "huber",
    f_scale: float = 1.5,
    max_nfev: int = 2000,
    verbose: bool = True,
) -> tuple[CalibrationResult, list[FrameView]]:
    K = initial_K_from_hfov(image_size, hfov_rad)
    D = np.zeros(4)
    if verbose:
        print(f"[init] f0 = {K[0,0]:.2f} px (HFoV={hfov_rad:.3f} rad), "
              f"cx0={K[0,2]:.1f}, cy0={K[1,2]:.1f}")

    current = list(frame_views)

    # Warm-up: linear loss to converge from the rough initial guess.
    # Robust losses (Cauchy, Huber) saturate at large residuals and can stall
    # when initial residuals are huge (typical with HFoV-only K_init + D=0).
    if verbose:
        print(f"[BA warm-up] linear loss, frames={len(current)}")
    K, D, rvecs, tvecs, current, _ = bundle_adjust(
        current, K, D, fix_principal_point=False, fix_k4=fix_k4,
        loss="linear", f_scale=f_scale, max_nfev=max_nfev, verbose=verbose,
    )

    # Robust passes with outlier rejection
    for it in range(max_outer_iter):
        if len(current) < 12:
            raise RuntimeError(f"Only {len(current)} frames left after outlier rejection.")
        pf = per_frame_rms(current, K, D, rvecs, tvecs)
        med = float(np.median(pf))
        thresh = max(outlier_k * med, 1.0)
        keep = pf <= thresh
        n_drop = int((~keep).sum())
        if verbose:
            print(f"[outlier {it}] frames={len(current)}  median={med:.3f} "
                  f"max={pf.max():.3f}  thresh={thresh:.3f}  drop={n_drop}")
        if n_drop == 0 and it > 0:
            break
        if n_drop > 0:
            current = [v for v, k in zip(current, keep) if k]
        if verbose:
            print(f"[BA pass {it+1}] {loss} loss, frames={len(current)}")
        K, D, rvecs, tvecs, current, _ = bundle_adjust(
            current, K, D, fix_principal_point=False, fix_k4=fix_k4,
            loss=loss, f_scale=f_scale, max_nfev=max_nfev, verbose=verbose,
        )

    pf = per_frame_rms(current, K, D, rvecs, tvecs)
    final_rms = float(np.sqrt(np.mean(pf ** 2)))
    return (
        CalibrationResult(
            K=K, D=D, image_size=image_size,
            rms=final_rms, per_frame_rms=pf,
            used_frame_ids=[v.frame_id for v in current],
            rvecs=rvecs, tvecs=tvecs,
        ),
        current,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_calibration(result: CalibrationResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "fisheye_kannala_brandt",
        "image_size": list(result.image_size),
        "K": result.K.tolist(),
        "D": result.D.tolist(),
        "rms_reprojection_error_px": float(result.rms),
        "n_frames_used": len(result.used_frame_ids),
        "per_frame_rms_summary": {
            "median": float(np.median(result.per_frame_rms)),
            "p95": float(np.percentile(result.per_frame_rms, 95)),
            "max": float(np.max(result.per_frame_rms)),
        },
    }
    with open(out_dir / "calibration.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "calibration.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def save_diagnostics(used_views: list[FrameView], result: CalibrationResult, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = result.image_size
    K, D = result.K, result.D

    all_pts: list[np.ndarray] = []
    all_resid: list[np.ndarray] = []
    for v, rv, tv in zip(used_views, result.rvecs, result.tvecs):
        proj, _ = cv2.fisheye.projectPoints(
            v.object_points.reshape(-1, 1, 3),
            np.asarray(rv).reshape(3, 1), np.asarray(tv).reshape(3, 1), K, D,
        )
        proj = proj.reshape(-1, 2)
        all_pts.append(v.image_points)
        all_resid.append(proj - v.image_points)
    pts = np.vstack(all_pts)
    resid = np.vstack(all_resid)
    resid_norm = np.linalg.norm(resid, axis=1)

    # 1. Coverage map
    fig, ax = plt.subplots(figsize=(7, 7 * H / W))
    ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, c="C0")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_title(f"Corner coverage (n={len(pts)} points, {len(used_views)} frames)")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    fig.tight_layout(); fig.savefig(out_dir / "coverage.png", dpi=120); plt.close(fig)

    # 2. Per-frame RMS histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(result.per_frame_rms, bins=30, color="C2", edgecolor="black")
    med = float(np.median(result.per_frame_rms))
    ax.axvline(med, color="C3", ls="--", label=f"median={med:.3f}px")
    ax.set_xlabel("Per-frame RMS reprojection error (px)")
    ax.set_ylabel("# frames"); ax.legend()
    ax.set_title(f"Final RMS = {result.rms:.4f} px")
    fig.tight_layout(); fig.savefig(out_dir / "per_frame_rms.png", dpi=120); plt.close(fig)

    # 3. Residual vector field
    fig, ax = plt.subplots(figsize=(7, 7 * H / W))
    sel = (np.random.default_rng(0).choice(len(pts), size=2000, replace=False)
           if len(pts) > 2000 else np.arange(len(pts)))
    ax.quiver(pts[sel, 0], pts[sel, 1], resid[sel, 0], resid[sel, 1],
              resid_norm[sel], angles="xy", scale_units="xy", scale=0.5,
              cmap="viridis", width=0.002)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_title("Reprojection residuals (x2 magnification)")
    fig.tight_layout(); fig.savefig(out_dir / "residual_field.png", dpi=120); plt.close(fig)

    # 4. |residual| vs radius - flat/random pattern means distortion model fits
    cx, cy = K[0, 2], K[1, 2]
    r = np.linalg.norm(pts - np.array([[cx, cy]]), axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(r, resid_norm, s=3, alpha=0.3)
    ax.set_xlabel("Distance from principal point (px)")
    ax.set_ylabel("|residual| (px)")
    ax.set_title("Residual magnitude vs. radius (flat = good distortion fit)")
    fig.tight_layout(); fig.savefig(out_dir / "residual_vs_radius.png", dpi=120); plt.close(fig)

    with open(out_dir / "used_frames.txt", "w") as f:
        for fid, e in zip(result.used_frame_ids, result.per_frame_rms):
            f.write(f"{fid}\t{e:.6f}\n")


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_frames: int = 400,
    image_size: tuple[int, int] = (1280, 720),
    hfov_rad: float = 2.30,
    label_noise_px: float = 0.4,
    seed: int = 0,
) -> tuple[list[dict], dict[str, np.ndarray], tuple[int, int], dict]:
    rng = np.random.default_rng(seed)
    W, H = image_size
    f_true = W / hfov_rad
    K_true = np.array([[f_true, 0, W / 2 + 3.0],
                       [0, f_true, H / 2 - 2.0],
                       [0, 0, 1]], dtype=np.float64)
    D_true = np.array([-0.04, 0.005, -0.001, 0.0005], dtype=np.float64)

    gate_ids = [f"gate_{i:02d}" for i in range(6)]
    gate_centers = np.array([
        [0, 0, 0], [4, 1, 0.2], [8, -1, 0.5],
        [10, 3, 0.8], [6, 5, 0.3], [2, 4, 0.6],
    ], dtype=np.float64)
    half = 0.7
    local = np.array([[-half,  half, 0],
                      [ half,  half, 0],
                      [ half, -half, 0],
                      [-half, -half, 0]])
    gates_3d: dict[str, np.ndarray] = {}
    for gid, c in zip(gate_ids, gate_centers):
        a = rng.uniform(-0.6, 0.6)
        Rz = np.array([[np.cos(a), -np.sin(a), 0],
                       [np.sin(a),  np.cos(a), 0],
                       [0, 0, 1]])
        gates_3d[gid] = (local @ Rz.T) + c

    frames_raw: list[dict] = []
    for i in range(n_frames):
        cam_pos = np.array([
            rng.uniform(-3, 11),
            rng.uniform(-4, 7),
            rng.uniform(-1.5, 2.5),
        ])
        target = gate_centers[rng.integers(0, len(gate_centers))] + rng.normal(0, 0.6, 3)
        forward = target - cam_pos
        forward /= np.linalg.norm(forward) + 1e-9
        up_world = np.array([0, 0, 1.0]) + rng.normal(0, 0.15, 3)
        right = np.cross(forward, up_world); right /= np.linalg.norm(right) + 1e-9
        up = np.cross(right, forward)
        R_wc = np.column_stack([right, -up, forward])
        R_cw = R_wc.T
        t_cw = -R_cw @ cam_pos
        rvec, _ = cv2.Rodrigues(R_cw)

        gates_in_frame = []
        for gid, corners in gates_3d.items():
            cam_pts = (R_cw @ corners.T + t_cw.reshape(3, 1)).T
            if (cam_pts[:, 2] <= 0.3).any():
                continue
            proj, _ = cv2.fisheye.projectPoints(
                corners.reshape(-1, 1, 3), rvec, t_cw.reshape(3, 1), K_true, D_true
            )
            proj = proj.reshape(-1, 2)
            if (proj[:, 0] < 5).any() or (proj[:, 0] >= W - 5).any():
                continue
            if (proj[:, 1] < 5).any() or (proj[:, 1] >= H - 5).any():
                continue
            proj_noisy = proj + rng.normal(0, label_noise_px, proj.shape)
            gates_in_frame.append({"gate_id": gid, "corners_2d": proj_noisy.tolist()})
        if gates_in_frame:
            frames_raw.append({"frame_id": f"synth_{i:04d}", "gates": gates_in_frame})

    truth = {"K": K_true.tolist(), "D": D_true.tolist(), "image_size": list(image_size)}
    return frames_raw, gates_3d, image_size, truth


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust fisheye calibration via bundle adjustment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--labels", type=Path, help="Path to labels JSON.")
    p.add_argument("--gates-3d", type=Path, help="Path to 3D gate corners JSON.")
    p.add_argument("--image-size", type=int, nargs=2, metavar=("W", "H"))
    p.add_argument("--hfov-rad", type=float, default=2.30,
                   help="Initial horizontal FoV (rad). Default 2.30.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=150,
                   help="Target number of frames after coverage selection.")
    p.add_argument("--min-gates-per-frame", type=int, default=1,
                   help=">=2 enforces multi-gate frames (recommended).")
    p.add_argument("--max-outer-iter", type=int, default=3,
                   help="Outlier-rejection iterations after the first BA.")
    p.add_argument("--outlier-k", type=float, default=3.0)
    p.add_argument("--fix-k4", action="store_true",
                   help="Fix the 4th distortion coefficient at zero.")
    p.add_argument("--ba-loss", choices=["linear", "huber", "cauchy", "soft_l1"], default="huber",
                   help="Robust loss for the post-warm-up passes (default: huber).")
    p.add_argument("--ba-fscale", type=float, default=1.5,
                   help="Scale (px) for robust loss.")
    p.add_argument("--ba-max-nfev", type=int, default=2000)
    p.add_argument("--demo", action="store_true",
                   help="Synthetic self-test (returns non-zero on bad recovery).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    if args.demo:
        print("=== DEMO MODE: synthetic dataset ===")
        frames_raw, gates_3d, image_size, truth = generate_synthetic_dataset(
            n_frames=400, hfov_rad=args.hfov_rad,
        )
        print(f"  synthetic frames: {len(frames_raw)}, gates: {len(gates_3d)}")
        with open(args.output_dir / "demo_labels.json", "w") as f:
            json.dump({"image_size": list(image_size), "frames": frames_raw}, f)
        with open(args.output_dir / "demo_gates_3d.json", "w") as f:
            json.dump({k: v.tolist() for k, v in gates_3d.items()}, f)
        with open(args.output_dir / "demo_truth.json", "w") as f:
            json.dump(truth, f, indent=2)
    else:
        if not args.labels or not args.gates_3d:
            print("ERROR: --labels and --gates-3d required (or use --demo).", file=sys.stderr)
            return 2
        gates_3d = load_gates_3d(args.gates_3d)
        frames_raw, image_size_from_file = load_labels(args.labels)
        image_size = tuple(args.image_size) if args.image_size else image_size_from_file
        if image_size is None:
            print("ERROR: image_size missing; pass --image-size W H.", file=sys.stderr)
            return 2
        truth = None

    frame_views_all = build_frame_views(frames_raw, gates_3d, min_gates=args.min_gates_per_frame)
    print(f"[load] usable frames: {len(frame_views_all)} of {len(frames_raw)} input "
          f"(>= {args.min_gates_per_frame} gates each)")
    if len(frame_views_all) < 12:
        print("ERROR: not enough usable frames.", file=sys.stderr)
        return 3

    selected = select_frames_by_coverage(frame_views_all, image_size, n_target=args.n_frames)
    print(f"[select] {len(selected)} frames chosen by coverage greedy")

    result, frames_used = calibrate_iterative(
        selected, image_size, args.hfov_rad,
        max_outer_iter=args.max_outer_iter, outlier_k=args.outlier_k,
        fix_k4=args.fix_k4, loss=args.ba_loss, f_scale=args.ba_fscale,
        max_nfev=args.ba_max_nfev, verbose=verbose,
    )

    print(f"\n[final] RMS={result.rms:.4f} px on {len(result.used_frame_ids)} frames")
    print(f"  K = \n{result.K}")
    print(f"  D = {result.D.tolist()}")

    save_calibration(result, args.output_dir)
    save_diagnostics(frames_used, result, args.output_dir)
    print(f"[saved] {args.output_dir}/calibration.{{json,yaml}} + diagnostic PNGs")

    if truth is not None:
        K_true = np.asarray(truth["K"]); D_true = np.asarray(truth["D"])
        f_err = abs(result.K[0, 0] - K_true[0, 0]) / K_true[0, 0] * 100
        cx_err = abs(result.K[0, 2] - K_true[0, 2])
        cy_err = abs(result.K[1, 2] - K_true[1, 2])
        print("\n=== Demo recovery vs. ground truth ===")
        print(f"  f relative error : {f_err:.3f} %  (true f = {K_true[0,0]:.2f})")
        print(f"  cx error (px)    : {cx_err:.2f}   (true cx = {K_true[0,2]:.2f})")
        print(f"  cy error (px)    : {cy_err:.2f}   (true cy = {K_true[1,2]:.2f})")
        print(f"  D true     : {D_true.tolist()}")
        print(f"  D recovered: {result.D.tolist()}")
        ok = (f_err < 0.5) and (cx_err < 3.0) and (cy_err < 3.0) and (result.rms < 1.0)
        print("  RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
