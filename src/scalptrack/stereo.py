"""Stereo disparity and calibrated depth (Lab 8, REQ-08, spec Section 12).

Real-data gate: a single photo, two unrelated images, or a shifted copy of one
photo is NOT a stereo pair. Metric depth needs intrinsics, distortion and the
baseline in known units. Without them the result is ``unavailable``.

For an ideal rectified pair with equal principal points: ``Z = f_px * B / d_px``
(Z in the baseline's units). The general case uses the Q matrix with
``cv2.reprojectImageTo3D``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .schema import Status

REQUIRED_CALIBRATION_KEYS = ("K1", "D1", "K2", "D2", "R", "T", "units")


# --------------------------------------------------------------------------- validation


def load_calibration(path: str | Path) -> dict:
    """Load a calibration JSON: K1, D1, K2, D2 (lists), R (3x3), T (3,), units, calibration_id."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for k in ("K1", "D1", "K2", "D2", "R", "T"):
        if k in data:
            data[k] = np.asarray(data[k], dtype=np.float64)
    return data


def validate_calibration(calib: dict | None) -> list[str]:
    """Return the list of problems (empty list = usable for metric depth)."""
    if calib is None:
        return ["no calibration supplied"]
    problems = [f"missing '{k}'" for k in REQUIRED_CALIBRATION_KEYS if k not in calib or calib[k] is None]
    if problems:
        return problems
    for k in ("K1", "K2", "R"):
        if np.asarray(calib[k]).shape != (3, 3):
            problems.append(f"{k} must be 3x3")
    if np.asarray(calib["T"]).size != 3:
        problems.append("T must have 3 elements")
    elif np.linalg.norm(np.asarray(calib["T"], dtype=float)) == 0:
        problems.append("baseline |T| is zero")
    return problems


def validate_stereo_inputs(left: np.ndarray | None, right: np.ndarray | None, pair_meta: dict | None,
                           calib: dict | None) -> dict:
    """Apply the real-data gate (spec Section 12.1). Returns status and every failed check."""
    reasons: list[str] = []
    if left is None or right is None:
        reasons.append("a stereo pair needs two images; only one (or none) was supplied")
    elif left.shape[:2] != right.shape[:2]:
        reasons.append(f"left/right sizes differ: {left.shape[:2]} vs {right.shape[:2]}")
    meta = pair_meta or {}
    if not meta.get("pair_id"):
        reasons.append("no documented pair_id linking the two views")
    if not meta.get("same_static_region_confirmed"):
        reasons.append("not confirmed that both views show the same static region")
    reasons += [f"calibration: {p}" for p in validate_calibration(calib)]
    if left is not None and right is not None and left.shape == right.shape:
        if np.array_equal(left, right):
            reasons.append("left and right images are identical")
    ok = not reasons
    return {"status": str(Status.OK if ok else Status.UNAVAILABLE),
            "reason": "inputs pass the stereo gate" if ok else "; ".join(reasons), "failed_checks": reasons}


# --------------------------------------------------------------------------- geometry


def q_matrix_rectified(focal_px: float, cx: float, cy: float, baseline: float, cx_right: float | None = None) -> np.ndarray:
    """Q for an already-rectified horizontal pair (OpenCV convention, right camera at +X = baseline).

    ``reprojectImageTo3D`` gives Z = f * B / (d - (cx - cx_right)).
    """
    cx_right = cx if cx_right is None else cx_right
    return np.array([[1, 0, 0, -cx],
                     [0, 1, 0, -cy],
                     [0, 0, 0, focal_px],
                     [0, 0, 1.0 / baseline, -(cx - cx_right) / baseline]], dtype=np.float64)


def rectify_pair(left: np.ndarray, right: np.ndarray, calib: dict) -> dict:
    """Rectify a real calibrated pair with ``cv2.stereoRectify`` + remap; returns images and Q."""
    h, w = left.shape[:2]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        calib["K1"], calib["D1"], calib["K2"], calib["D2"], (w, h), calib["R"], calib["T"],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1 = cv2.initUndistortRectifyMap(calib["K1"], calib["D1"], R1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(calib["K2"], calib["D2"], R2, P2, (w, h), cv2.CV_32FC1)
    return {"left": cv2.remap(left, *m1, cv2.INTER_LINEAR), "right": cv2.remap(right, *m2, cv2.INTER_LINEAR),
            "Q": Q, "P1": P1, "P2": P2, "valid_roi_left": roi1, "valid_roi_right": roi2}


# --------------------------------------------------------------------------- matching


def make_sgbm(config: dict, min_disparity: int = 0):
    s = config["stereo"]
    nd, bs = int(s["num_disparities"]), int(s["block_size"])
    if nd <= 0 or nd % 16:
        raise ValueError(f"num_disparities must be a positive multiple of 16, got {nd}")
    if bs < 1 or bs % 2 == 0:
        raise ValueError(f"block_size must be odd, got {bs}")
    return cv2.StereoSGBM_create(
        minDisparity=min_disparity, numDisparities=nd, blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=int(s["disp12_max_diff"]),
        uniquenessRatio=int(s["uniqueness_ratio"]), speckleWindowSize=int(s["speckle_window_size"]),
        speckleRange=int(s["speckle_range"]), mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def disparity_sgbm(left_gray: np.ndarray, right_gray: np.ndarray, config: dict) -> np.ndarray:
    """Float disparity in pixels (OpenCV's fixed-point output divided by 16). Invalid -> NaN."""
    raw = make_sgbm(config).compute(left_gray, right_gray).astype(np.float32) / 16.0
    raw[raw <= 0] = np.nan  # (minDisparity - 1) marks failures; zero disparity has no finite depth
    return raw


def disparity_bm(left_gray: np.ndarray, right_gray: np.ndarray, config: dict, block_size: int = 15) -> np.ndarray:
    """StereoBM comparison (block matching without the semi-global smoothness term)."""
    bm = cv2.StereoBM_create(numDisparities=int(config["stereo"]["num_disparities"]), blockSize=block_size)
    raw = bm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    raw[raw <= 0] = np.nan
    return raw


def right_view_disparity(left_gray: np.ndarray, right_gray: np.ndarray, config: dict) -> np.ndarray:
    """Disparity referenced to the right image, via the horizontal-flip trick (no ximgproc needed)."""
    d = disparity_sgbm(np.ascontiguousarray(right_gray[:, ::-1]), np.ascontiguousarray(left_gray[:, ::-1]), config)
    return np.ascontiguousarray(d[:, ::-1])


def left_right_consistency(disp_left: np.ndarray, disp_right: np.ndarray, tol: float) -> np.ndarray:
    """True where ``|dL(x) - dR(x - dL(x))| <= tol``."""
    h, w = disp_left.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    xr = np.rint(xs - np.nan_to_num(disp_left, nan=-1e6)).astype(int)
    inb = (xr >= 0) & (xr < w) & np.isfinite(disp_left)
    dr = np.full_like(disp_left, np.nan)
    rows = np.arange(h)[:, None].repeat(w, 1)
    dr[inb] = disp_right[rows[inb], xr[inb]]
    with np.errstate(invalid="ignore"):
        return inb & np.isfinite(dr) & (np.abs(disp_left - dr) <= tol)


def compute_stereo(left: np.ndarray, right: np.ndarray, calibration: dict | None, config: dict,
                   pair_meta: dict | None = None, already_rectified: bool = False) -> dict:
    """Disparity, validity masks and (when calibrated) depth / point cloud for a stereo pair.

    Returns float arrays (NaN = invalid) kept separate from any display image.
    """
    gate = validate_stereo_inputs(left, right, pair_meta, calibration)
    s = config["stereo"]
    if left is None or right is None or left.shape[:2] != right.shape[:2]:
        return {"status": str(Status.UNAVAILABLE), "reason": gate["reason"], "gate": gate}
    calib_ok = calibration is not None and not validate_calibration(calibration)
    if calib_ok and not (already_rectified or calibration.get("rectified")):
        rect = rectify_pair(left, right, calibration)
        left, right, Q = rect["left"], rect["right"], rect["Q"]
    elif calib_ok:
        K = np.asarray(calibration["K1"])
        Q = q_matrix_rectified(K[0, 0], K[0, 2], K[1, 2], float(np.linalg.norm(calibration["T"])), np.asarray(calibration["K2"])[0, 2])
    else:
        Q = None
    lg, rg = _gray(left), _gray(right)
    disp = disparity_sgbm(lg, rg, config)
    disp_r = right_view_disparity(lg, rg, config)
    lr_ok = left_right_consistency(disp, disp_r, float(s["lr_check_tolerance_px"]))

    h, w = disp.shape
    border = np.zeros((h, w), dtype=bool)
    b = int(s["border_px"])
    border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = True
    border[:, : int(s["num_disparities"])] = True  # no full search range in the leftmost columns
    matched = np.isfinite(disp)
    valid = matched & lr_ok & ~border

    out = {
        "disparity_px": disp, "disparity_right_px": disp_r, "matched": matched, "lr_consistent": lr_ok,
        "border": border, "valid": valid, "Q": Q, "gate": gate,
        "params": {k: s[k] for k in ("num_disparities", "block_size", "uniqueness_ratio", "speckle_window_size",
                                     "speckle_range", "disp12_max_diff", "lr_check_tolerance_px", "border_px")},
        "valid_fraction": float(valid.mean()),
    }
    if Q is None:
        out.update(depth=None, points=None, depth_unit=None, status=str(Status.UNAVAILABLE),
                   reason="Disparity computed, but no calibration: metric depth is unavailable. " + gate["reason"])
        return out
    pts = cv2.reprojectImageTo3D(np.nan_to_num(disp, nan=0.0).astype(np.float32), Q)
    pts[~valid] = np.nan
    out.update(depth=pts[..., 2].astype(np.float64), points=pts, depth_unit=calibration.get("units"),
               status=str(Status.SYNTHETIC_DEMO if (pair_meta or {}).get("provenance") == "synthetic" else Status.OK),
               reason="Depth from calibrated Q matrix over valid pixels")
    return out


# --------------------------------------------------------------------------- evaluation


def evaluate_against_truth(result: dict, gt_disp: np.ndarray, gt_depth: np.ndarray, visible: np.ndarray,
                           error_threshold_px: float = 1.0) -> dict:
    """Coverage and error together over the ground-truth-visible evaluation region (excluding borders)."""
    region = visible & ~result["border"]
    valid = result["valid"] & region
    n_region = int(region.sum())
    err = np.abs(result["disparity_px"] - gt_disp)
    derr = (result["depth"] - gt_depth) if result.get("depth") is not None else None
    out = {
        "eval_region_px": n_region,
        "valid_coverage": float(valid.sum() / n_region) if n_region else None,
        "disparity_mae_px": float(np.nanmean(err[valid])) if valid.any() else None,
        f"pct_error_gt_{error_threshold_px:g}px": float(100.0 * np.mean(err[valid] > error_threshold_px)) if valid.any() else None,
        "occluded_fraction": float((~visible).mean()),
        "invalid_fraction_in_region": float(1.0 - valid.sum() / n_region) if n_region else None,
    }
    if derr is not None and valid.any():
        out["depth_mae"] = float(np.nanmean(np.abs(derr[valid])))
        out["depth_rmse"] = float(np.sqrt(np.nanmean(derr[valid] ** 2)))
    return out


def sparse_point_cloud(points: np.ndarray, valid: np.ndarray, step: int = 6, max_points: int = 6000, seed: int = 0) -> np.ndarray:
    """Subsampled Nx3 array of valid 3-D points for plotting."""
    sub = np.zeros_like(valid)
    sub[::step, ::step] = True
    p = points[valid & sub]
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) > max_points:
        p = p[np.random.default_rng(seed).choice(len(p), max_points, replace=False)]
    return p
