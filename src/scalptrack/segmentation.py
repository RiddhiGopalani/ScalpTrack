"""Threshold segmentation, conservative cleanup and coverage descriptors (Lab 5, Sections 8.5-8.6).

All masks are *heuristic candidates*. Without ground-truth masks they cannot be
scored, and agreement between two threshold methods is not accuracy.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import to_gray, to_hsv

# --------------------------------------------------------------------------- thresholds


def _ttype(inverse: bool) -> int:
    return cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY


def threshold_fixed(gray: np.ndarray, t: float, inverse: bool = True) -> np.ndarray:
    """Fixed global threshold. ``inverse=True`` selects pixels *darker* than ``t`` (hair candidates)."""
    _check_gray(gray)
    _, m = cv2.threshold(gray, t, 255, _ttype(inverse))
    return m > 0


def threshold_otsu(gray: np.ndarray, inverse: bool = True) -> tuple[np.ndarray, float]:
    """Otsu global threshold; returns the mask and the selected threshold."""
    _check_gray(gray)
    t, m = cv2.threshold(gray, 0, 255, _ttype(inverse) | cv2.THRESH_OTSU)
    return m > 0, float(t)


def threshold_adaptive(
    gray: np.ndarray, method: str, block_size: int, c: float, inverse: bool = True
) -> np.ndarray:
    """Adaptive mean or Gaussian threshold (``block_size`` odd and > 1)."""
    _check_gray(gray)
    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError(f"Adaptive block size must be odd and > 1, got {block_size}")
    if block_size > min(gray.shape):
        raise ValueError(f"Block size {block_size} exceeds the ROI's smaller side {min(gray.shape)}")
    flag = {"mean": cv2.ADAPTIVE_THRESH_MEAN_C, "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C}[method]
    m = cv2.adaptiveThreshold(gray, 255, flag, _ttype(inverse), block_size, c)
    return m > 0


def compare_thresholds(gray: np.ndarray, config: dict, inverse: bool = True) -> dict:
    """The four mandatory methods (fixed, Otsu, adaptive mean, adaptive Gaussian) with parameters.

    Adaptive block sizes that do not fit inside the ROI are skipped and listed.
    """
    s = config["segmentation"]
    masks: dict[str, np.ndarray] = {}
    params: dict[str, dict] = {}
    masks["fixed"] = threshold_fixed(gray, s["fixed_threshold"], inverse)
    params["fixed"] = {"threshold": s["fixed_threshold"], "inverse": inverse}
    masks["otsu"], t = threshold_otsu(gray, inverse)
    params["otsu"] = {"threshold": t, "inverse": inverse}
    skipped = []
    for method in ("mean", "gaussian"):
        for bs in s["adaptive_block_sizes"]:
            if bs > min(gray.shape):
                skipped.append(bs)
                continue
            name = f"adaptive_{method}_{bs}"
            masks[name] = threshold_adaptive(gray, method, bs, s["adaptive_c"], inverse)
            params[name] = {"block_size": bs, "C": s["adaptive_c"], "inverse": inverse}
    return {
        "masks": masks,
        "params": params,
        "skipped_block_sizes": sorted(set(skipped)),
        "foreground_fraction": {k: float(v.mean()) for k, v in masks.items()},
    }


def adaptive_sweep(
    gray: np.ndarray, block_sizes, c_values, method: str = "gaussian", inverse: bool = True
) -> list[dict]:
    """Foreground fraction for each (block size, C) pair - a development-only sensitivity sweep."""
    rows = []
    for bs in block_sizes:
        if bs > min(gray.shape):
            continue
        for c in c_values:
            m = threshold_adaptive(gray, method, bs, c, inverse)
            rows.append({"method": method, "block_size": bs, "C": c, "foreground_fraction": float(m.mean())})
    return rows


# --------------------------------------------------------------------------- cleanup


def remove_small_components(mask: np.ndarray, min_area: int, connectivity: int = 8) -> np.ndarray:
    """Drop connected components with fewer than ``min_area`` pixels."""
    m = mask.astype(np.uint8)
    if min_area <= 1 or not m.any():
        return mask.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=connectivity)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def filter_components_by_area(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """Keep only connected components whose area lies in ``[min_area, max_area]``."""
    m = mask.astype(np.uint8)
    if not m.any():
        return mask.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = (areas >= min_area) & (areas <= max_area)
    keep[0] = False
    return keep[labels]


def cleanup_mask(
    mask: np.ndarray, open_kernel: int = 0, close_kernel: int = 3, min_area: int = 30
) -> np.ndarray:
    """Conservative cleanup: optional opening, small closing, then small-component removal.

    Opening is off by default because a 3x3 opening deletes 1-2 px wide shafts.
    A small closing bridges single-pixel gaps but slightly thickens edges; notebook
    02 shows the before/after and notebook 04 measures the effect on width.
    """
    m = mask.astype(np.uint8)
    if open_kernel and open_kernel > 1:
        m = cv2.morphologyEx(
            m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        )
    if close_kernel and close_kernel > 1:
        m = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        )
    return remove_small_components(m > 0, min_area)


# --------------------------------------------------------------------------- candidates


def unknown_mask(rgb: np.ndarray, config: dict) -> np.ndarray:
    """Pixels excluded from analysis: deep shadow and saturated colourless glare."""
    s = config["segmentation"]
    hsv = to_hsv(rgb)
    shadow = hsv[..., 2] <= s["shadow_v_max"]
    glare = (hsv[..., 2] >= s["glare_v_min"]) & (hsv[..., 1] <= s["glare_s_max"])
    return shadow | glare


def hair_candidate_mask(gray: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Dark thin structures via the configured threshold; returns (raw, cleaned, params)."""
    s = config["segmentation"]
    method = s["hair_method"]
    bs = min(s["hair_block_size"], _largest_odd_below(min(gray.shape)))
    if method == "otsu":
        raw, t = threshold_otsu(gray, inverse=True)
        params = {"method": "otsu", "threshold": t}
    elif method == "fixed":
        raw = threshold_fixed(gray, s["fixed_threshold"], inverse=True)
        params = {"method": "fixed", "threshold": s["fixed_threshold"]}
    elif method in ("adaptive_mean", "adaptive_gaussian"):
        if bs < 3:
            raise ValueError("ROI too small for adaptive thresholding")
        raw = threshold_adaptive(gray, method.split("_")[1], bs, s["adaptive_c"], inverse=True)
        params = {"method": method, "block_size": bs, "C": s["adaptive_c"]}
    else:
        raise ValueError(f"Unknown hair_method '{method}'")
    clean = cleanup_mask(raw, s["open_kernel"], s["close_kernel"], s["min_component_area"])
    params.update(
        {
            "open_kernel": s["open_kernel"],
            "close_kernel": s["close_kernel"],
            "min_component_area": s["min_component_area"],
            "inverse": True,
        }
    )
    return raw, clean, params


def flake_candidate_mask(
    rgb: np.ndarray, config: dict, exclude: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """Bright, locally prominent, low-saturation blobs with size limits.

    Bright glare and pale skin pass this test too; that failure is shown in notebook 02.
    """
    s = config["segmentation"]
    gray = to_gray(rgb).astype(np.float64)
    hsv = to_hsv(rgb)
    local = cv2.GaussianBlur(gray, (0, 0), 7)
    level = float(np.percentile(gray, s["flake_bright_percentile"]))
    bright = (gray >= level) & (gray - local > 10.0) & (hsv[..., 1] <= s["flake_max_saturation"])
    if exclude is not None:
        bright &= ~exclude
    mask = filter_components_by_area(bright, s["flake_min_area"], s["flake_max_area"])
    return mask, {
        "percentile": s["flake_bright_percentile"],
        "intensity_level": level,
        "local_contrast_min": 10.0,
        "max_saturation": s["flake_max_saturation"],
        "area_range": [s["flake_min_area"], s["flake_max_area"]],
    }


def segment_candidates(rgb_roi: np.ndarray, config: dict, valid_mask: np.ndarray | None = None) -> dict:
    """Disjoint hair / visible-scalp / flake / unknown candidate masks for one RGB ROI.

    Priority: unknown (shadow, glare, outside ``valid_mask``) > hair > flake > scalp.
    """
    if rgb_roi.ndim != 3 or rgb_roi.shape[2] != 3:
        raise ValueError("segment_candidates expects an HxWx3 RGB ROI")
    gray = to_gray(rgb_roi)
    unknown = unknown_mask(rgb_roi, config)
    if valid_mask is not None:
        if valid_mask.shape != gray.shape:
            raise ValueError("valid_mask must match the ROI shape")
        unknown |= ~valid_mask.astype(bool)
    hair_raw, hair_clean, hair_params = hair_candidate_mask(gray, config)
    hair = hair_clean & ~unknown
    flake, flake_params = flake_candidate_mask(rgb_roi, config, exclude=hair | unknown)
    scalp = ~(hair | flake | unknown)
    return {
        "hair_raw": hair_raw,
        "hair": hair,
        "flake": flake,
        "scalp": scalp,
        "unknown": unknown,
        "params": {
            "hair": hair_params,
            "flake": flake_params,
            "unknown": {
                "shadow_v_max": config["segmentation"]["shadow_v_max"],
                "glare_v_min": config["segmentation"]["glare_v_min"],
                "glare_s_max": config["segmentation"]["glare_s_max"],
            },
        },
    }


def coverage_descriptors(
    hair: np.ndarray, scalp: np.ndarray, flake: np.ndarray | None = None, valid: np.ndarray | None = None
) -> dict:
    """Pixel-area descriptors from disjoint candidate masks (Section 8.6).

    Each value is ``None`` when its denominator is empty. These are *not* hairs/cm^2
    and flake coverage is not a clinical dandruff percentage.
    """
    h, s = hair.astype(bool), scalp.astype(bool)
    if np.any(h & s):
        raise ValueError("hair and scalp masks must be disjoint")
    union = int((h | s).sum())
    nh, ns = int(h.sum()), int(s.sum())
    out = {
        "visible_scalp_fraction": ns / union if union else None,
        "hair_coverage_fraction": nh / union if union else None,
        "scalp_to_hair_area_ratio": ns / nh if nh else None,
        "flake_candidate_coverage": None,
        "excluded_fraction": None,
        "unit": "pixel-area fraction",
    }
    if flake is not None:
        v = valid.astype(bool) if valid is not None else np.ones_like(h)
        nv = int(v.sum())
        out["flake_candidate_coverage"] = int((flake.astype(bool) & v).sum()) / nv if nv else None
        out["excluded_fraction"] = 1.0 - nv / v.size if v.size else None
    return out


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend a boolean mask onto an RGB image (display only)."""
    out = rgb.astype(np.float64).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float64)
    return out.astype(np.uint8)


def label_overlay(rgb: np.ndarray, masks: dict[str, np.ndarray], colors: dict[str, tuple]) -> np.ndarray:
    """Overlay several named masks, each in its own colour (display only)."""
    out = rgb.copy()
    for name, m in masks.items():
        out = overlay_mask(out, m, colors[name], 0.55)
    return out


def _check_gray(gray: np.ndarray) -> None:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"Expected a 2-D uint8 grayscale image, got {gray.shape} {gray.dtype}")


def _largest_odd_below(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n - 1
