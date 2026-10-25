"""Image-quality descriptors (spec Section 8.1).

Thresholds come from the configuration and are development choices, not
validated universal cutoffs. Quality flags can disable *measurement* while
classification still runs.
"""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import to_gray


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the float64 Laplacian: an empirical sharpness indicator (higher = sharper)."""
    return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())


def assess_quality(rgb_roi: np.ndarray, config: dict) -> dict:
    """Report blur, exposure and resolution flags for an RGB ROI.

    Returns a dict with the raw descriptors, a list of flag strings and
    ``measurement_eligible`` (hair count/width should only run when True).
    """
    if rgb_roi.ndim != 3 or rgb_roi.shape[2] != 3:
        raise ValueError(f"assess_quality expects an HxWx3 RGB array, got shape {rgb_roi.shape}")
    q = config["quality"]
    gray = to_gray(rgb_roi)
    h, w = gray.shape
    hsv = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2HSV)

    lap_var = laplacian_variance(gray)
    dark_frac = float((gray <= q["dark_level"]).mean())
    bright_frac = float((gray >= q["bright_level"]).mean())
    # Specular highlight candidates: very bright and nearly colourless.
    specular_frac = float(((hsv[..., 2] >= 250) & (hsv[..., 1] <= 25)).mean())

    flags: list[str] = []
    if lap_var < q["blur_laplacian_var_min"]:
        flags.append("possible_blur")
    if dark_frac > q["dark_fraction_max"]:
        flags.append("underexposed_or_shadow")
    if bright_frac > q["bright_fraction_max"]:
        flags.append("overexposed_or_glare")
    if min(h, w) < q["min_roi_side_px"]:
        flags.append("roi_too_small")
    if specular_frac > 0.02:
        flags.append("specular_highlights")

    # Blur and tiny ROIs make shaft geometry unreliable; exposure flags are warnings.
    eligible = not ({"possible_blur", "roi_too_small"} & set(flags))
    return {
        "width_px": int(w),
        "height_px": int(h),
        "laplacian_var": lap_var,
        "dark_fraction": dark_frac,
        "bright_fraction": bright_frac,
        "specular_fraction": specular_frac,
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "flags": flags,
        "measurement_eligible": bool(eligible),
        "thresholds": dict(q),
    }
