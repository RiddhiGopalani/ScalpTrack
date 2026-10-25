"""Colour conversion and spatial filtering (Labs 1, 2 and 4).

Numerical conventions:
* Filters operate on a copy of the same original grayscale ROI.
* Signed results (Laplacian, sharpened image) stay ``float64``; only display copies are clipped.
* One boundary policy is used everywhere: ``cv2.BORDER_REFLECT_101`` (configurable).
"""

from __future__ import annotations

import cv2
import numpy as np

BORDERS = {
    "reflect101": cv2.BORDER_REFLECT_101,
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
}


def border_flag(config: dict | None) -> int:
    name = (config or {}).get("filters", {}).get("border", "reflect101")
    if name not in BORDERS:
        raise ValueError(f"Unknown border policy '{name}'; choose from {sorted(BORDERS)}")
    return BORDERS[name]


# --------------------------------------------------------------------------- colour


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> grayscale uint8 using OpenCV's BT.601 weights (0.299R + 0.587G + 0.114B)."""
    if rgb.ndim == 2:
        return rgb.copy()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxW or HxWx3 image, got shape {rgb.shape}")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> HSV uint8. OpenCV encodes H in [0, 179] (degrees / 2); S and V in [0, 255]."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def to_lab_uint8(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> OpenCV 8-bit LAB: L*255/100, a+128, b+128 (NOT textbook signed a*/b*)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def to_lab_float(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> CIE L*a*b* float32: L in [0, 100], a*/b* signed (approx. [-128, 127])."""
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def lab_uint8_to_signed(lab_u8: np.ndarray) -> np.ndarray:
    """Convert OpenCV 8-bit LAB to physical units so it is comparable with ``to_lab_float``."""
    lab = lab_u8.astype(np.float32)
    out = np.empty_like(lab)
    out[..., 0] = lab[..., 0] * 100.0 / 255.0
    out[..., 1] = lab[..., 1] - 128.0
    out[..., 2] = lab[..., 2] - 128.0
    return out


def hue_degrees(hsv_u8: np.ndarray) -> np.ndarray:
    """OpenCV uint8 hue (0..179) -> degrees (0..358)."""
    return hsv_u8[..., 0].astype(np.float32) * 2.0


def channel_stack(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """All individual channels used in notebook 01, in their natural numeric units."""
    hsv = to_hsv(rgb)
    lab = to_lab_float(rgb)
    return {
        "R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2],
        "Gray": to_gray(rgb),
        "H (deg)": hue_degrees(hsv), "S": hsv[..., 1], "V": hsv[..., 2],
        "L* (0-100)": lab[..., 0], "a* (signed)": lab[..., 1], "b* (signed)": lab[..., 2],
    }


def circular_hue_stats(hue_deg: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    """Circular mean / resultant length of hue (hue wraps at 360, so a plain mean is wrong)."""
    rad = np.deg2rad(hue_deg.astype(np.float64).ravel())
    wts = np.ones_like(rad) if weights is None else weights.astype(np.float64).ravel()
    if wts.sum() == 0:
        return {"hue_mean_deg": float("nan"), "hue_resultant": 0.0}
    c, s = np.sum(wts * np.cos(rad)) / wts.sum(), np.sum(wts * np.sin(rad)) / wts.sum()
    return {"hue_mean_deg": float(np.rad2deg(np.arctan2(s, c)) % 360.0), "hue_resultant": float(np.hypot(c, s))}


def patch_statistics(channel: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Mean, std and quantiles of one channel (optionally inside a boolean mask)."""
    vals = channel.astype(np.float64)
    vals = vals[mask.astype(bool)] if mask is not None else vals.ravel()
    if vals.size == 0:
        return {k: float("nan") for k in ("mean", "std", "p05", "p25", "p50", "p75", "p95")}
    p = np.percentile(vals, [5, 25, 50, 75, 95])
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "p05": float(p[0]), "p25": float(p[1]), "p50": float(p[2]), "p75": float(p[3]), "p95": float(p[4])}


# --------------------------------------------------------------------------- filters


def mean_filter(gray: np.ndarray, ksize: int, border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    _check_odd(ksize, "mean kernel")
    return cv2.blur(gray, (ksize, ksize), borderType=border)


def gaussian_filter(gray: np.ndarray, ksize: int, sigma: float, border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    _check_odd(ksize, "Gaussian kernel")
    return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=border)


def median_filter(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Median filter (extra comparison; OpenCV's medianBlur uses replicate borders internally)."""
    _check_odd(ksize, "median kernel")
    return cv2.medianBlur(gray, ksize)


def bilateral_filter(gray: np.ndarray, d: int, sigma_color: float, sigma_space: float,
                     border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    """Edge-preserving bilateral filter (extra comparison beyond the Lab 2 requirement)."""
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space, borderType=border)


def signed_laplacian(gray: np.ndarray, ksize: int = 1, border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    """Float64 Laplacian with OpenCV's negative-centre kernel ([0,1,0],[1,-4,1],[0,1,0] for ksize=1)."""
    return cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F, ksize=ksize, borderType=border)


def laplacian_sharpen(gray: np.ndarray, alpha: float, ksize: int = 1,
                      border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    """``I_sharp = I - alpha * Laplacian(I)`` in float64 (unclipped).

    With OpenCV's negative-centre convention, subtracting the Laplacian boosts
    the centre pixel relative to its neighbours, i.e. sharpens.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return gray.astype(np.float64) - alpha * signed_laplacian(gray, ksize, border)


def to_display_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip a float result to [0, 255] uint8 for display/export only."""
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def signed_to_display(arr: np.ndarray) -> np.ndarray:
    """Map a signed array symmetrically around 128 so zero stays mid-gray (display only)."""
    m = float(np.max(np.abs(arr))) or 1.0
    return np.clip(128.0 + 127.0 * arr / m, 0, 255).astype(np.uint8)


def compare_filters(gray_roi: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    """Mean, Gaussian, median, bilateral and Laplacian-sharpened outputs of one grayscale ROI.

    Smoothing outputs are uint8; ``laplacian`` and ``sharpened`` are float64.
    """
    if gray_roi.ndim != 2:
        raise ValueError("compare_filters expects a single-channel grayscale ROI")
    f = config["filters"]
    b = border_flag(config)
    out: dict[str, np.ndarray] = {"original": gray_roi.copy()}
    for k in f["mean_kernels"]:
        out[f"mean_{k}x{k}"] = mean_filter(gray_roi, k, b)
    for k in f["gaussian_kernels"]:
        out[f"gaussian_{k}x{k}_s{f['gaussian_sigma']}"] = gaussian_filter(gray_roi, k, f["gaussian_sigma"], b)
    for k in f.get("median_kernels", []):
        out[f"median_{k}x{k}"] = median_filter(gray_roi, k)
    if f.get("bilateral"):
        bl = f["bilateral"]
        out["bilateral"] = bilateral_filter(gray_roi, bl["d"], bl["sigma_color"], bl["sigma_space"], b)
    out["laplacian"] = signed_laplacian(gray_roi, f["laplacian_ksize"], b)
    out[f"sharpened_a{f['laplacian_alpha']}"] = laplacian_sharpen(gray_roi, f["laplacian_alpha"], f["laplacian_ksize"], b)
    return out


def add_gaussian_noise(gray: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Deterministic additive Gaussian noise for the synthetic denoising experiment."""
    rng = np.random.default_rng(seed)
    noisy = gray.astype(np.float64) + rng.normal(0.0, sigma, gray.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    if reference.shape != test.shape:
        raise ValueError(f"Shape mismatch {reference.shape} vs {test.shape}")
    d = reference.astype(np.float64) - test.astype(np.float64)
    return float(np.mean(d * d))


def psnr(reference: np.ndarray, test: np.ndarray, peak: float = 255.0) -> float:
    """PSNR in dB. Only meaningful against a known clean reference (synthetic fixture)."""
    m = mse(reference, test)
    return float("inf") if m == 0 else float(10.0 * np.log10(peak * peak / m))


def _check_odd(k: int, name: str) -> None:
    if k < 1 or k % 2 == 0:
        raise ValueError(f"{name} size must be a positive odd integer, got {k}")
