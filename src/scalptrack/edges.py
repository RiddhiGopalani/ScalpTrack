"""Gradient and edge detection (Lab 3, spec Section 8.4).

Edge maps locate intensity transitions. They do not identify follicles and do
not count hairs: a single shaft produces *two* Canny boundaries.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .preprocessing import border_flag

PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
PREWITT_Y = PREWITT_X.T.copy()


def sobel_gradients(gray: np.ndarray, ksize: int = 3, border: int = cv2.BORDER_REFLECT_101) -> dict[str, np.ndarray]:
    """Signed float64 Sobel derivatives, magnitude and orientation (radians, [-pi, pi])."""
    g = gray.astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=ksize, borderType=border)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=ksize, borderType=border)
    return {"gx": gx, "gy": gy, "magnitude": np.hypot(gx, gy), "orientation": np.arctan2(gy, gx)}


def prewitt_gradients(gray: np.ndarray, border: int = cv2.BORDER_REFLECT_101) -> dict[str, np.ndarray]:
    """Prewitt derivatives via explicit 3x3 kernels and ``cv2.filter2D``.

    ``filter2D`` computes correlation, so the kernels above give +response for
    intensity increasing to the right (x) and downwards (y).
    """
    g = gray.astype(np.float64)
    gx = cv2.filter2D(g, cv2.CV_64F, PREWITT_X, borderType=border)
    gy = cv2.filter2D(g, cv2.CV_64F, PREWITT_Y, borderType=border)
    return {"gx": gx, "gy": gy, "magnitude": np.hypot(gx, gy), "orientation": np.arctan2(gy, gx)}


def threshold_magnitude(magnitude: np.ndarray, fraction: float) -> np.ndarray:
    """Binary edge map at ``fraction * max(magnitude)`` - the documented Sobel/Prewitt threshold."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    peak = float(magnitude.max())
    if peak == 0:
        return np.zeros(magnitude.shape, dtype=bool)
    return magnitude >= fraction * peak


def canny(gray: np.ndarray, low: float, high: float, blur_ksize: int = 5, blur_sigma: float = 1.2) -> np.ndarray:
    """Gaussian pre-smoothing followed by Canny; returns a boolean edge map."""
    if low >= high:
        raise ValueError(f"Canny low threshold ({low}) must be below high ({high})")
    smoothed = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
    return cv2.Canny(smoothed, low, high, L2gradient=True) > 0


def edge_summary(edge_map: np.ndarray) -> dict[str, float]:
    """Edge pixel fraction and fragmentation (number / mean size of 8-connected edge fragments)."""
    e = edge_map.astype(bool)
    labels, n = ndimage.label(e, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:] if n else np.array([])
    return {
        "edge_fraction": float(e.mean()),
        "n_fragments": int(n),
        "mean_fragment_px": float(sizes.mean()) if n else 0.0,
        "frac_fragments_le_5px": float((sizes <= 5).mean()) if n else 0.0,
    }


def orientation_histogram(gx: np.ndarray, gy: np.ndarray, bins: int = 4, weight_threshold: float = 0.0) -> np.ndarray:
    """Magnitude-weighted histogram of *undirected* gradient orientation in [0, 180) degrees.

    Normalised to sum to 1 (all zeros when there is no gradient energy).
    """
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    keep = mag > weight_threshold
    hist, _ = np.histogram(ang[keep], bins=bins, range=(0.0, 180.0), weights=mag[keep])
    total = hist.sum()
    return hist / total if total > 0 else hist.astype(np.float64)


def orientation_coherence(gx: np.ndarray, gy: np.ndarray) -> float:
    """Global structure-tensor coherence in [0, 1]: 1 = one dominant direction, 0 = isotropic."""
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    tr = jxx + jyy
    if tr == 0:
        return 0.0
    return float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / tr)


def compare_edges(gray_roi: np.ndarray, config: dict) -> dict:
    """Sobel, Prewitt and several Canny maps on the same ROI, plus comparison summaries."""
    e = config["edges"]
    b = border_flag(config)
    sob = sobel_gradients(gray_roi, e["sobel_ksize"], b)
    pre = prewitt_gradients(gray_roi, b)
    maps: dict[str, np.ndarray] = {
        "sobel_binary": threshold_magnitude(sob["magnitude"], e["gradient_threshold_fraction"]),
        "prewitt_binary": threshold_magnitude(pre["magnitude"], e["gradient_threshold_fraction"]),
    }
    for low, high in e["canny_pairs"]:
        maps[f"canny_{low}_{high}"] = canny(gray_roi, low, high, e["canny_blur_ksize"], e["canny_blur_sigma"])
    return {
        "sobel": sob,
        "prewitt": pre,
        "binary_maps": maps,
        "summary": {name: edge_summary(m) for name, m in maps.items()},
    }
