"""Gaussian and Laplacian pyramids with exact reconstruction (Lab 7, spec Section 9.2).

Residuals are float64 and expansion uses explicit destination sizes, so odd
dimensions reconstruct without the off-by-one cropping errors that appear when
``cv2.pyrUp`` is called without ``dstsize``.
"""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_pyramid(image: np.ndarray, max_levels: int = 4, min_side_px: int = 16) -> list[np.ndarray]:
    """Level 0 (float64 copy of the input) plus up to ``max_levels`` reductions.

    Stops before any level would have a side shorter than ``min_side_px``.
    """
    levels = [image.astype(np.float64)]
    for _ in range(max_levels):
        h, w = levels[-1].shape[:2]
        nh, nw = (h + 1) // 2, (w + 1) // 2
        if min(nh, nw) < min_side_px:
            break
        levels.append(cv2.pyrDown(levels[-1], dstsize=(nw, nh)))
    return levels


def expand(level: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Upsample one pyramid level to an explicit (height, width) - needed for odd sizes."""
    return cv2.pyrUp(level, dstsize=(size_hw[1], size_hw[0]))


def laplacian_pyramid(gauss: list[np.ndarray]) -> list[np.ndarray]:
    """Signed residuals ``G_k - expand(G_{k+1})``; the last element is the coarsest Gaussian level."""
    lap = []
    for k in range(len(gauss) - 1):
        lap.append(gauss[k] - expand(gauss[k + 1], gauss[k].shape[:2]))
    lap.append(gauss[-1].copy())
    return lap


def reconstruct(lap: list[np.ndarray]) -> np.ndarray:
    """Rebuild level 0 from the coarsest level by repeated expansion plus residual addition."""
    img = lap[-1].copy()
    for residual in reversed(lap[:-1]):
        img = expand(img, residual.shape[:2]) + residual
    return img


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Max absolute error and MSE between the original and the reconstruction."""
    d = original.astype(np.float64) - reconstructed
    return {"max_abs_error": float(np.max(np.abs(d))), "mse": float(np.mean(d * d))}


def band_energies(lap: list[np.ndarray]) -> list[float]:
    """Mean squared residual per Laplacian level (excludes the coarsest low-pass level)."""
    return [float(np.mean(r * r)) for r in lap[:-1]]


def pyramid_table(gauss: list[np.ndarray], lap: list[np.ndarray]) -> list[dict]:
    """One row per level: size, Gaussian std and Laplacian residual energy."""
    rows = []
    for k, g in enumerate(gauss):
        row = {"level": k, "height": g.shape[0], "width": g.shape[1], "gauss_std": float(g.std())}
        if k < len(lap) - 1:
            row["laplacian_energy"] = float(np.mean(lap[k] ** 2))
            row["laplacian_abs_max"] = float(np.max(np.abs(lap[k])))
        rows.append(row)
    return rows
