"""Statistical texture: GLCM descriptors and histogram entropy (Lab 6, spec Section 9.1).

The GLCM is implemented directly with NumPy so a validity mask can be honoured:
only pixel *pairs* whose two members are both valid are counted. Filling the
outside of a mask with zeros (and then calling a whole-array GLCM) would create
artificial texture. Without a mask the result matches
``skimage.feature.graycomatrix(..., symmetric=True, normed=True)`` (checked in tests).
"""

from __future__ import annotations

import numpy as np

GLCM_PROPS = ("contrast", "asm", "energy", "glcm_entropy_bits", "homogeneity", "correlation")


def quantize(gray: np.ndarray, levels: int = 32) -> np.ndarray:
    """Map uint8 intensities 0..255 to ``levels`` equal-width bins (0..levels-1)."""
    if gray.dtype != np.uint8:
        raise ValueError("quantize expects uint8 input")
    if not 2 <= levels <= 256:
        raise ValueError("levels must be in [2, 256]")
    return (gray.astype(np.uint16) * levels // 256).astype(np.uint8)


def offset_for(distance: int, angle_deg: float) -> tuple[int, int]:
    """(dy, dx) for a distance/angle, matching scikit-image's convention."""
    a = np.deg2rad(angle_deg)
    return int(round(np.sin(a) * distance)), int(round(np.cos(a) * distance))


def glcm(
    q: np.ndarray,
    distance: int,
    angle_deg: float,
    levels: int,
    mask: np.ndarray | None = None,
    symmetric: bool = True,
) -> tuple[np.ndarray, int]:
    """Normalised GLCM for one offset; returns ``(P, n_valid_pairs)``. ``P`` is all-zero if no pairs."""
    dy, dx = offset_for(distance, angle_deg)
    h, w = q.shape
    # Pair (r, c) with (r + dy, c + dx); slice both arrays to the overlapping region.
    r0, r1 = max(0, -dy), min(h, h - dy)
    c0, c1 = max(0, -dx), min(w, w - dx)
    if r1 <= r0 or c1 <= c0:
        return np.zeros((levels, levels)), 0
    a = q[r0:r1, c0:c1]
    b = q[r0 + dy : r1 + dy, c0 + dx : c1 + dx]
    if mask is not None:
        m = mask.astype(bool)
        valid = m[r0:r1, c0:c1] & m[r0 + dy : r1 + dy, c0 + dx : c1 + dx]
        a, b = a[valid], b[valid]
    else:
        a, b = a.ravel(), b.ravel()
    counts = np.bincount(a.astype(np.int64) * levels + b.astype(np.int64), minlength=levels * levels)
    P = counts.reshape(levels, levels).astype(np.float64)
    if symmetric:
        P = P + P.T
    n_pairs = int(a.size)
    total = P.sum()
    return (P / total if total > 0 else P), n_pairs


def glcm_properties(P: np.ndarray) -> dict[str, float]:
    """Contrast, ASM, energy, GLCM entropy (bits), homogeneity and correlation of a normalised GLCM."""
    if P.sum() == 0:
        return {k: float("nan") for k in GLCM_PROPS}
    L = P.shape[0]
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    asm = float(np.sum(P * P))
    pos = P[P > 0]
    mu_i, mu_j = float(np.sum(i * P)), float(np.sum(j * P))
    sd_i = float(np.sqrt(np.sum(P * (i - mu_i) ** 2)))
    sd_j = float(np.sqrt(np.sum(P * (j - mu_j) ** 2)))
    corr = float(np.sum(P * (i - mu_i) * (j - mu_j)) / (sd_i * sd_j)) if sd_i > 0 and sd_j > 0 else 1.0
    return {
        "contrast": float(np.sum(P * (i - j) ** 2)),
        "asm": asm,
        "energy": float(np.sqrt(asm)),
        "glcm_entropy_bits": float(-np.sum(pos * np.log2(pos))),
        "homogeneity": float(np.sum(P / (1.0 + (i - j) ** 2))),
        "correlation": corr,
    }


def glcm_features(
    gray: np.ndarray,
    levels: int = 32,
    distances=(1, 2, 4),
    angles_deg=(0, 45, 90, 135),
    mask: np.ndarray | None = None,
) -> dict:
    """GLCM properties for every distance/angle, plus mean/std aggregation across offsets.

    Returns ``{"per_offset": [...], "aggregate": {"glcm_contrast_mean": ..., ...}}``.
    """
    q = quantize(gray, levels)
    rows = []
    for d in distances:
        for a in angles_deg:
            P, n = glcm(q, d, a, levels, mask)
            props = glcm_properties(P)
            rows.append({"distance": d, "angle_deg": a, "n_pairs": n, **props})
    agg: dict[str, float] = {}
    for k in GLCM_PROPS:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        name = k if k.startswith("glcm_") else f"glcm_{k}"
        agg[f"{name}_mean"] = float(vals.mean()) if vals.size else float("nan")
        agg[f"{name}_std"] = float(vals.std()) if vals.size else float("nan")
    return {
        "per_offset": rows,
        "aggregate": agg,
        "settings": {
            "levels": levels,
            "distances": list(distances),
            "angles_deg": list(angles_deg),
            "symmetric": True,
            "masked": mask is not None,
        },
    }


def histogram_entropy(gray: np.ndarray, bins: int = 256, mask: np.ndarray | None = None) -> float:
    """Shannon entropy (bits) of the *intensity histogram* - a first-order statistic.

    Distinct from GLCM entropy, which describes the distribution of *pairs* of
    intensities at an offset and therefore depends on spatial arrangement.
    """
    vals = gray[mask.astype(bool)] if mask is not None else gray.ravel()
    if vals.size == 0:
        return float("nan")
    hist = np.bincount(vals.astype(np.int64), minlength=256).astype(np.float64)
    if bins != 256:
        hist = np.add.reduceat(hist, np.linspace(0, 256, bins + 1, dtype=int)[:-1])
    p = hist[hist > 0] / vals.size
    return float(-np.sum(p * np.log2(p)))


def interior_patches(
    mask: np.ndarray, patch: int, stride: int | None = None
) -> list[tuple[int, int, int, int]]:
    """Fully-valid square patches inside ``mask`` as ``(x0, y0, x1, y1)``: an alternative to masked GLCMs."""
    stride = stride or patch
    m = mask.astype(bool)
    h, w = m.shape
    ii = np.pad(np.cumsum(np.cumsum(m, 0), 1), ((1, 0), (1, 0)))
    out = []
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            s = ii[y + patch, x + patch] - ii[y, x + patch] - ii[y + patch, x] + ii[y, x]
            if s == patch * patch:
                out.append((x, y, x + patch, y + patch))
    return out
