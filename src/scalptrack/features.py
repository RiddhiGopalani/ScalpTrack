"""Fixed-schema numerical feature vector for the classical classifier (REQ-03, spec Section 14.2).

The schema is a fixed, ordered list of names grouped so the controlled
comparison (with vs without texture descriptors, spec Section 15.2) can drop
whole groups. The schema is saved with every model bundle so inference uses
the training feature order.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import edges as edg
from . import pyramids, texture
from .preprocessing import circular_hue_stats, hue_degrees, to_gray, to_hsv, to_lab_float

N_PYRAMID_BANDS = 4

FEATURE_GROUPS: dict[str, list[str]] = {
    "intensity": ["gray_mean", "gray_std", "gray_p05", "gray_p50", "gray_p95"],
    "color": ["hue_cos_mean", "hue_sin_mean", "hue_resultant", "s_mean", "s_std", "v_mean", "v_std",
              "lab_L_mean", "lab_L_std", "lab_a_mean", "lab_a_std", "lab_b_mean", "lab_b_std"],
    "edge": ["canny_edge_fraction", "sobel_mag_mean", "sobel_mag_std",
             "grad_orient_0_45", "grad_orient_45_90", "grad_orient_90_135", "grad_orient_135_180",
             "grad_orientation_coherence"],
    "texture_hist": ["hist_entropy_bits"],
    "texture_glcm": [f"{n}_{s}" for n in ("glcm_contrast", "glcm_asm", "glcm_energy", "glcm_entropy_bits",
                                           "glcm_homogeneity", "glcm_correlation") for s in ("mean", "std")],
    "multiscale": [f"laplacian_band_energy_l{k}" for k in range(N_PYRAMID_BANDS)],
}

# Groups removed in the "without texture descriptors" ablation.
TEXTURE_GROUPS = ("texture_hist", "texture_glcm", "multiscale")
ALL_GROUPS = tuple(FEATURE_GROUPS)


def feature_names(groups=ALL_GROUPS) -> list[str]:
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise KeyError(f"Unknown feature groups: {sorted(unknown)}")
    return [name for g in ALL_GROUPS if g in groups for name in FEATURE_GROUPS[g]]


def prepare_for_features(rgb: np.ndarray, size: int) -> np.ndarray:
    """Classifier-geometry copy: square ``size`` x ``size`` via INTER_AREA (never used for measurement)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB input, got {rgb.shape}")
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def extract_features(rgb_roi: np.ndarray, config: dict, resize: bool = True) -> dict[str, float]:
    """Compute every feature in the schema for an RGB ROI. Returns an ordered dict of floats."""
    t = config["texture"]
    img = prepare_for_features(rgb_roi, t["feature_size"]) if resize else rgb_roi
    gray = to_gray(img)
    hsv = to_hsv(img)
    lab = to_lab_float(img)
    f: dict[str, float] = {}

    p = np.percentile(gray, [5, 50, 95])
    f.update(gray_mean=float(gray.mean()), gray_std=float(gray.std()),
             gray_p05=float(p[0]), gray_p50=float(p[1]), gray_p95=float(p[2]))

    hue = circular_hue_stats(hue_degrees(hsv), weights=hsv[..., 1])
    rad = np.deg2rad(hue["hue_mean_deg"]) if not np.isnan(hue["hue_mean_deg"]) else 0.0
    f.update(hue_cos_mean=float(np.cos(rad) * hue["hue_resultant"]),
             hue_sin_mean=float(np.sin(rad) * hue["hue_resultant"]),
             hue_resultant=hue["hue_resultant"],
             s_mean=float(hsv[..., 1].mean()), s_std=float(hsv[..., 1].std()),
             v_mean=float(hsv[..., 2].mean()), v_std=float(hsv[..., 2].std()),
             lab_L_mean=float(lab[..., 0].mean()), lab_L_std=float(lab[..., 0].std()),
             lab_a_mean=float(lab[..., 1].mean()), lab_a_std=float(lab[..., 1].std()),
             lab_b_mean=float(lab[..., 2].mean()), lab_b_std=float(lab[..., 2].std()))

    e = config["edges"]
    low, high = e["canny_pairs"][len(e["canny_pairs"]) // 2]
    sob = edg.sobel_gradients(gray, e["sobel_ksize"])
    oh = edg.orientation_histogram(sob["gx"], sob["gy"], bins=4)
    f.update(canny_edge_fraction=float(edg.canny(gray, low, high, e["canny_blur_ksize"], e["canny_blur_sigma"]).mean()),
             sobel_mag_mean=float(sob["magnitude"].mean()), sobel_mag_std=float(sob["magnitude"].std()),
             grad_orient_0_45=float(oh[0]), grad_orient_45_90=float(oh[1]),
             grad_orient_90_135=float(oh[2]), grad_orient_135_180=float(oh[3]),
             grad_orientation_coherence=edg.orientation_coherence(sob["gx"], sob["gy"]))

    f["hist_entropy_bits"] = texture.histogram_entropy(gray)
    f.update(texture.glcm_features(gray, t["levels"], t["distances"], t["angles_deg"])["aggregate"])

    pc = config["pyramids"]
    lap = pyramids.laplacian_pyramid(pyramids.gaussian_pyramid(gray, pc["max_levels"], pc["min_side_px"]))
    energies = pyramids.band_energies(lap)
    for k in range(N_PYRAMID_BANDS):
        f[f"laplacian_band_energy_l{k}"] = energies[k] if k < len(energies) else float("nan")

    return {name: float(f[name]) for name in feature_names()}


def features_to_vector(features: dict[str, float], names: list[str]) -> np.ndarray:
    """Order a feature dict by ``names``; a missing name is an error, not a silent zero."""
    missing = [n for n in names if n not in features]
    if missing:
        raise KeyError(f"Feature dict is missing {len(missing)} schema entries, e.g. {missing[:3]}")
    return np.array([features[n] for n in names], dtype=np.float64)
