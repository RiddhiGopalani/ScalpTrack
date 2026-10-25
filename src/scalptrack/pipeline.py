"""Integrated single-image analyzer (Lab 10, REQ-10/11, spec Sections 6 and 16-17).

``analyze_image`` never invents missing inputs: no model -> classification
unavailable; no stereo pair -> stereo unavailable; no scale -> micrometres null;
unreadable file or invalid ROI -> a ``failed`` record instead of a crash.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import edges, face_roi, measurements, segmentation
from .classification import package_versions, predict_condition
from .exports import config_id
from .features import extract_features
from .io_utils import ImageLoadError, ROIError, array_sha256, crop_roi, load_image, validate_roi
from .preprocessing import to_gray
from .quality import assess_quality
from .schema import Status, empty_result
from .stereo import compute_stereo


@contextmanager
def _timed(timings: dict[str, float], step: str):
    """Record the wall-clock seconds of the enclosed block under ``timings[step]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[step] = time.perf_counter() - t0


def _load_input(
    source: str | Path | np.ndarray, image_id: str | None, config: dict, result: dict
) -> np.ndarray:
    """Decode a path or accept an RGB array; fill the provenance fields of ``result``.

    Raises ``ImageLoadError`` for unreadable, unsupported or malformed input.
    """
    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
            raise ImageLoadError(f"Array input must be HxWx3 uint8 RGB, got {source.shape} {source.dtype}")
        result["input_hash"] = array_sha256(source)
        result["source"] = "in_memory_array"
        result["image_id"] = image_id or f"array_{result['input_hash'][:10]}"
        return source

    rec = load_image(source, config["image"]["supported_extensions"], config["image"]["min_side_px"])
    result["input_hash"] = rec.file_hash
    result["source"] = Path(source).as_posix()
    result["image_id"] = image_id or Path(source).stem
    result["image_meta"] = {
        "width_px": rec.width_px,
        "height_px": rec.height_px,
        "format": rec.format,
        "mode_in_file": rec.mode_in_file,
        "exif_transposed": rec.exif_transposed,
        "notes": rec.notes,
    }
    return rec.rgb


def analyze_image(
    source: str | Path | np.ndarray,
    roi=None,
    model_bundle: dict | None = None,
    config: dict | None = None,
    provenance: str = "real",
    image_id: str | None = None,
    run_faces: bool = True,
    stereo_pair: dict | None = None,
    pixel_scale_um_per_px: float | None = None,
    scale_source: str | None = None,
    cascade=None,
) -> dict:
    """Assemble the analysis record for one image (path or RGB array) and optional native-pixel ROI.

    Steps: 1 load + validate -> 2 quality -> 3 faces -> 4 segmentation/edges -> 5 features
    -> 6 classification -> 7 count/width -> 8 stereo (only if a pair is given).
    Internal arrays for plotting live under ``result['_arrays']`` and are not exported.
    """
    if config is None:
        raise ValueError("config is required (scalptrack.config.load_config())")
    t_all = time.perf_counter()
    timings: dict[str, float] = {}
    result = empty_result(image_id)
    result["run_config_id"] = config_id(config)
    result["provenance"] = provenance
    result["run"] = {
        "seed": config["seed"],
        "versions": package_versions(),
        "model_id": (model_bundle or {}).get("model_id"),
    }

    # 1. Acquisition + validation. Bad input gives a "failed" record, never an exception.
    try:
        with _timed(timings, "load_validate"):
            rgb = _load_input(source, image_id, config, result)
            roi_xyxy = validate_roi(rgb.shape, roi)
    except (ImageLoadError, ROIError) as exc:
        result["status"] = str(Status.FAILED)
        result["error"] = str(exc)
        result["timings_s"] = {"total": time.perf_counter() - t_all}
        return result
    result["roi_xyxy"] = list(roi_xyxy)
    roi_rgb = crop_roi(rgb, roi_xyxy)
    gray = to_gray(roi_rgb)

    # 2. Quality flags (they can disable measurement, never classification).
    with _timed(timings, "quality"):
        quality = assess_quality(roi_rgb, config)
        result["quality"] = {k: v for k, v in quality.items() if k != "thresholds"}

    # 3. Optional face detection on the full image (privacy: boxes only, no face crops exported).
    if run_faces:
        with _timed(timings, "faces"):
            try:
                cascade = cascade or face_roi.load_cascade_from_config(config)
                result["faces"] = face_roi.detect_faces(rgb, cascade, config)
            except face_roi.CascadeLoadError as exc:
                result["faces"] = {"boxes_xywh": [], "status": str(Status.FAILED), "reason": str(exc)}

    # 4. Candidate masks, coverage and edges.
    with _timed(timings, "segmentation_edges"):
        seg = segmentation.segment_candidates(roi_rgb, config)
        result["segmentation"] = {
            "method": seg["params"]["hair"],
            "flake": seg["params"]["flake"],
            "status": str(Status.SYNTHETIC_DEMO if provenance == "synthetic" else Status.EXPLORATORY),
            "reason": "Heuristic threshold candidates; no ground-truth masks",
        }
        result["coverage"] = segmentation.coverage_descriptors(
            seg["hair"], seg["scalp"], seg["flake"], valid=~seg["unknown"]
        )
        e = config["edges"]
        low, high = e["canny_pairs"][len(e["canny_pairs"]) // 2]  # the middle threshold pair
        edge_map = edges.canny(gray, low, high, e["canny_blur_ksize"], e["canny_blur_sigma"])
        result["edges"] = {"canny_thresholds": [low, high], **edges.edge_summary(edge_map)}

    # 5. Texture/colour features (computed on a resized copy, never used for measurement).
    with _timed(timings, "features"):
        result["texture_features"] = extract_features(roi_rgb, config)

    # 6. Classification with the saved model, if one was supplied.
    with _timed(timings, "classification"):
        result["classification"] = predict_condition(roi_rgb, model_bundle, config)
        if model_bundle is None:
            result["classification"]["reason"] = (
                "No trained model bundle found; train with notebook 05 / scripts/train_classifier.py"
            )

    # 7. Hair count and width on the native ROI (quality gated).
    with _timed(timings, "measurements"):
        count = measurements.estimate_visible_hairs(roi_rgb, seg["hair"], config, provenance, quality=quality)
        result["hair_count"] = count["crossings"]
        result["hair_components"] = count["components"]
        result["hair_count_flags"] = count["flags"]
        width = measurements.estimate_shaft_width(
            seg["hair"],
            config,
            provenance,
            quality=quality,
            pixel_scale_um_per_px=pixel_scale_um_per_px,
            scale_source=scale_source,
        )
        result["shaft_width"] = {k: v for k, v in width.items() if not k.startswith("_")}

    # 8. Stereo only when a pair is explicitly supplied: a single image never becomes 3-D.
    if stereo_pair is not None:
        with _timed(timings, "stereo"):
            result["stereo"] = _stereo_summary(stereo_pair, config)

    timings["total"] = time.perf_counter() - t_all
    result["timings_s"] = timings
    result["status"] = str(Status.OK)
    result["_arrays"] = {
        "rgb": rgb,
        "roi_rgb": roi_rgb,
        "gray": gray,
        "segmentation": seg,
        "edges": edge_map,
        "count": count,
        "width": width,
    }
    return result


def _stereo_summary(stereo_pair: dict, config: dict) -> dict:
    """Run ``compute_stereo`` on an explicit pair and keep the fields of the result schema."""
    meta = stereo_pair.get("meta") or {}
    calibration = stereo_pair.get("calibration") or {}
    st = compute_stereo(
        stereo_pair.get("left"),
        stereo_pair.get("right"),
        stereo_pair.get("calibration"),
        config,
        pair_meta=meta,
    )
    return {
        "pair_id": meta.get("pair_id"),
        "calibration_id": calibration.get("calibration_id"),
        "disparity_unit": "px",
        "depth_unit": st.get("depth_unit"),
        "valid_coverage": st.get("valid_fraction"),
        "status": st["status"],
        "reason": st["reason"],
    }


def capability_badges(result: dict) -> list[dict]:
    """Short available/unavailable badges with reasons for the result panel."""

    def badge(name, status, reason):
        """One capability row for the result panel."""
        return {"capability": name, "status": status, "reason": reason}

    out = [
        badge(
            "Quality",
            "ok" if result["quality"].get("measurement_eligible") else "warning",
            ", ".join(result["quality"].get("flags", [])) or "no flags",
        ),
        badge(
            "Classification", result["classification"]["status"], result["classification"].get("reason", "")
        ),
        badge("Line crossings", result["hair_count"]["status"], result["hair_count"]["reason"][:90]),
        badge("Width (px)", result["shaft_width"]["status"], result["shaft_width"].get("reason", "")[:90]),
        badge(
            "Width (um)",
            result["shaft_width"].get("physical_status", "unavailable"),
            result["shaft_width"].get("physical_reason", "")[:90],
        ),
        badge("Stereo / 3-D", result["stereo"]["status"], result["stereo"]["reason"][:90]),
        badge("Haar face", result["faces"]["status"], result["faces"].get("reason", "")[:90]),
    ]
    return out
