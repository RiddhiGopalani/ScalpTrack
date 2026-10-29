"""Inference layer for the web app: a thin wrapper around the existing ScalpTrack pipeline.

Nothing here invents results. The condition prediction comes from the saved v2 model
(frozen MobileNetV2 embeddings + linear head, ``scripts/train_deep.py``); hair
measurements come from ``scalptrack.pipeline.analyze_image``.

Every photo goes through these checks, in order:
1. Out-of-distribution: the photo must resemble the training images (cosine similarity of its
   embedding to the nearest training image, threshold calibrated on validation).
2. Healthy: if the model's probability for the "healthy" class reaches ``healthy_threshold``, the
   scalp is reported as healthy-looking.
3. Match level: each condition gets its own match score (``deep.MatchScorer``; scores do not add up
   to 100%). A condition is reported if its match reaches the match level, chosen on validation
   to keep recall high (the true condition is reported for >= 95% of validation photos).
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scalptrack import deep  # noqa: E402
from scalptrack.config import load_config, project_path  # noqa: E402
from scalptrack.pipeline import analyze_image  # noqa: E402
from scalptrack.segmentation import overlay_mask  # noqa: E402

MAX_SIDE_PX = 1024  # uploads are downscaled to roughly the training resolution (720 x 472)
MIN_SIDE_PX = 128


class UploadError(ValueError):
    """The uploaded file is not a usable image."""


@dataclass
class Resources:
    cfg: dict
    bundle: dict | None
    threshold: float | None  # match level: minimum per-condition match score to report a condition
    ood_threshold: float | None  # nearest-train cosine similarity threshold (validation-calibrated)
    test_metrics: dict | None
    report: dict = field(default_factory=dict)
    model_error: str | None = None
    backbone: deep.Backbone | None = None
    healthy_threshold: float | None = None  # None when the model was trained without a healthy class


def load_resources() -> Resources:
    """Load the configuration, the v2 model bundle, its thresholds and the frozen backbone."""
    cfg = load_config()
    source = cfg["datasets"]["primary"]
    path = project_path(cfg, "models", source, "deep_bundle.joblib")
    if not path.is_file():
        return Resources(cfg, None, None, None, None, model_error=(
            f"No trained model at {path}. Run: python scripts/embed_dataset.py, then python scripts/train_deep.py"))
    bundle = joblib.load(path)
    if "match_scorer" not in bundle:
        return Resources(cfg, None, None, None, None, model_error=(
            f"The model at {path} is outdated (no per-condition match scores). Re-run: python scripts/train_deep.py"))
    report_path = project_path(cfg, "results", "model_v2", "metrics.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    return Resources(
        cfg=cfg,
        bundle=bundle,
        threshold=float(bundle["match_level"]),
        healthy_threshold=float(bundle["healthy_threshold"]) if bundle.get("healthy_label") else None,
        ood_threshold=float(bundle["ood_similarity_threshold"]),
        test_metrics=report.get("after_embedding_head_v2_split"),
        report=report,
        backbone=deep.Backbone(),
    )


def sample_images(res: Resources) -> dict[str, Path]:
    """One image per condition from the v2 TEST split (never used for training or model selection)."""
    cfg = res.cfg
    source = cfg["datasets"]["primary"]
    splits = project_path(cfg, "manifests", f"{source}_splits_v2.csv")
    base = Path(cfg["root"]) / cfg["datasets"]["sources"][source]["dir"]
    if not splits.is_file():
        return {}
    m = pd.read_csv(splits)
    m = m[m["split_v2"] == "test"].sort_values("relative_path").groupby("canonical_label").head(1)
    labels = {r.canonical_label: r.relative_path for r in m.itertuples()}
    return {k.replace("_", " ").title(): base / v for k, v in sorted(labels.items()) if (base / v).is_file()}


def decode_upload(data: bytes) -> np.ndarray:
    """Decode an uploaded file to an EXIF-corrected RGB array, downscaled to ``MAX_SIDE_PX``."""
    if not data:
        raise UploadError("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise UploadError("This file could not be read as an image. Please upload a JPG or PNG photo.") from exc
    if min(im.size) < MIN_SIDE_PX:
        raise UploadError(f"The image is too small ({im.size[0]}×{im.size[1]} px). Please use a photo at least {MIN_SIDE_PX} px on each side.")
    im.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.Resampling.LANCZOS)
    arr = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
    if float(arr.std()) < 4.0:
        raise UploadError("The image is almost uniform (blank or solid colour). Please upload a photo of your scalp.")
    return arr


def classify(rgb: np.ndarray, res: Resources) -> dict:
    """Embedding -> OOD gate -> healthy check -> independent per-condition match scores -> match level."""
    b = res.bundle
    scorer: deep.MatchScorer = b["match_scorer"]
    emb = res.backbone.embed([rgb])
    similarity = float(scorer.nearest_similarity(emb)[0])
    scores = dict(sorted(zip(scorer.match_classes, map(float, scorer(emb)[0])), key=lambda kv: -kv[1]))
    top_label, top = next(iter(scores.items()))
    p_healthy = float(scorer.healthy_probability(emb)[0])
    if similarity < res.ood_threshold:
        verdict = "ood"
    elif res.healthy_threshold is not None and p_healthy >= res.healthy_threshold:
        verdict = "healthy"
    elif top < res.threshold:
        verdict = "no_match"
    else:
        verdict = "match"
    return {"status": "ok", "predicted_label": top_label, "class_scores": scores, "top_score": top,
            "p_healthy": p_healthy, "similarity": similarity, "verdict": verdict, "model_id": b["model_id"]}


def analyze(rgb: np.ndarray, res: Resources) -> dict:
    """Run the classifier and the existing measurement pipeline on one image."""
    result = analyze_image(rgb, None, None, res.cfg, provenance="real", image_id="upload")
    if result.get("status") == "failed":
        raise UploadError(result.get("error", "Analysis failed."))
    cls = classify(rgb, res)
    result["classification"] = {**result["classification"], **cls, "task_mode": "image_classification",
                                "reason": "v2 model: frozen MobileNetV2 embeddings + linear head, per-condition match scores"}
    return {"result": result, "verdict": cls["verdict"], "top_score": cls["top_score"], "p_healthy": cls["p_healthy"],
            "similarity": cls["similarity"], "overlay": _overlay(result)}


def _overlay(result: dict) -> np.ndarray:
    """Hair candidates (rose), flake candidates (cyan) and the counting line (gold) on the analysed image."""
    arrays = result["_arrays"]
    seg = arrays["segmentation"]
    img = overlay_mask(arrays["roi_rgb"], seg["hair"], (236, 72, 120), 0.45)
    img = overlay_mask(img, seg["flake"], (34, 211, 238), 0.75)
    line = arrays["count"].get("line_xyxy")
    if line is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in line)
        thickness = max(2, min(img.shape[:2]) // 250)
        cv2.line(img, (x0, y0), (x1, y1), (250, 204, 21), thickness, cv2.LINE_AA)
    return img
