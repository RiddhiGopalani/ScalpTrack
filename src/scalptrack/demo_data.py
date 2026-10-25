"""Consistent choice of notebook demonstration inputs.

* Real development images come only from the TRAIN/VALIDATION splits of the audited
  manifest, never from TEST (spec Section 4.4).
* When the dataset is absent, notebooks use labelled synthetic close-ups instead
  and say so. Synthetic and real figures are saved to different folders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import synthetic as syn
from .audit import load_manifest, missing_dataset_message
from .config import project_path
from .io_utils import crop_roi, list_images, load_image


def dataset_status(cfg: dict, source: str | None = None) -> dict:
    """Is the source downloaded? Is it audited? Returns flags plus a human-readable message."""
    source = source or cfg["datasets"]["primary"]
    scfg = cfg["datasets"]["sources"][source]
    root = Path(cfg["root"])
    n = len(list_images(root / scfg["dir"], cfg["image"]["supported_extensions"]))
    mp = project_path(cfg, "manifests", f"{source}_manifest.csv")
    if n == 0:
        msg = "REAL DATASET NOT AVAILABLE - synthetic fixtures are used below.\n" + missing_dataset_message(
            source, scfg, root
        )
    elif not mp.is_file():
        msg = f"{n} images found for '{source}' but no manifest yet: run notebook 00 (or scripts/audit_dataset.py)."
    else:
        shown = mp.relative_to(root) if mp.is_relative_to(root) else mp
        msg = f"Real dataset '{source}': {n} image files, manifest at {shown}."
    return {
        "source": source,
        "n_files": n,
        "downloaded": n > 0,
        "audited": mp.is_file(),
        "manifest_path": mp,
        "available": n > 0 and mp.is_file(),
        "message": msg,
    }


def development_images(
    cfg: dict,
    n_per_class: int = 1,
    max_total: int = 6,
    source: str | None = None,
    splits=("train", "validation"),
) -> list[dict]:
    """A deterministic sample of real development images (train/validation only)."""
    st = dataset_status(cfg, source)
    if not st["available"]:
        return []
    man = load_manifest(st["manifest_path"])
    man = man[man["split"].isin(splits)]
    pick = man.groupby("canonical_label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), n_per_class), random_state=cfg["seed"])
    )
    base = Path(cfg["root"]) / cfg["datasets"]["sources"][st["source"]]["dir"]
    out = []
    for r in pick.head(max_total).itertuples(index=False):
        rec = load_image(base / r.relative_path)
        rgb = crop_roi(rec.rgb, tuple(r.bbox_xyxy)) if isinstance(r.bbox_xyxy, list) else rec.rgb
        out.append(
            {
                "rgb": rgb,
                "image_id": r.image_id,
                "label": r.raw_label,
                "split": r.split,
                "path": r.relative_path,
                "provenance": "real",
            }
        )
    return out


def synthetic_closeups(cfg: dict, size: int = 200) -> dict[str, dict]:
    """Labelled synthetic close-ups with known masks, including the spec's failure cases."""
    seed = cfg["seed"]
    base_mask = syn.curved_shafts(7, 4.0, amplitude=7, size=size, seed=seed)["mask"]
    clean = syn.render_hair_image(base_mask, seed=seed)
    flaky, flake_truth = syn.add_flakes(clean, n=18, seed=seed)
    out = {
        "clean": {"rgb": clean, "hair_truth": base_mask},
        "flakes": {"rgb": flaky, "hair_truth": base_mask, "flake_truth": flake_truth},
        "shadow": {"rgb": syn.add_shadow(clean, 0.8), "hair_truth": base_mask},
        "glare": {"rgb": syn.add_glare(clean, radius=9, peak=238), "hair_truth": base_mask},
        "light_hair": {
            "rgb": syn.render_hair_image(base_mask, contrast=0.25, seed=seed),
            "hair_truth": base_mask,
        },
        "blurred": {
            "rgb": syn.render_hair_image(base_mask, blur_sigma=3.5, seed=seed),
            "hair_truth": base_mask,
        },
    }
    for k, v in out.items():
        v.update(image_id=f"SYNTHETIC_{k}", provenance="synthetic", label=f"synthetic {k}")
    return out


def figure_path(cfg: dict, notebook: str, name: str, provenance: str) -> Path:
    """outputs/figures/<notebook>/<real|synthetic>/<name>.png (synthetic names carry a SYNTHETIC_ prefix)."""
    sub = "synthetic" if provenance == "synthetic" else "real"
    fname = f"SYNTHETIC_{name}.png" if provenance == "synthetic" else f"{name}.png"
    p = project_path(cfg, "figures", notebook, sub, fname)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def mask_precision_recall(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Pixel precision/recall/F1 against a KNOWN synthetic truth mask (not meaningful without truth)."""
    p, t = pred.astype(bool), truth.astype(bool)
    tp = int((p & t).sum())
    prec = tp / int(p.sum()) if p.any() else None
    rec = tp / int(t.sum()) if t.any() else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"precision": prec, "recall": rec, "f1": f1}
