"""Compute (or reuse) frozen MobileNetV2 embeddings for every image in the audited manifest.

Usage (from the project root):  python scripts/embed_dataset.py
Writes data/derived/<source>_mobilenet_v2.npz with arrays ``image_id`` and ``emb`` (float16).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from scalptrack.audit import load_manifest
from scalptrack.config import load_config, project_path
from scalptrack.deep import BACKBONE, Backbone


def embeddings_path(cfg: dict, source: str) -> Path:
    return project_path(cfg, "derived", f"{source}_{BACKBONE}.npz")


def load_or_compute(cfg: dict, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (image_ids, embeddings) for all manifest images, computing only missing ones."""
    manifest = load_manifest(project_path(cfg, "manifests", f"{source}_manifest.csv"))
    manifest = manifest[manifest["readable"].fillna(False).astype(bool)]
    path = embeddings_path(cfg, source)
    cache: dict[str, np.ndarray] = {}
    if path.is_file():
        data = np.load(path, allow_pickle=False)
        cache = dict(zip(data["image_id"].tolist(), data["emb"]))
    todo = manifest[~manifest["image_id"].isin(cache)]
    if len(todo):
        base = Path(cfg["root"]) / cfg["datasets"]["sources"][source]["dir"]
        t0 = time.perf_counter()
        emb = Backbone().embed_paths([base / p for p in todo["relative_path"]])
        print(f"[embed] {len(todo)} images in {time.perf_counter() - t0:.0f}s")
        cache.update(zip(todo["image_id"].tolist(), emb.astype(np.float16)))
        ids = np.array(sorted(cache))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, image_id=ids, emb=np.stack([cache[i] for i in ids]).astype(np.float16))
    ids = manifest["image_id"].to_numpy()
    return ids, np.stack([cache[i] for i in ids]).astype(np.float32)


if __name__ == "__main__":
    cfg = load_config()
    ids, emb = load_or_compute(cfg, cfg["datasets"]["primary"])
    print(f"[embed] {len(ids)} embeddings of dim {emb.shape[1]} -> {embeddings_path(cfg, cfg['datasets']['primary'])}")
