"""Pretrained-CNN embeddings (transfer learning) for condition classification and out-of-distribution checks.

A frozen ImageNet-pretrained MobileNetV2 (3.5M parameters, the model named in the project spec)
turns each image into a 1280-D, L2-normalised
embedding. A small linear head is trained on top (``scripts/train_deep.py``). Frozen features
are used deliberately: the dataset holds far fewer independent photos than images (heavy
augmentation), so fine-tuning millions of weights would mostly memorise those copies.

The same embeddings support:
* near-duplicate detection that survives rotation, colour jitter and cropping;
* out-of-distribution detection (cosine similarity to the nearest training image).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BACKBONE = "mobilenet_v2"
EMBED_DIM = 1280
INPUT_SIZE = 224
RESIZE_SIZE = 256
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Backbone:
    """Frozen MobileNetV2 feature extractor (global-average-pooled, L2-normalised)."""

    def __init__(self, threads: int | None = None):
        import torch
        import torchvision.models as models

        torch.set_num_threads(threads or os.cpu_count() or 4)
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        net.classifier = torch.nn.Identity()  # drop dropout + the 1000-class ImageNet head; keep pooled features
        self.net = net.eval()
        self.torch = torch

    @staticmethod
    def preprocess(rgb: np.ndarray) -> np.ndarray:
        """Resize the short side to 256, centre-crop 224, normalise with ImageNet statistics. Returns CHW float32."""
        im = Image.fromarray(rgb)
        w, h = im.size
        scale = RESIZE_SIZE / min(w, h)
        im = im.resize((max(INPUT_SIZE, round(w * scale)), max(INPUT_SIZE, round(h * scale))), Image.Resampling.BILINEAR)
        w, h = im.size
        left, top = (w - INPUT_SIZE) // 2, (h - INPUT_SIZE) // 2
        arr = np.asarray(im.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE)), dtype=np.float32) / 255.0
        return ((arr - _MEAN) / _STD).transpose(2, 0, 1)

    def embed(self, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Embeddings for a list of RGB uint8 arrays, shape (n, 1280), unit length."""
        out = []
        with self.torch.inference_mode():
            for s in range(0, len(images), batch_size):
                batch = np.stack([self.preprocess(im) for im in images[s : s + batch_size]])
                feats = self.net(self.torch.from_numpy(batch)).numpy()
                out.append(feats)
        emb = np.concatenate(out) if out else np.zeros((0, EMBED_DIM), np.float32)
        return normalise(emb)

    def embed_paths(self, paths: list[Path], batch_size: int = 32, progress: bool = True) -> np.ndarray:
        """Embeddings for image files (EXIF-corrected RGB), streamed in batches."""
        chunks = []
        for s in range(0, len(paths), batch_size):
            chunks.append(self.embed([load_rgb(p) for p in paths[s : s + batch_size]], batch_size))
            if progress and (s // batch_size) % 25 == 0:
                print(f"[embed] {min(s + batch_size, len(paths))}/{len(paths)}", flush=True)
        return np.concatenate(chunks) if chunks else np.zeros((0, EMBED_DIM), np.float32)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)


def normalise(x: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm (cosine similarity becomes a dot product)."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def max_similarity(query: np.ndarray, reference: np.ndarray, chunk: int = 2048, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """For each query row: the highest cosine similarity to any reference row, and that row's index."""
    best = np.full(len(query), -1.0, dtype=np.float32)
    arg = np.zeros(len(query), dtype=np.int64)
    for s in range(0, len(query), chunk):
        sim = query[s : s + chunk] @ reference.T
        if exclude_self:
            rows = np.arange(s, min(s + chunk, len(query)))
            sim[rows - s, rows] = -1.0
        arg[s : s + chunk] = sim.argmax(axis=1)
        best[s : s + chunk] = sim.max(axis=1)
    return best, arg


def similar_pairs(emb: np.ndarray, threshold: float, chunk: int = 1024) -> np.ndarray:
    """All index pairs (i < j) whose embedding cosine similarity is at least ``threshold``."""
    pairs = []
    for s in range(0, len(emb), chunk):
        sim = emb[s : s + chunk] @ emb.T
        ii, jj = np.nonzero(sim >= threshold)
        ii = ii + s
        keep = jj > ii
        if keep.any():
            pairs.append(np.stack([ii[keep], jj[keep]], axis=1))
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)


class MatchScorer:
    """Per-condition match scores that do NOT have to add up to 100%.

    Softmax scores always sum to 1, so a photo unlike every condition still puts most of its score
    on one class. Here each condition is scored on its own:

    raw_c   = log p_head(c) + weight * mean cosine of the k most similar TRAIN photos of class c
    match_c = share of VALIDATION photos of class c whose raw_c is at most this photo's raw_c
              (a conformal p-value), times (1 - p_head(healthy)) when the head has a healthy class

    A typical photo of condition c gets a match around 50% for c. A photo that is atypical for every
    condition, or that the head considers healthy, gets a low match for all of them. Only classes
    calibrated with validation photos get a match score (``match_classes``).
    """

    def __init__(self, head, train_emb: np.ndarray, train_y: np.ndarray, k: int = 5, weight: float = 20.0,
                 healthy_label: str | None = None):
        self.head, self.k, self.weight = head, k, weight
        self.classes = list(head.classes_)
        self.healthy_label = healthy_label if healthy_label in self.classes else None
        emb = normalise(train_emb).astype(np.float16)
        self.train_emb = {c: emb[np.asarray(train_y) == c] for c in self.classes}
        self.reference: dict[str, np.ndarray] = {}

    @property
    def match_classes(self) -> list[str]:
        return list(self.reference)

    def nearest_similarity(self, x: np.ndarray) -> np.ndarray:
        """Cosine similarity of each row to its nearest training photo (any class)."""
        x = normalise(x)
        return np.max([(x @ e.astype(np.float32).T).max(1) for e in self.train_emb.values()], axis=0)

    def healthy_probability(self, x: np.ndarray) -> np.ndarray:
        if self.healthy_label is None:
            return np.zeros(len(x))
        return self.head.predict_proba(normalise(x))[:, self.classes.index(self.healthy_label)]

    def raw(self, x: np.ndarray) -> np.ndarray:
        x = normalise(x)
        sim = np.stack([np.sort(x @ self.train_emb[c].astype(np.float32).T, 1)[:, -self.k:].mean(1) for c in self.classes], 1)
        return np.log(self.head.predict_proba(x) + 1e-9) + self.weight * sim

    def calibrate(self, x_val: np.ndarray, y_val: np.ndarray) -> MatchScorer:
        raw, y_val = self.raw(x_val), np.asarray(y_val)
        self.reference = {c: np.sort(raw[y_val == c, j]) for j, c in enumerate(self.classes) if (y_val == c).any()}
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Match scores, one column per ``match_classes`` entry."""
        raw = self.raw(x)
        out = np.zeros((len(x), len(self.reference)))
        for j, (c, ref) in enumerate(self.reference.items()):
            out[:, j] = np.searchsorted(ref, raw[:, self.classes.index(c)], side="right") / (len(ref) + 1)
        return out * (1.0 - self.healthy_probability(x))[:, None]


def occlude(rgb: np.ndarray, keep: str, fraction: float = 0.6) -> np.ndarray:
    """Grey out the centre (``keep='border'``) or the border (``keep='centre'``) - a shortcut-learning probe."""
    h, w = rgb.shape[:2]
    y0, y1 = int(h * (1 - fraction) / 2), int(h * (1 + fraction) / 2)
    x0, x1 = int(w * (1 - fraction) / 2), int(w * (1 + fraction) / 2)
    out = rgb.copy()
    if keep == "border":
        out[y0:y1, x0:x1] = 128
    elif keep == "centre":
        centre = out[y0:y1, x0:x1].copy()
        out[:] = 128
        out[y0:y1, x0:x1] = centre
    else:
        raise ValueError("keep must be 'border' or 'centre'")
    return out
