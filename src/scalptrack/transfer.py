"""Optional P1 path: frozen MobileNetV2 ImageNet embeddings (spec Section 14.3).

TensorFlow is NOT part of the base environment (see requirements-training.txt).
Every function checks availability first and reports ``unavailable`` instead of
failing the rest of the project. The cached embeddings feed the same
LogisticRegression selection used for the classical baseline, on the same split.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from .schema import Status

INPUT_SIZE = 224


def tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def status() -> dict:
    if tensorflow_available():
        return {"status": str(Status.OK), "reason": "TensorFlow importable; ImageNet weights download on first use"}
    return {"status": str(Status.UNAVAILABLE),
            "reason": "TensorFlow not installed (optional). `pip install -r requirements-training.txt` to enable."}


def build_frozen_backbone():
    """MobileNetV2 without its top, global-average pooled, weights frozen."""
    if not tensorflow_available():
        raise RuntimeError(status()["reason"])
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(input_shape=(INPUT_SIZE, INPUT_SIZE, 3), include_top=False,
                                             weights="imagenet", pooling="avg")
    base.trainable = False
    return base


def embed(rgb_images: list[np.ndarray], backbone=None, batch_size: int = 32) -> np.ndarray:
    """1280-D embeddings for RGB uint8 images (resized to 224x224, MobileNetV2 preprocessing)."""
    import cv2
    import tensorflow as tf

    backbone = backbone or build_frozen_backbone()
    out = []
    for s in range(0, len(rgb_images), batch_size):
        batch = np.stack([cv2.resize(im, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
                          for im in rgb_images[s:s + batch_size]]).astype(np.float32)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        out.append(backbone.predict(batch, verbose=0))
    return np.concatenate(out) if out else np.zeros((0, 1280), np.float32)
