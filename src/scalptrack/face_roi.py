"""Haar-cascade face detection and ROI support (Lab 9, REQ-09, spec Section 13).

Haar detects frontal *faces*. It does not standardise head pose, detect hairs or
find the scalp in a trichoscopic close-up. A crop above a face is a heuristic
proposal that must be confirmed manually.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import validate_roi
from .schema import Status

DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"


class CascadeLoadError(RuntimeError):
    pass


def load_cascade(name: str = DEFAULT_CASCADE, search_roots=()) -> cv2.CascadeClassifier:
    """Load and verify a cascade from an explicit path, project assets, or OpenCV.

    ``name`` remains backwards compatible with the previous OpenCV-only API. A
    configured relative path is useful for portable/offline builds such as the
    Member C package, which shipped its own cascade XML.
    """
    if not hasattr(cv2, "CascadeClassifier"):
        raise CascadeLoadError("This OpenCV build has no CascadeClassifier (OpenCV 5 moved it); install opencv<5")
    requested = Path(name)
    candidates = [requested] if requested.is_absolute() else [Path(root) / requested for root in search_roots]
    if not requested.is_absolute():
        candidates.extend([Path(cv2.data.haarcascades) / requested, Path(cv2.data.haarcascades) / requested.name])
    candidates = list(dict.fromkeys(candidates))
    for path in candidates:
        if not path.is_file():
            continue
        cascade = cv2.CascadeClassifier(str(path))
        if not cascade.empty():
            return cascade
        raise CascadeLoadError(f"Cascade failed to load: {path}")
    searched = ", ".join(str(p) for p in candidates)
    raise CascadeLoadError(f"Cascade file not found; searched: {searched}")


def detect_faces(rgb_image: np.ndarray, cascade: cv2.CascadeClassifier, config: dict) -> dict:
    """Run ``detectMultiScale`` on a grayscale (optionally equalised) copy. Boxes are (x, y, w, h)."""
    f = config["face"]
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if rgb_image.ndim == 3 else rgb_image.copy()
    if f.get("equalize", True):
        gray = cv2.equalizeHist(gray)  # used for this branch only, never for measurement
    params = {"scaleFactor": float(f["scale_factor"]), "minNeighbors": int(f["min_neighbors"]),
              "minSize": tuple(int(v) for v in f["min_size"])}
    if min(gray.shape) < max(params["minSize"]):
        return {"boxes_xywh": [], "n_faces": 0, "params": _jsonable(params), "status": str(Status.UNAVAILABLE),
                "reason": f"Image smaller than the minimum face size {params['minSize']}"}
    boxes = cascade.detectMultiScale(gray, **params)
    boxes = [] if len(boxes) == 0 else [[int(v) for v in b] for b in boxes]
    return {
        "boxes_xywh": boxes,
        "n_faces": len(boxes),
        "params": _jsonable(params),
        "status": str(Status.OK),
        "reason": (f"{len(boxes)} frontal face(s) detected" if boxes else
                   "No frontal face detected (expected for scalp close-ups). Use the manual ROI path."),
    }


def propose_upper_head_roi(image_shape, face_xywh, up: float = 0.9, side: float = 0.25, down: float = 0.15) -> dict:
    """Heuristic crop above a detected face (forehead / hairline / upper head), clipped to the image.

    The proposal ALWAYS requires manual confirmation; it is not scalp segmentation.
    """
    h, w = image_shape[:2]
    x, y, fw, fh = face_xywh
    x0 = int(max(0, round(x - side * fw)))
    x1 = int(min(w, round(x + fw + side * fw)))
    y0 = int(max(0, round(y - up * fh)))
    y1 = int(min(h, round(y + down * fh)))
    if x1 <= x0 or y1 <= y0:
        return {"roi_xyxy": None, "requires_confirmation": True, "reason": "Proposal collapsed at the image border"}
    return {"roi_xyxy": [x0, y0, x1, y1], "requires_confirmation": True,
            "reason": "Heuristic region above a detected face; confirm manually before scalp analysis",
            "clipped_at_top": y0 == 0 and y - up * fh < 0}


def choose_roi(image_shape, detection: dict | None, manual_roi=None) -> dict:
    """Manual ROI wins; otherwise a face-based proposal (unconfirmed); otherwise the full image."""
    if manual_roi is not None:
        return {"roi_xyxy": list(validate_roi(image_shape, manual_roi)), "source": "manual", "confirmed": True}
    if detection and detection.get("boxes_xywh"):
        biggest = max(detection["boxes_xywh"], key=lambda b: b[2] * b[3])
        prop = propose_upper_head_roi(image_shape, biggest)
        if prop["roi_xyxy"] is not None:
            return {"roi_xyxy": prop["roi_xyxy"], "source": "haar_upper_head_proposal", "confirmed": False,
                    "reason": prop["reason"]}
    h, w = image_shape[:2]
    return {"roi_xyxy": [0, 0, w, h], "source": "full_image_default", "confirmed": False,
            "reason": "No manual ROI and no face-based proposal; whole image used (close-up assumption)"}


def blur_faces(rgb: np.ndarray, boxes_xywh, ksize: int = 51) -> np.ndarray:
    """Privacy helper: blur detected face boxes in an exported figure."""
    out = rgb.copy()
    k = ksize if ksize % 2 else ksize + 1
    for x, y, w, h in boxes_xywh:
        out[y:y + h, x:x + w] = cv2.GaussianBlur(out[y:y + h, x:x + w], (k, k), 0)
    return out


def draw_boxes(rgb: np.ndarray, boxes_xywh, color=(0, 255, 0), roi_xyxy=None) -> np.ndarray:
    out = rgb.copy()
    t = max(2, min(rgb.shape[:2]) // 200)
    for x, y, w, h in boxes_xywh:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, t)
    if roi_xyxy is not None:
        x0, y0, x1, y1 = roi_xyxy
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 200, 0), t)
    return out


def _jsonable(params: dict) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}
