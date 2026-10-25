"""Image loading, validation and ROI handling (Lab 1, REQ-01).

Conventions (spec Section 6.1):
* OpenCV decodes to BGR. ``load_image`` always returns RGB.
* The returned array is the immutable original after EXIF orientation correction.
* ROIs are ``(x0, y0, x1, y1)`` in original-image pixels; the max coordinates are exclusive.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class ImageLoadError(ValueError):
    """Raised when a file is missing, unsupported, corrupt or otherwise unusable."""


class ROIError(ValueError):
    """Raised when an ROI is malformed or falls outside the image."""


@dataclass
class ImageRecord:
    """Decoded RGB image plus the metadata needed for provenance."""

    rgb: np.ndarray
    path: Path
    width_px: int
    height_px: int
    channels_in_file: int
    mode_in_file: str
    format: str | None
    file_hash: str
    exif_transposed: bool
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rgb.shape


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the raw file bytes (exact-duplicate detection and run provenance)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """SHA-256 of an array's bytes plus its shape/dtype (for in-memory inputs)."""
    h = hashlib.sha256()
    h.update(str((arr.shape, str(arr.dtype))).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def is_supported(path: str | Path, extensions: Iterable[str] = DEFAULT_EXTENSIONS) -> bool:
    return Path(path).suffix.lower() in {e.lower() for e in extensions}


def load_image(
    path: str | Path,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    min_side_px: int = 1,
) -> ImageRecord:
    """Validate and decode an image file into an RGB ``uint8`` array plus metadata.

    Pillow is used for decoding so EXIF orientation can be applied; the result is
    converted to 3-channel RGB: grayscale is replicated and an alpha channel is
    discarded (recorded in ``notes``).
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"File does not exist: {path}")
    if not path.is_file():
        raise ImageLoadError(f"Not a file: {path}")
    if not is_supported(path, extensions):
        raise ImageLoadError(f"Unsupported extension '{path.suffix}' for {path.name}")
    if path.stat().st_size == 0:
        raise ImageLoadError(f"Empty file: {path}")

    notes: list[str] = []
    try:
        with Image.open(path) as im:
            im.load()  # force full decode so truncated files fail here
            fmt = im.format
            mode = im.mode
            exif_changed = _exif_orientation(im) not in (None, 1)
            transposed = ImageOps.exif_transpose(im)
            bands = len(im.getbands())
            if mode in ("RGBA", "LA", "PA") or "A" in im.getbands():
                notes.append("alpha channel discarded")
            if mode in ("I;16", "I", "F"):
                notes.append(f"high bit-depth mode {mode} rescaled to uint8")
                arr = np.asarray(transposed, dtype=np.float64)
                lo, hi = float(arr.min()), float(arr.max())
                arr = np.zeros_like(arr) if hi == lo else (arr - lo) / (hi - lo) * 255.0
                rgb = np.repeat(arr.astype(np.uint8)[..., None], 3, axis=2)
            else:
                rgb = np.asarray(transposed.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageLoadError(f"Corrupt or unreadable image {path.name}: {exc}") from exc

    h, w = rgb.shape[:2]
    if min(h, w) < min_side_px:
        raise ImageLoadError(f"Image {path.name} is {w}x{h}px, below the minimum side {min_side_px}px")

    return ImageRecord(
        rgb=np.ascontiguousarray(rgb),
        path=path,
        width_px=w,
        height_px=h,
        channels_in_file=bands,
        mode_in_file=mode,
        format=fmt,
        file_hash=file_sha256(path),
        exif_transposed=bool(exif_changed),
        notes=notes,
    )


def _exif_orientation(im: Image.Image) -> int | None:
    try:
        return im.getexif().get(0x0112)
    except Exception:  # noqa: BLE001 - malformed EXIF is non-fatal; orientation is then unknown
        return None


def save_rgb(path: str | Path, rgb: np.ndarray) -> Path:
    """Write an RGB uint8 image with OpenCV (converting to BGR)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb.ndim == 2:
        ok = cv2.imwrite(str(path), rgb)
    else:
        ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"cv2.imwrite failed for {path}")
    return path


def validate_roi(image_shape: tuple[int, ...], roi: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    """Bound-check a native-pixel ROI ``(x0, y0, x1, y1)``; ``None`` means the full image."""
    h, w = int(image_shape[0]), int(image_shape[1])
    if roi is None:
        return (0, 0, w, h)
    if len(roi) != 4:
        raise ROIError(f"ROI must have 4 values (x0, y0, x1, y1), got {roi!r}")
    try:
        x0, y0, x1, y1 = (int(v) for v in roi)
    except (TypeError, ValueError) as exc:
        raise ROIError(f"ROI values must be integers: {roi!r}") from exc
    if any(int(v) != v for v in roi):
        raise ROIError(f"ROI values must be whole pixels: {roi!r}")
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        raise ROIError(f"ROI {roi} lies outside the {w}x{h} image")
    if x1 <= x0 or y1 <= y0:
        raise ROIError(f"ROI {roi} is empty (max coordinates are exclusive)")
    return (x0, y0, x1, y1)


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    """Return a copy of the ROI so later processing cannot modify the original."""
    x0, y0, x1, y1 = validate_roi(image.shape, roi)
    return image[y0:y1, x0:x1].copy()


def resize_with_record(image: np.ndarray, size_wh: tuple[int, int], interpolation: int = cv2.INTER_AREA) -> tuple[np.ndarray, dict]:
    """Resize and return the transform record needed to map measurements back to original pixels."""
    h, w = image.shape[:2]
    out = cv2.resize(image, size_wh, interpolation=interpolation)
    sx, sy = size_wh[0] / w, size_wh[1] / h
    return out, {
        "op": "resize",
        "from_wh": [w, h],
        "to_wh": list(size_wh),
        "scale_x": sx,
        "scale_y": sy,
        "isotropic": bool(np.isclose(sx, sy, rtol=1e-3)),
    }


def compare_resize_methods(image: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    """Nearest / bilinear / area resizing of the same image (Lab 1 comparison)."""
    h, w = image.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return {
        "nearest": cv2.resize(image, size, interpolation=cv2.INTER_NEAREST),
        "bilinear": cv2.resize(image, size, interpolation=cv2.INTER_LINEAR),
        "area": cv2.resize(image, size, interpolation=cv2.INTER_AREA),
    }


def pixel_neighbourhood(image: np.ndarray, x: int, y: int, radius: int = 2) -> np.ndarray:
    """Raw pixel values in a ``(2r+1)`` square around ``(x, y)``, clipped at borders."""
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ROIError(f"Pixel ({x}, {y}) outside {w}x{h} image")
    return image[max(0, y - radius): y + radius + 1, max(0, x - radius): x + radius + 1].copy()


def describe_array(arr: np.ndarray) -> dict:
    """Shape, dtype and dynamic range of an array (used in the Lab 1 table)."""
    return {
        "shape": tuple(int(s) for s in arr.shape),
        "dtype": str(arr.dtype),
        "channels": 1 if arr.ndim == 2 else int(arr.shape[2]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def list_images(folder: str | Path, extensions: Iterable[str] = DEFAULT_EXTENSIONS) -> list[Path]:
    """Recursively list supported image files, sorted for reproducibility."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    exts = {e.lower() for e in extensions}
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)
