"""JSON / flattened CSV / PNG export of analysis records (REQ-10, spec Section 17.3)."""

from __future__ import annotations

import csv
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any, max_array: int = 64) -> Any:
    """Recursively convert NumPy/Path/Enum values. Keys starting with '_' (internal arrays) are dropped.

    Arrays larger than ``max_array`` elements are summarised rather than dumped.
    NaN becomes null so the JSON stays standard.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_array) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, max_array) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size <= max_array:
            return to_jsonable(obj.tolist(), max_array)
        return {"array_shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Nested dict -> single-level dict with dotted keys; lists become JSON strings."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def config_id(cfg: dict) -> str:
    """Short stable hash of the configuration (excluding machine-specific paths)."""
    clean = {k: v for k, v in cfg.items() if k not in ("root", "config_path")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]


def export_result(
    result: dict,
    output_dir: str | Path,
    stem: str | None = None,
    figures: dict[str, Any] | None = None,
    dpi: int = 130,
) -> dict[str, str]:
    """Write ``<stem>.json``, ``<stem>.csv`` (flattened) and one PNG per Matplotlib figure."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or str(result.get("image_id") or "result")
    paths: dict[str, str] = {}
    for name, fig in (figures or {}).items():
        p = out / f"{stem}_{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths[f"figure_{name}"] = p.as_posix()
    record = to_jsonable(result)
    record.setdefault("artifacts", {}).update({k: Path(v).name for k, v in paths.items()})
    jp = out / f"{stem}.json"
    jp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    paths["json"] = jp.as_posix()
    flat = flatten({k: v for k, v in record.items() if k not in ("texture_features",)})
    flat.update({f"texture.{k}": v for k, v in record.get("texture_features", {}).items()})
    cp = out / f"{stem}.csv"
    with cp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat))
        w.writeheader()
        w.writerow(flat)
    paths["csv"] = cp.as_posix()
    return paths


def write_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as indented JSON (NumPy values converted), creating parent folders."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(obj, max_array=10_000), indent=2), encoding="utf-8")
    return p
