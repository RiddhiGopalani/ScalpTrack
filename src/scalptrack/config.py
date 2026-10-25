"""Configuration loading and project-root discovery."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_RELATIVE = Path("configs") / "default.yaml"


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk upwards from ``start`` (default: cwd) to the folder holding ``configs/default.yaml``.

    Falls back to the location of this installed package so the package also
    works when imported from an unrelated working directory.
    """
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parent)
    for base in candidates:
        for folder in (base, *base.parents):
            if (folder / DEFAULT_CONFIG_RELATIVE).is_file() and (folder / "pyproject.toml").is_file():
                return folder
    raise FileNotFoundError(
        "Could not locate the ScalpTrack project root (a folder containing "
        "pyproject.toml and configs/default.yaml). Run from inside the repository."
    )


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the YAML configuration and attach the resolved project root as ``cfg['root']``.

    ``path`` wins; otherwise the ``SCALPTRACK_CONFIG`` environment variable; otherwise
    ``configs/default.yaml``.
    """
    root = find_project_root()
    env = os.environ.get("SCALPTRACK_CONFIG")
    cfg_path = Path(path) if path is not None else Path(env) if env else root / DEFAULT_CONFIG_RELATIVE
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    cfg["root"] = str(root)
    cfg["config_path"] = str(cfg_path)
    return cfg


def project_path(cfg: dict, key: str, *parts: str, create: bool = False) -> Path:
    """Resolve ``cfg['paths'][key]`` (relative to the root) and optional sub-parts."""
    rel = cfg["paths"][key]
    path = Path(cfg["root"]) / rel
    path = path.joinpath(*parts) if parts else path
    if create:
        (path if not path.suffix else path.parent).mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy global RNGs and return a fresh ``Generator``."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def display_path(path: str | Path, root: str | Path) -> str:
    """Path relative to the project root when inside it, otherwise the absolute path (data may live elsewhere)."""
    path, root = Path(path), Path(root)
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
