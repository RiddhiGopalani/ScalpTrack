"""Regenerate the notebooks in notebooks/ from their definitions in scripts/nbbuild/.

Usage (from the project root):
    python scripts/build_notebooks.py            # write all notebooks (outputs cleared)
    python scripts/build_notebooks.py 04 05      # only these prefixes
Then execute them with:  python scripts/run_notebooks.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULES = ["nb00", "nb01", "nb02", "nb03", "nb04", "nb05", "nb06", "nb07", "nb08"]


def main(argv=None) -> int:
    prefixes = argv if argv is not None else sys.argv[1:]
    out_dir = HERE.parent / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for name in MODULES:
        if prefixes and not any(name[2:].startswith(p[:2]) for p in prefixes):
            continue
        mod = importlib.import_module(f"nbbuild.{name}")
        nb = mod.build()
        nbformat.validate(nb)
        path = out_dir / f"{mod.TITLE}.ipynb"
        nbformat.write(nb, path)
        print(f"wrote {path.relative_to(HERE.parent)} ({len(nb.cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
