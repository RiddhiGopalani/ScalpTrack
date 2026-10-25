"""Validate notebook JSON and summarise executed outputs (errors, stderr warnings, figures).

Usage:  python scripts/check_notebooks.py [--verbose]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print stderr text")
    args = ap.parse_args(argv)
    bad = 0
    for p in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        nb = nbformat.read(p, as_version=4)
        nbformat.validate(nb)
        code = [c for c in nb.cells if c.cell_type == "code"]
        executed = sum(c.get("execution_count") is not None for c in code)
        errors, stderr, images = [], [], 0
        for c in code:
            for o in c.get("outputs", []):
                if o.output_type == "error":
                    errors.append(f"{o.ename}: {o.evalue}")
                elif o.output_type == "stream" and o.name == "stderr":
                    stderr.append(o.text.strip())
                elif o.output_type in ("display_data", "execute_result") and "image/png" in o.get("data", {}):
                    images += 1
        bad += bool(errors)
        print(f"{p.name:34s} cells={len(nb.cells):3d} code={len(code):3d} executed={executed:3d} "
              f"figures={images:3d} errors={len(errors)} stderr={len(stderr)}")
        for e in errors:
            print("   ERROR", e)
        if args.verbose:
            for s in stderr:
                print("   STDERR", s[:400].replace("\n", " | "))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
