"""Execute the project notebooks top-to-bottom in fresh kernels and save their outputs in place.

Usage (from the project root):
    python scripts/run_notebooks.py                 # all notebooks, in order
    python scripts/run_notebooks.py 04 06           # only notebooks whose name starts with these prefixes
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    if sys.platform == "win32":  # pyzmq needs a selector loop on Windows; avoids a noisy warning
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="*")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    ap.add_argument("--output-dir", type=Path, help="write executed copies here instead of overwriting notebooks/")
    args = ap.parse_args(argv)
    os.environ["SCALPTRACK_HEADLESS"] = "1"  # inherited by kernels: skips interactive widgets
    books = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    if args.prefixes:
        books = [b for b in books if any(b.name.startswith(p) for p in args.prefixes)]
    failures = []
    for nb_path in books:
        nb = nbformat.read(nb_path, as_version=4)
        t0 = time.perf_counter()
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(nb_path.parent)}})
        try:
            client.execute()
            status = "ok"
        except CellExecutionError as exc:
            status = "FAILED"
            failures.append((nb_path.name, str(exc).splitlines()[-1] if str(exc) else repr(exc)))
        target = (args.output_dir / nb_path.name) if args.output_dir else nb_path
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, target)
        print(f"[{status}] {nb_path.name} ({time.perf_counter() - t0:.1f}s)")
    for name, msg in failures:
        print(f"  {name}: {msg}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
