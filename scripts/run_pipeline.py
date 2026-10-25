"""Run the integrated ScalpTrack analyzer on one image and export JSON / CSV / PNG.

Usage (from the project root):
    python scripts/run_pipeline.py --image path\\to\\image.jpg
    python scripts/run_pipeline.py --image img.jpg --roi 100 80 420 360 --model models\\kaggle_hair_diseases\\baseline_bundle.joblib
    python scripts/run_pipeline.py --synthetic-demo        # generated close-up, clearly labelled

Without --model the script uses models/<primary source>/baseline_bundle.joblib if it exists,
otherwise classification is reported as unavailable. ROI is x0 y0 x1 y1 in original pixels (max exclusive).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from scalptrack import classification as clf
from scalptrack import synthetic as syn
from scalptrack.config import load_config, project_path
from scalptrack.exports import export_result
from scalptrack.pipeline import analyze_image, capability_badges
from scalptrack.viz import result_panel, synthetic_banner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="input image path")
    src.add_argument(
        "--synthetic-demo", action="store_true", help="analyze a generated, labelled synthetic close-up"
    )
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--model", type=Path, help="model bundle (.joblib)")
    ap.add_argument("--pixel-scale-um", type=float, help="documented um/px scale (requires --scale-source)")
    ap.add_argument("--scale-source", help="provenance of the pixel scale")
    ap.add_argument("--out", type=Path, help="output folder (default outputs/results/pipeline)")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    model_path = args.model or project_path(
        cfg, "models", cfg["datasets"]["primary"], "baseline_bundle.joblib"
    )
    try:
        bundle = clf.load_bundle(model_path)
        print(f"[pipeline] model: {bundle['model_id']} (provenance: {bundle['data_provenance']})")
    except clf.ModelMissingError as exc:
        bundle = None
        print(f"[pipeline] {exc}")

    if args.synthetic_demo:
        mask = syn.curved_shafts(7, 4.0, size=200, seed=cfg["seed"])["mask"]
        source, provenance, image_id = (
            syn.render_hair_image(mask, seed=cfg["seed"]),
            "synthetic",
            "synthetic_closeup",
        )
    else:
        source, provenance, image_id = args.image, "real", None

    r = analyze_image(
        source,
        tuple(args.roi) if args.roi else None,
        bundle,
        cfg,
        provenance=provenance,
        image_id=image_id,
        pixel_scale_um_per_px=args.pixel_scale_um,
        scale_source=args.scale_source,
    )
    if r.get("status") == "failed":
        print(f"[pipeline] FAILED: {r['error']}", file=sys.stderr)
        return 1
    out = args.out or project_path(
        cfg, "results", "pipeline" if provenance == "real" else "pipeline_synthetic_demo"
    )
    badges = capability_badges(r)
    fig = result_panel(r, badges, f"ScalpTrack analysis - {r['image_id']}")
    if provenance == "synthetic":
        synthetic_banner(fig)
    paths = export_result(
        r, out, r["image_id"] if provenance == "real" else f"SYNTHETIC_{r['image_id']}", {"panel": fig}
    )
    for b in badges:
        print(f"  [{b['status']:>14s}] {b['capability']}: {b['reason']}")
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
