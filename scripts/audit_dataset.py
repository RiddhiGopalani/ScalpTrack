"""Audit a downloaded dataset source and write its manifest, split manifest, report and plots.

Usage (from the project root):
    python scripts/audit_dataset.py                       # primary source from configs/default.yaml
    python scripts/audit_dataset.py --source maheshwariss_hair_and_scalp
    python scripts/audit_dataset.py --list                # show configured sources and whether data is present

Exit code 2 means the dataset is missing; the message says exactly where to put it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scalptrack import audit
from scalptrack.config import load_config, project_path
from scalptrack.io_utils import list_images


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="source key under datasets.sources (default: datasets.primary)")
    ap.add_argument("--config", help="alternative YAML configuration")
    ap.add_argument("--workers", type=int, default=8, help="threads for image decoding")
    ap.add_argument("--list", action="store_true", help="list configured sources and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(cfg["root"])
    if args.list:
        for name, s in cfg["datasets"]["sources"].items():
            n = len(list_images(root / s["dir"]))
            print(f"{name:32s} {s['task_type']:24s} {n:6d} images  ({s['dir']})")
        return 0

    source = args.source or cfg["datasets"]["primary"]
    if source not in cfg["datasets"]["sources"]:
        print(f"Unknown source '{source}'. Configured: {list(cfg['datasets']['sources'])}", file=sys.stderr)
        return 1
    try:
        manifest, report = audit.build_manifest(cfg, source, workers=args.workers)
    except audit.DatasetMissingError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    paths = audit.save_manifest(manifest, report, project_path(cfg, "manifests"))
    fig_dir = project_path(cfg, "figures", "00_data_audit", source, create=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    usable = manifest[manifest["split"].isin(audit.SPLITS)]
    if len(usable):
        counts = usable.groupby(["canonical_label", "split"]).size().unstack(fill_value=0)
        ax = counts.plot.barh(stacked=True, figsize=(8, 0.45 * len(counts) + 1.5))
        ax.set_xlabel("images")
        ax.set_title(f"{source}: images per class and split (after audit)")
        plt.tight_layout()
        plt.savefig(fig_dir / "class_split_distribution.png", dpi=130)
        plt.close()
    print(f"[audit] manifest: {paths['manifest']}")
    print(f"[audit] report:   {paths['report']}")
    print(
        f"[audit] unreadable={report['n_unreadable']}  missing_label={report['n_missing_label']}  "
        f"label_conflicts={report['n_label_conflict']}"
    )
    print(f"[audit] duplicates: {report['duplicates']}")
    print(f"[audit] split decision: {report['split'].get('decision')}")
    print(audit.split_count_table(manifest).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
