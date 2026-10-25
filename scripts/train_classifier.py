"""Train, select (validation) and evaluate (test, once) the classical scalp-condition classifier.

Usage (from the project root):
    python scripts/train_classifier.py                    # real data: needs scripts/audit_dataset.py first
    python scripts/train_classifier.py --source kaggle_hair_diseases --max-per-class 300
    python scripts/train_classifier.py --synthetic        # SOFTWARE CHECK ONLY on generated fixtures

Outputs: models/<source>/baseline_bundle.joblib (+ .json metadata),
outputs/results/classification/<source>/metrics.json, confusion matrices under outputs/figures/.
Synthetic runs go to */synthetic_demo/ and are labelled as such.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

from scalptrack import classification as clf
from scalptrack import synthetic as syn
from scalptrack.audit import DatasetMissingError, grouped_stratified_split
from scalptrack.config import load_config, seed_everything
from scalptrack.experiments import load_real_feature_table, run_experiment


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="audited source key (default: datasets.primary)")
    ap.add_argument(
        "--synthetic", action="store_true", help="run on synthetic fixtures (software check only)"
    )
    ap.add_argument("--max-per-class", type=int, help="cap images per class and split for a quick CPU run")
    ap.add_argument("--config", help="alternative YAML configuration")
    args = ap.parse_args(argv)

    overrides = (
        {"classification": {"max_images_per_class": args.max_per_class}} if args.max_per_class else None
    )
    cfg = load_config(args.config, overrides)
    seed_everything(cfg["seed"])

    if args.synthetic:
        samples = syn.synthetic_classification_set(
            n_sources_per_class=30, variants_per_source=3, seed=cfg["seed"]
        )
        table = clf.feature_table_from_arrays(samples, cfg)
        table["split"] = grouped_stratified_split(
            table["duplicate_group_id"],
            table["canonical_label"],
            cfg["classification"]["split_fractions"],
            cfg["seed"],
        ).values
        out = run_experiment(cfg, table, "synthetic", "synthetic_fixture", "image_classification")
    else:
        source = args.source or cfg["datasets"]["primary"]
        try:
            manifest, table = load_real_feature_table(cfg, source)
        except DatasetMissingError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        task = (
            manifest.loc[manifest["split"].isin(["train", "validation", "test"]), "task_type"].mode().iat[0]
        )
        out = run_experiment(cfg, table, "real", source, task, manifest)

    m = out["metrics"]
    print(f"[train] provenance={m['provenance']}  selected={m['selected_model']}  model_id={m['model_id']}")
    print(out["selection"]["selection_table"].to_string(index=False))
    if m["test"]["status"] == "ok":
        t = m["test"]
        print(
            f"[test] selected: acc={t['selected']['accuracy']:.3f}  macroF1={t['selected']['macro_f1']:.3f}  "
            f"balanced={t['selected']['balanced_accuracy']:.3f}"
        )
        print(
            f"[test] majority: acc={t['majority_baseline']['accuracy']:.3f}  macroF1={t['majority_baseline']['macro_f1']:.3f}"
        )
    print(out["ablation"].to_string(index=False))
    print(f"[train] bundle: {out['bundle_path']}")
    print(f"[train] metrics: {out['dirs']['results'] / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
