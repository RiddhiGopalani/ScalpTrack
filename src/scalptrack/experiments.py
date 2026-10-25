"""End-to-end classification experiment shared by notebook 05 and ``scripts/train_classifier.py``.

Real-data and synthetic runs write to separate folders so synthetic software
checks can never be mistaken for scalp-condition results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import classification as clf
from .audit import DatasetMissingError, load_manifest, missing_dataset_message, split_count_table
from .config import project_path
from .exports import write_json
from .viz import confusion_plot, synthetic_banner


def output_dirs(cfg: dict, provenance: str, source: str) -> dict[str, Path]:
    tag = "synthetic_demo" if provenance == "synthetic" else source
    return {
        "model": project_path(cfg, "models", tag),
        "results": project_path(cfg, "results", "classification", tag),
        "figures": project_path(cfg, "figures", "05_classification", tag),
    }


def manifest_path(cfg: dict, source: str) -> Path:
    return project_path(cfg, "manifests", f"{source}_manifest.csv")


def load_real_feature_table(cfg: dict, source: str | None = None, progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Manifest + cached feature table for a downloaded, audited source (raises with instructions if absent)."""
    source = source or cfg["datasets"]["primary"]
    scfg = cfg["datasets"]["sources"][source]
    mp = manifest_path(cfg, source)
    if not mp.is_file():
        raise DatasetMissingError(
            f"No manifest at {mp}. Run notebook 00 or `python scripts/audit_dataset.py --source {source}` first.\n"
            + missing_dataset_message(source, scfg, Path(cfg["root"])))
    manifest = load_manifest(mp)
    table = clf.feature_table_from_manifest(manifest, Path(cfg["root"]) / scfg["dir"], cfg,
                                            cache_path=project_path(cfg, "derived", f"{source}_features.csv"),
                                            progress=progress)
    return manifest, table


def run_experiment(cfg: dict, table: pd.DataFrame, provenance: str, trained_on: str, task_mode: str,
                   manifest: pd.DataFrame | None = None) -> dict:
    """Select on validation, test once, run the texture ablation, save bundle/metrics/figures."""
    dirs = output_dirs(cfg, provenance, trained_on)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    sel = clf.select_on_validation(table, cfg)
    thr = clf.choose_uncertainty_threshold(sel, table) if cfg["classification"].get("uncertainty_threshold") == "auto" else None
    ev = clf.final_test_evaluation(sel, table)
    ablation = clf.texture_ablation(table, cfg)
    bundle = clf.make_bundle(sel, cfg, task_mode, provenance, trained_on, thr)
    bundle_path = clf.save_bundle(bundle, dirs["model"] / "baseline_bundle.joblib")

    counts = (split_count_table(manifest) if manifest is not None else
              table.groupby(["canonical_label", "split"]).size().unstack(fill_value=0))
    metrics = {
        "provenance": provenance, "trained_on": trained_on, "task_mode": task_mode, "model_id": bundle["model_id"],
        "selected_model": sel["best_name"], "classes": sel["classes"],
        "validation_selection": sel["selection_table"].to_dict("records"),
        "test": {k: v for k, v in ev.items() if k != "predictions"},
        "ablation": ablation.to_dict("records"),
        "split_counts": counts.reset_index().to_dict("records"),
        "versions": bundle["versions"], "seed": cfg["seed"],
    }
    write_json(metrics, dirs["results"] / "metrics.json")
    sel["selection_table"].to_csv(dirs["results"] / "validation_selection.csv", index=False)
    ablation.to_csv(dirs["results"] / "texture_ablation.csv", index=False)
    if ev["status"] == "ok":
        ev["predictions"].to_csv(dirs["results"] / "test_predictions.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(4 + 1.2 * len(sel["classes"]), 2 + 0.6 * len(sel["classes"])))
        confusion_plot(ev["selected"]["confusion_matrix"], sel["classes"], f"{sel['best_name']} (test)", axes[0])
        confusion_plot(ev["majority_baseline"]["confusion_matrix"], sel["classes"], "majority baseline (test)", axes[1])
        if provenance == "synthetic":
            synthetic_banner(fig, "SYNTHETIC fixtures - software check only, not scalp-condition performance")
        fig.tight_layout()
        fig.savefig(dirs["figures"] / "confusion_matrices.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    return {"selection": sel, "evaluation": ev, "ablation": ablation, "bundle": bundle, "bundle_path": bundle_path,
            "metrics": metrics, "dirs": dirs}
