"""Build reports/capability_status.md from the evidence actually on disk (spec Section 25).

Nothing is assumed: every row is derived from files written by the notebooks/scripts.
Usage (from the project root):
    python scripts/generate_report.py              # summarise existing evidence
    python scripts/generate_report.py --run-tests  # also run pytest and record the outcome
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd

from scalptrack.config import load_config, project_path

ROOT = Path(__file__).resolve().parents[1]


def notebook_status() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        nb = nbformat.read(p, as_version=4)
        code = [c for c in nb.cells if c.cell_type == "code"]
        executed = sum(c.get("execution_count") is not None for c in code)
        errors = sum(o.output_type == "error" for c in code for o in c.get("outputs", []))
        rows.append(
            {
                "notebook": p.name,
                "code_cells": len(code),
                "executed": executed,
                "errors": errors,
                "status": "executed"
                if executed == len(code) and not errors
                else ("errors" if errors else "not executed"),
            }
        )
    return rows


def md_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown table (avoids an extra dependency on `tabulate`)."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    return "\n".join(out)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-tests", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    primary = cfg["datasets"]["primary"]
    res = project_path(cfg, "results")
    lines = [
        "# ScalpTrack capability status",
        "",
        f"Generated {dt.datetime.now().isoformat(timespec='seconds')} by `scripts/generate_report.py` from files on disk.",
        "",
    ]

    tests = None
    if args.run_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        summary = [
            ln for ln in proc.stdout.splitlines() if " passed" in ln or " failed" in ln or " error" in ln
        ]
        tests = summary[-1].strip(" =") if summary else "(no summary line)"
        lines += ["## Automated tests", "", f"`pytest`: **{tests}** (exit code {proc.returncode})", ""]

    nbs = notebook_status()
    lines += ["## Notebooks", "", md_table(pd.DataFrame(nbs)), ""]

    manifest = project_path(cfg, "manifests", f"{primary}_manifest.csv")
    audit = read_json(project_path(cfg, "manifests", f"{primary}_audit_report.json"))
    real_bundle = project_path(cfg, "models", primary, "baseline_bundle.json")
    syn_bundle = project_path(cfg, "models", "synthetic_demo", "baseline_bundle.json")
    real_metrics = read_json(res / "classification" / primary / "metrics.json")
    syn_metrics = read_json(res / "classification" / "synthetic_demo" / "metrics.json")

    def fmt_metrics(m):
        if not m or m["test"]["status"] != "ok":
            return "no test evaluation"
        t, b = m["test"]["selected"], m["test"]["majority_baseline"]
        return (
            f"{m['selected_model']}: test acc {t['accuracy']:.3f}, macro-F1 {t['macro_f1']:.3f} "
            f"(majority baseline macro-F1 {b['macro_f1']:.3f}, n={t['n']})"
        )

    count_csv = res / "04_hair_measurements" / "SYNTHETIC_count_verification.csv"
    width_csv = res / "04_hair_measurements" / "SYNTHETIC_width_verification.csv"
    stereo_csv = res / "06_stereo_depth" / "SYNTHETIC_stereo_evaluation.csv"
    face_json = read_json(res / "07_face_roi" / "face_search_summary.json")
    stereo_real = read_json(res / "06_stereo_depth" / "real_stereo_status.json")

    count_txt = "not run"
    if count_csv.is_file():
        c = pd.read_csv(count_csv)
        count_txt = (
            f"synthetic: components exact on {c['components_ok'].mean():.0%}, crossings exact on "
            f"{c['crossings_ok'].mean():.0%} of {len(c)} separated fixtures"
        )
    width_txt = "not run"
    if width_csv.is_file():
        w = pd.read_csv(width_csv)
        w = w[w["offset"] == cfg["measurements"]["width_offset_px"]]
        width_txt = f"synthetic: MAE {w['error_px'].abs().mean():.2f} px, max {w['error_px'].abs().max():.2f} px over {len(w)} width/orientation fixtures"
    stereo_txt = "not run"
    if stereo_csv.is_file():
        s = pd.read_csv(stereo_csv).iloc[0]
        stereo_txt = (
            f"synthetic: coverage {s['valid_coverage']:.1%}, disparity MAE {s['disparity_mae_px']:.2f} px, "
            f"depth MAE {s['depth_mae']:.1f} mm / RMSE {s['depth_rmse']:.1f} mm"
        )

    rows = [
        (
            "Real dataset downloaded & audited",
            "yes" if manifest.is_file() else "NO",
            (
                f"{audit['n_image_files']} files, split: {audit['split'].get('decision')}"
                if audit
                else "manual Kaggle download required - see docs/dataset_setup.md"
            ),
        ),
        (
            "Scalp-condition classifier (real data)",
            "trained" if real_bundle.is_file() else "NOT trained",
            fmt_metrics(real_metrics) if real_metrics else "requires the real dataset",
        ),
        (
            "Classifier software check (synthetic)",
            "trained" if syn_bundle.is_file() else "not run",
            (fmt_metrics(syn_metrics) + " - synthetic labels, not scalp conditions") if syn_metrics else "-",
        ),
        (
            "Texture ablation",
            "real" if real_metrics else ("synthetic only" if syn_metrics else "not run"),
            "; ".join(
                f"{r['variant']}: test macro-F1 {r['test_macro_f1']:.3f}"
                for r in (real_metrics or syn_metrics or {"ablation": []})["ablation"]
            ),
        ),
        (
            "Hair count (crossings / components)",
            "exploratory",
            count_txt + "; real images unvalidated (no count labels)",
        ),
        ("Hair width in pixels", "exploratory", width_txt),
        ("Hair width in micrometres", "unavailable", "no documented pixel scale in the linked data"),
        ("Follicle count / hairs per cm2", "unavailable", "no instance labels or scale"),
        ("Stereo disparity / depth", "synthetic demo", stereo_txt),
        (
            "Real scalp 3-D",
            (stereo_real or {}).get("real_scalp_stereo", "unavailable"),
            "no verified stereo pair/calibration",
        ),
        ("Haar face detection", "implemented", (face_json or {}).get("positive_case_status", "not run")),
        (
            "MobileNetV2 transfer learning (P1)",
            "not run",
            "TensorFlow not installed in the tested environment",
        ),
        ("Scalp health index / treatment advice", "out of scope", "deferred by the specification"),
    ]
    lines += [
        "## Capability table (from executed evidence)",
        "",
        md_table(pd.DataFrame(rows, columns=["capability", "status", "evidence"])),
        "",
    ]
    out = project_path(cfg, "reports", "capability_status.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
