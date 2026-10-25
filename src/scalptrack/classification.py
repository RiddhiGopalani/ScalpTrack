"""Scalp-condition classification with a classical feature baseline (REQ-04, spec Sections 14-15).

Leakage rules implemented here:
* Splits come from the manifest (grouped by duplicate cluster in ``audit``).
* Imputer, scaler and model are fitted on TRAIN only inside one sklearn Pipeline.
* Hyper-parameters / model family are selected on VALIDATION macro-F1.
* TEST is scored once, after selection is locked (``final_test_evaluation``).

Predictions are dataset-label scores from an experimental image classifier, not
a medical diagnosis and not calibrated probabilities of clinical correctness.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ALL_GROUPS, TEXTURE_GROUPS, extract_features, feature_names
from .io_utils import crop_roi, load_image
from .schema import Status

BUNDLE_VERSION = "1.0"


class ModelMissingError(FileNotFoundError):
    pass


# --------------------------------------------------------------------------- features


def feature_table_from_manifest(manifest: pd.DataFrame, source_dir: Path, cfg: dict,
                                cache_path: Path | None = None, progress: bool = True) -> pd.DataFrame:
    """Extract the fixed-schema features for every usable manifest row (cached as CSV).

    Region rows are cropped to ``bbox_xyxy`` from their parent image first.
    """
    rows = manifest[manifest["split"].isin(["train", "validation", "test"])].copy()
    cap = cfg["classification"].get("max_images_per_class")
    if cap:
        rows = (rows.groupby(["split", "canonical_label"], group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), int(cap)), random_state=cfg["seed"])))
    names = feature_names()
    cached = None
    if cache_path is not None and cache_path.is_file():
        cached = pd.read_csv(cache_path).set_index("image_id")
        if list(cached.columns[: len(names)]) != names:
            cached = None  # schema changed: recompute
    out, todo = [], []
    for r in rows.itertuples(index=False):
        if cached is not None and r.image_id in cached.index:
            out.append(cached.loc[r.image_id, names].to_dict() | {"image_id": r.image_id})
        else:
            todo.append(r)
    t0 = time.perf_counter()
    for k, r in enumerate(todo):
        rec = load_image(source_dir / r.relative_path)
        img = crop_roi(rec.rgb, tuple(r.bbox_xyxy)) if isinstance(r.bbox_xyxy, (list, tuple)) else rec.rgb
        out.append(extract_features(img, cfg) | {"image_id": r.image_id})
        if progress and (k + 1) % 500 == 0:
            print(f"[features] {k + 1}/{len(todo)} images ({time.perf_counter() - t0:.0f}s)")
    feats = pd.DataFrame(out).set_index("image_id")[names]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged = feats if cached is None else pd.concat([cached[names], feats])
        merged[~merged.index.duplicated(keep="last")].to_csv(cache_path)
    meta = rows.set_index("image_id")[["split", "canonical_label", "duplicate_group_id"]]
    return feats.join(meta, how="inner")


def feature_table_from_arrays(samples: list[dict], cfg: dict) -> pd.DataFrame:
    """Feature table for in-memory samples with keys image, label, group_id, image_id (synthetic fixtures)."""
    rows = [extract_features(s["image"], cfg) | {"image_id": s["image_id"], "canonical_label": s["label"],
                                                 "duplicate_group_id": s["group_id"]} for s in samples]
    return pd.DataFrame(rows).set_index("image_id")


# --------------------------------------------------------------------------- models


def make_candidates(cfg: dict) -> dict[str, Pipeline]:
    """Majority baseline, LogisticRegression grid (scaled) and a RandomForest fallback."""
    c = cfg["classification"]
    seed = cfg["seed"]
    cw = c.get("class_weight")
    models: dict[str, Pipeline] = {
        "majority_baseline": Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("clf", DummyClassifier(strategy="most_frequent"))]),
    }
    for C in c["logreg_C_grid"]:
        models[f"logreg_C{C:g}"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, class_weight=cw, max_iter=5000, random_state=seed)),
        ])
    models["random_forest"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=c["rf_n_estimators"], class_weight=cw, random_state=seed, n_jobs=-1)),
    ])
    return models


def metrics(y_true, y_pred, classes: list[str]) -> dict:
    """Accuracy, balanced accuracy, macro/weighted F1, per-class report and confusion matrix."""
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "labels": classes,
    }


def split_xy(table: pd.DataFrame, split: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    part = table[table["split"] == split]
    return part[names].to_numpy(dtype=np.float64), part["canonical_label"].to_numpy()


def select_on_validation(table: pd.DataFrame, cfg: dict, groups=ALL_GROUPS) -> dict:
    """Fit every candidate on train, score on validation, keep the best non-baseline by macro-F1."""
    names = feature_names(groups)
    Xtr, ytr = split_xy(table, "train", names)
    Xva, yva = split_xy(table, "validation", names)
    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError("Both train and validation splits need samples")
    classes = sorted(np.unique(ytr).tolist())
    missing_val = sorted(set(np.unique(yva)) - set(classes))
    rows, fitted = [], {}
    for name, model in make_candidates(cfg).items():
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        m = metrics(yva, model.predict(Xva), classes)
        rows.append({"model": name, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
                     "val_balanced_accuracy": m["balanced_accuracy"], "fit_seconds": fit_s})
        fitted[name] = model
    table_val = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False, kind="stable")
    best = table_val[table_val["model"] != "majority_baseline"].iloc[0]["model"]
    return {"selection_table": table_val, "best_name": best, "best_model": fitted[best],
            "baseline_model": fitted["majority_baseline"], "classes": classes, "feature_names": names,
            "feature_groups": list(groups), "classes_missing_from_train": missing_val}


def final_test_evaluation(selection: dict, table: pd.DataFrame) -> dict:
    """Score the LOCKED selected model and the majority baseline on the test split, once."""
    names, classes = selection["feature_names"], selection["classes"]
    Xte, yte = split_xy(table, "test", names)
    if len(Xte) == 0:
        return {"status": str(Status.UNAVAILABLE), "reason": "No test samples"}
    t0 = time.perf_counter()
    pred = selection["best_model"].predict(Xte)
    infer_s = time.perf_counter() - t0
    return {
        "status": str(Status.OK),
        "selected": metrics(yte, pred, classes),
        "majority_baseline": metrics(yte, selection["baseline_model"].predict(Xte), classes),
        "predictions": pd.DataFrame({"image_id": table.index[table["split"] == "test"], "true": yte, "pred": pred}),
        "timing": {"n_samples": int(len(Xte)), "classifier_predict_seconds": infer_s,
                   "note": "classifier step only (features precomputed); see per-image timing in notebook 05"},
    }


def texture_ablation(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Controlled comparison (spec Section 15.2): same split, with vs without texture descriptors.

    Each variant selects its own settings on validation; test is evaluated once per locked variant.
    """
    no_tex = tuple(g for g in ALL_GROUPS if g not in TEXTURE_GROUPS)
    rows = []
    for label, groups in (("with_texture", ALL_GROUPS), ("without_texture", no_tex)):
        sel = select_on_validation(table, cfg, groups)
        ev = final_test_evaluation(sel, table)
        val_best = sel["selection_table"].set_index("model").loc[sel["best_name"]]
        rows.append({"variant": label, "n_features": len(sel["feature_names"]), "selected_model": sel["best_name"],
                     "val_macro_f1": float(val_best["val_macro_f1"]),
                     "test_macro_f1": ev["selected"]["macro_f1"] if ev["status"] == "ok" else None,
                     "test_accuracy": ev["selected"]["accuracy"] if ev["status"] == "ok" else None,
                     "test_balanced_accuracy": ev["selected"]["balanced_accuracy"] if ev["status"] == "ok" else None})
    return pd.DataFrame(rows)


def exploratory_group_cv(table: pd.DataFrame, cfg: dict, n_splits: int = 5, model_name: str = "logreg_C1") -> dict:
    """Grouped stratified CV - the fallback when independent test support is inadequate."""
    names = feature_names()
    X = table[names].to_numpy(dtype=np.float64)
    y = table["canonical_label"].to_numpy()
    g = table["duplicate_group_id"].to_numpy()
    n_groups_min = pd.Series(g).groupby(y).nunique().min()
    k = int(min(n_splits, n_groups_min))
    if k < 2:
        return {"status": str(Status.UNAVAILABLE), "reason": "Fewer than 2 groups in some class"}
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["seed"])
    scores = []
    for tr, te in cv.split(X, y, g):
        model = make_candidates(cfg)[model_name]
        model.fit(X[tr], y[tr])
        scores.append(f1_score(y[te], model.predict(X[te]), average="macro", zero_division=0))
    return {"status": str(Status.EXPLORATORY), "n_splits": k, "macro_f1_per_fold": [float(s) for s in scores],
            "macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores))}


# --------------------------------------------------------------------------- bundles


def package_versions() -> dict:
    import cv2
    import skimage
    return {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "scikit-learn": sklearn.__version__, "scikit-image": skimage.__version__, "pandas": pd.__version__,
            "platform": platform.platform(), "processor": platform.processor()}


def make_bundle(selection: dict, cfg: dict, task_mode: str, data_provenance: str, trained_on: str,
                uncertainty_threshold: float | None = None, extra: dict | None = None) -> dict:
    """Everything needed to reproduce inference: model, class order, feature schema, config, versions."""
    model_id = f"{trained_on}__{selection['best_name']}__seed{cfg['seed']}"
    return {
        "bundle_version": BUNDLE_VERSION,
        "model_id": model_id,
        "model": selection["best_model"],
        "model_name": selection["best_name"],
        "classes": selection["classes"],
        "label_to_index": {c: i for i, c in enumerate(selection["classes"])},
        "feature_names": selection["feature_names"],
        "feature_groups": selection["feature_groups"],
        "task_mode": task_mode,
        "data_provenance": data_provenance,     # "real" or "synthetic"
        "trained_on": trained_on,
        "seed": cfg["seed"],
        "input_geometry": {"feature_size": cfg["texture"]["feature_size"], "interpolation": "INTER_AREA",
                           "color_order": "RGB"},
        "config_snapshot": {k: cfg[k] for k in ("texture", "edges", "pyramids", "classification")},
        "uncertainty_threshold": uncertainty_threshold,
        "versions": package_versions(),
        "created_unix": time.time(),
        **(extra or {}),
    }


def save_bundle(bundle: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    meta = {k: v for k, v in bundle.items() if k != "model"}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def load_bundle(path: Path | str | None) -> dict:
    if path is None or not Path(path).is_file():
        raise ModelMissingError(f"No trained model bundle at {path}. Train one with notebook 05 or "
                                "`python scripts/train_classifier.py` after downloading the dataset.")
    bundle = joblib.load(path)
    for key in ("model", "classes", "feature_names"):
        if key not in bundle:
            raise ValueError(f"Model bundle {path} is missing '{key}'")
    return bundle


def predict_condition(rgb_roi: np.ndarray, bundle: dict | None, cfg: dict) -> dict:
    """Ordered class scores and the model decision for one RGB ROI."""
    if bundle is None:
        return {"status": str(Status.UNAVAILABLE), "predicted_label": None, "class_scores": {}, "model_id": None,
                "task_mode": None, "reason": "No trained model bundle available"}
    t0 = time.perf_counter()
    feats = extract_features(rgb_roi, cfg)
    x = np.array([[feats[n] for n in bundle["feature_names"]]], dtype=np.float64)
    model = bundle["model"]
    proba = model.predict_proba(x)[0]
    order = list(model.classes_)
    scores = {c: float(proba[order.index(c)]) if c in order else 0.0 for c in bundle["classes"]}
    ranked = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
    top = next(iter(ranked))
    thr = bundle.get("uncertainty_threshold")
    decision = "low_confidence" if thr is not None and ranked[top] < thr else "predicted"
    synthetic = bundle.get("data_provenance") == "synthetic"
    return {
        "status": str(Status.SYNTHETIC_DEMO if synthetic else Status.OK),
        "predicted_label": top,
        "decision": decision,
        "class_scores": ranked,
        "model_id": bundle["model_id"],
        "task_mode": bundle["task_mode"],
        "inference_seconds": time.perf_counter() - t0,
        "reason": ("Model trained on SYNTHETIC fixtures: labels are not scalp conditions" if synthetic else
                   "Dataset-label scores from an experimental classifier; not a diagnosis or calibrated probability"),
    }


def misclassified(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions[predictions["true"] != predictions["pred"]].copy()


def choose_uncertainty_threshold(selection: dict, table: pd.DataFrame, target_coverage: float = 0.9) -> float:
    """Pick a max-score threshold on VALIDATION so that ``target_coverage`` of samples stay 'predicted'."""
    Xva, _ = split_xy(table, "validation", selection["feature_names"])
    top = selection["best_model"].predict_proba(Xva).max(axis=1)
    return float(np.quantile(top, 1.0 - target_coverage))


def time_per_image(fn: Callable[[], object], repeats: int = 5) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
