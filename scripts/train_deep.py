"""Train and evaluate the v2 condition model: frozen MobileNetV2 embeddings + linear head.

Usage (from the project root, after scripts/audit_dataset.py and scripts/embed_dataset.py):
    python scripts/train_deep.py

Pipeline (every decision uses train/validation only; test is scored once at the end):
 1. Stricter leakage control. Embedding-similar pairs are verified as copies of the same photo with ORB
    geometry (rotation/zoom/flip/colour-robust) and merged with the audit's copy groups.
 2. A new grouped 70/15/15 split ("v2"), so no copy group spans train/validation/test.
 3. Before/after on the same v2 split: the old feature random forest vs the new embedding model.
 4. Head and hyper-parameters selected on validation macro-F1. Confidence and out-of-distribution
    thresholds calibrated on validation. Independent per-condition match scores (deep.MatchScorer,
    not summing to 100%) are calibrated on validation, so photos unlike every condition score low.
    The match level that reports a condition is the highest one that keeps validation recall >= 95%.
 4b. Optional "healthy" class from the user-added folder ``healthy_dir`` (config). It is far too small for
    a held-out split, so it is evaluated by leave-one-out: each healthy photo is scored by a model
    trained without it.
 5. One locked test evaluation, plus memorisation, shortcut (occlusion) and OOD probes.

Outputs: models/<source>/deep_bundle.joblib, outputs/results/model_v2/metrics.json, figures in
outputs/figures/model_v2/, data/manifests/<source>_splits_v2.csv, reports/model_v2_report.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_dataset import load_or_compute  # noqa: E402

from scalptrack import audit, deep  # noqa: E402
from scalptrack.classification import metrics, package_versions  # noqa: E402
from scalptrack.config import load_config, project_path, seed_everything  # noqa: E402
from scalptrack.features import feature_names  # noqa: E402
from scalptrack.viz import confusion_plot  # noqa: E402

CANDIDATE_SIM = 0.85  # embedding cosine that makes a pair a copy *candidate* (then ORB decides)
COPY_SIM_REPORT = 0.92  # used only to report how many test images have a very close train neighbour
CONFIDENT_COVERAGE = 0.80  # share of validation images that should receive a confident prediction
OOD_PERCENTILE = 1.0  # validation images below this percentile of nearest-train similarity count as unusual
MATCH_RECALL_TARGET = 0.95  # the match level is the highest that still reports the true condition for 95% of validation photos
MATCH_LEVELS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
HEALTHY = "healthy"
HEALTHY_THRESHOLD = 0.5  # head probability of "healthy" from which the app reports a healthy-looking scalp
MATCH_WEIGHTS = (10.0, 20.0, 40.0)  # similarity weight in the match score, chosen on validation macro-F1


def log(msg: str) -> None:
    print(f"[v2] {msg}", flush=True)


# --------------------------------------------------------------------------- 1. leakage control


def strict_groups(man: pd.DataFrame, emb: np.ndarray, base: Path, cfg: dict) -> tuple[np.ndarray, dict, np.ndarray]:
    """Merge audit copy groups with embedding candidates that ORB confirms as the same photograph."""
    old_groups = man["duplicate_group_id"].to_numpy()
    cand = deep.similar_pairs(emb, CANDIDATE_SIM)
    cand = cand[old_groups[cand[:, 0]] != old_groups[cand[:, 1]]]  # already grouped: nothing to check
    log(f"{len(cand)} embedding candidate pairs (cos >= {CANDIDATE_SIM}) not already grouped; verifying with ORB ...")
    verifier = audit.OrbVerifier(
        [base / p for p in man["relative_path"]], min_inliers=int(cfg["classification"]["near_duplicate_orb_inliers"])
    )
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = np.array(list(ex.map(lambda p: verifier(int(p[0]), int(p[1])), cand)), dtype=bool) if len(cand) else np.zeros(0, bool)
    confirmed = cand[ok]
    uf = audit._UnionFind(len(man))
    first: dict = {}
    for i, g in enumerate(old_groups):
        if g in first:
            uf.union(first[g], i)
        else:
            first[g] = i
    for i, j in confirmed:
        uf.union(int(i), int(j))
    roots = np.array([uf.find(i) for i in range(len(man))])
    _, gidx = np.unique(roots, return_inverse=True)
    sizes = np.bincount(gidx)
    stats = {
        "embedding_candidates": int(len(cand)),
        "orb_confirmed_new_copy_pairs": int(len(confirmed)),
        "groups_before": int(len(np.unique(old_groups))),
        "groups_after": int(len(sizes)),
        "largest_group_after": int(sizes.max()),
    }
    return gidx, stats, confirmed


# --------------------------------------------------------------------------- 3-4. models


def make_head(C, seed):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed))])


def fit_linear_head(Xtr, ytr, Xva, yva, classes, seed):
    """Logistic-regression head on standardised embeddings; C chosen on validation macro-F1."""
    rows, best = [], None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        model = make_head(C, seed)
        model.fit(Xtr, ytr)
        m = metrics(yva, model.predict(Xva), classes)
        rows.append({"model": f"linear_head_C{C:g}", "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"]})
        if best is None or m["macro_f1"] > best[0]:
            best = (m["macro_f1"], C, model)
    return best[2], best[1], pd.DataFrame(rows)


def fit_old_model(feat: pd.DataFrame, tr, va, classes, seed):
    """The previous model family (random forest on the 43 hand-crafted features), retrained on the v2 split."""
    names = feature_names()
    model = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1))])
    model.fit(feat.loc[tr, names].to_numpy(), feat.loc[tr, "y"].to_numpy())
    return model, names


def load_healthy(cfg: dict, source: str, bb: deep.Backbone) -> tuple[np.ndarray, list[str]]:
    """Embeddings of the user-added healthy photos (byte-identical copies dropped); empty if none."""
    import hashlib

    rel = cfg["datasets"]["sources"][source].get("healthy_dir")
    folder = Path(cfg["root"]) / cfg["datasets"]["sources"][source]["dir"] / rel if rel else None
    if folder is None or not folder.is_dir():
        return np.zeros((0, deep.EMBED_DIM), np.float32), []
    files, seen = [], set()
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            digest = hashlib.md5(f.read_bytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                files.append(f)
    return deep.normalise(bb.embed([deep.load_rgb(f) for f in files])), [f.name for f in files]


def healthy_leave_one_out(Xtr, ytr, Xva, yva, H, C, weight, seed) -> pd.DataFrame:
    """Score each healthy photo with a model trained without it."""
    rows = []
    for i in range(len(H)):
        rest = np.delete(H, i, 0)
        X, Y = np.vstack([Xtr, rest]), np.concatenate([ytr, [HEALTHY] * len(rest)])
        sc = deep.MatchScorer(make_head(C, seed).fit(X, Y), X, Y, weight=weight, healthy_label=HEALTHY).calibrate(Xva, yva)
        rows.append({"photo": i, "p_healthy": float(sc.healthy_probability(H[i:i + 1])[0]),
                     "highest_condition_match": float(sc(H[i:i + 1]).max())})
    return pd.DataFrame(rows)


def choose_match_level(match: np.ndarray, true_idx: np.ndarray) -> float:
    """Highest match level whose validation recall (true condition reported) is >= MATCH_RECALL_TARGET."""
    true_match = match[np.arange(len(true_idx)), true_idx]
    ok = [lvl for lvl in MATCH_LEVELS if (true_match >= lvl).mean() >= MATCH_RECALL_TARGET]
    return max(ok) if ok else min(MATCH_LEVELS)


# --------------------------------------------------------------------------- figures


def example_grid(paths, titles, path, ncols=6, suptitle=""):
    n = len(paths)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.9 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p, t in zip(axes.ravel(), paths, titles):
        ax.imshow(deep.load_rgb(p))
        ax.set_title(t, fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main() -> int:
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    source = cfg["datasets"]["primary"]
    base = Path(cfg["root"]) / cfg["datasets"]["sources"][source]["dir"]
    fig_dir = project_path(cfg, "figures", "model_v2")
    res_dir = project_path(cfg, "results", "model_v2")
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    manifest = audit.load_manifest(project_path(cfg, "manifests", f"{source}_manifest.csv"))
    ids, emb_all = load_or_compute(cfg, source)
    emb_by_id = dict(zip(ids, emb_all))
    man = manifest[manifest["split"].isin(audit.SPLITS)].reset_index(drop=True)
    emb = np.stack([emb_by_id[i] for i in man["image_id"]])
    y = man["canonical_label"].to_numpy()
    classes = sorted(np.unique(y).tolist())
    log(f"{len(man)} usable images, {len(classes)} classes, embeddings {emb.shape}")

    # ---- 1. leakage diagnosis on the current (v1) split
    v1 = man["split"].to_numpy()
    def nn_leak(split):
        tr, te = split == "train", split == "test"
        sim, _ = deep.max_similarity(emb[te], emb[tr])
        return float((sim >= COPY_SIM_REPORT).mean()), float(np.median(sim))
    leak_v1 = nn_leak(v1)
    log(f"v1 split: {leak_v1[0]:.1%} of test images have a train image with cosine >= {COPY_SIM_REPORT} (median nearest {leak_v1[1]:.3f})")

    t0 = time.perf_counter()
    gidx, dup_stats, confirmed = strict_groups(man, emb, base, cfg)
    log(f"copy groups: {dup_stats} ({time.perf_counter() - t0:.0f}s)")

    # Same photograph under different labels -> exclude both (cannot know which label is right)
    conflict = {k for a, b in confirmed if y[a] != y[b] for k in (a, b)}
    keep = ~np.isin(np.arange(len(man)), list(conflict))
    dup_stats["new_label_conflict_images_excluded"] = int((~keep).sum())

    # ---- 2. stricter grouped split
    groups = pd.Series([f"g{g}" for g in gidx])
    v2 = np.full(len(man), "excluded", dtype=object)
    v2[keep] = audit.grouped_stratified_split(groups[keep], pd.Series(y[keep]), tuple(cfg["classification"]["split_fractions"]), seed).to_numpy()
    leak_v2 = nn_leak(v2)
    log(f"v2 split: {leak_v2[0]:.1%} of test images have a train image with cosine >= {COPY_SIM_REPORT} (median nearest {leak_v2[1]:.3f})")
    pd.DataFrame({"image_id": man["image_id"], "relative_path": man["relative_path"], "canonical_label": y,
                  "copy_group_v2": groups, "split_v2": v2}).to_csv(project_path(cfg, "manifests", f"{source}_splits_v2.csv"), index=False)
    tr, va, te = (v2 == "train"), (v2 == "validation"), (v2 == "test")
    split_counts = pd.crosstab(pd.Series(y[keep], name="class"), pd.Series(v2[keep], name="split"))
    log("v2 split counts:\n" + split_counts.to_string())

    # ---- 3a. BEFORE: old model family on the v2 split
    feat = pd.read_csv(project_path(cfg, "derived", f"{source}_features.csv")).set_index("image_id")
    feat = feat.reindex(man["image_id"])
    feat["y"] = y
    feat.index = range(len(feat))
    old_model, names = fit_old_model(feat, tr, va, classes, seed)
    old_test = metrics(y[te], old_model.predict(feat.loc[te, names].to_numpy()), classes)
    old_val = metrics(y[va], old_model.predict(feat.loc[va, names].to_numpy()), classes)
    log(f"BEFORE (old RF, v2 split): val macro-F1 {old_val['macro_f1']:.3f} | test acc {old_test['accuracy']:.3f} macro-F1 {old_test['macro_f1']:.3f}")

    # ---- 3b. AFTER: embedding model, selected on validation (plus the optional healthy photos in training)
    bb = deep.Backbone()
    H, healthy_files = load_healthy(cfg, source, bb)
    log(f"healthy photos: {len(H)} unique" + ("" if len(H) else " (no healthy_dir folder; training without a healthy class)"))
    Xtr = np.vstack([emb[tr], H])
    ytr = np.concatenate([y[tr], [HEALTHY] * len(H)])
    head, best_C, selection = fit_linear_head(Xtr, ytr, emb[va], y[va], classes, seed)
    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine").fit(emb[tr], y[tr])
    selection = pd.concat([selection, pd.DataFrame([
        {"model": "old_feature_random_forest", "val_macro_f1": old_val["macro_f1"], "val_accuracy": old_val["accuracy"]},
        {"model": "1nn_memorisation_reference", "val_macro_f1": metrics(y[va], knn.predict(emb[va]), classes)["macro_f1"], "val_accuracy": None},
    ])], ignore_index=True)
    log("validation selection:\n" + selection.round(4).to_string(index=False))

    # ---- 4. thresholds on validation only
    p_val = head.predict_proba(emb[va])
    top_val = p_val.max(1)
    conf_thr = float(np.quantile(top_val, 1 - CONFIDENT_COVERAGE))
    sim_val, _ = deep.max_similarity(emb[va], emb[tr])
    ood_thr = float(np.percentile(sim_val, OOD_PERCENTILE))
    pred_val = head.classes_[p_val.argmax(1)]
    conf_val_acc = float((pred_val[top_val >= conf_thr] == y[va][top_val >= conf_thr]).mean())
    log(f"confidence threshold {conf_thr:.3f} (val coverage {CONFIDENT_COVERAGE:.0%}, val accuracy when confident {conf_val_acc:.3f}); OOD similarity threshold {ood_thr:.3f}")

    # Independent per-condition match scores (do not sum to 100%), calibrated on validation
    scorers = {w: deep.MatchScorer(head, Xtr, ytr, weight=w, healthy_label=HEALTHY).calibrate(emb[va], y[va]) for w in MATCH_WEIGHTS}
    match_sel = {w: metrics(y[va], np.array(s.match_classes)[s(emb[va]).argmax(1)], classes)["macro_f1"] for w, s in scorers.items()}
    best_w = max(match_sel, key=match_sel.get)
    scorer = scorers[best_w]
    match_level = choose_match_level(scorer(emb[va]), np.array([scorer.match_classes.index(c) for c in y[va]]))
    log(f"match-score weight selection (val macro-F1): {match_sel} -> {best_w}; match level {match_level} (val recall >= {MATCH_RECALL_TARGET:.0%})")

    healthy_eval = None
    if len(H):
        loo = healthy_leave_one_out(emb[tr], y[tr], emb[va], y[va], H, best_C, best_w, seed)
        healthy_eval = {
            "unique_photos": int(len(H)),
            "leave_one_out_called_healthy": f"{int((loo.p_healthy >= HEALTHY_THRESHOLD).sum())}/{len(loo)}",
            "leave_one_out_no_condition_reported": f"{int(((loo.p_healthy >= HEALTHY_THRESHOLD) | (loo.highest_condition_match < match_level)).sum())}/{len(loo)}",
            "leave_one_out": loo.to_dict("records"),
        }
        log(f"healthy leave-one-out:\n{loo.round(3).to_string(index=False)}")

    # ---- 5. TEST, once
    p_te = head.predict_proba(emb[te])
    pred_te = head.classes_[p_te.argmax(1)]
    new_test = metrics(y[te], pred_te, classes)
    top_te = p_te.max(1)
    confident = top_te >= conf_thr
    conf_test = {"coverage": float(confident.mean()), "accuracy_when_confident": float((pred_te[confident] == y[te][confident]).mean()),
                 "accuracy_when_uncertain": float((pred_te[~confident] == y[te][~confident]).mean()) if (~confident).any() else None}
    knn_test = metrics(y[te], knn.predict(emb[te]), classes)
    log(f"AFTER (embedding head, v2 split): test acc {new_test['accuracy']:.3f} macro-F1 {new_test['macro_f1']:.3f} | 1-NN {knn_test['accuracy']:.3f}")
    log(f"confident subset: {conf_test}")

    m_te = scorer(emb[te])
    true_idx = np.array([scorer.match_classes.index(c) for c in y[te]])
    called_healthy = scorer.healthy_probability(emb[te]) >= HEALTHY_THRESHOLD
    shown = (m_te.max(1) >= match_level) & ~called_healthy
    match_pred = np.array(scorer.match_classes)[m_te.argmax(1)]
    match_test = {
        **{k: v for k, v in metrics(y[te], match_pred, classes).items() if k in ("accuracy", "balanced_accuracy", "macro_f1")},
        "recall_true_condition_reported": float(((m_te[np.arange(len(true_idx)), true_idx] >= match_level) & ~called_healthy).mean()),
        "condition_photos_called_healthy": float(called_healthy.mean()),
        "condition_photos_with_no_condition_reported": float((~shown).mean()),
        "accuracy_when_a_match_is_shown": float((match_pred[shown] == y[te][shown]).mean()),
        "median_true_condition_match": float(np.median(m_te[np.arange(len(true_idx)), true_idx])),
        "mean_conditions_at_or_above_match_level": float((m_te >= match_level).sum(1).mean()),
    }
    log(f"match scores on test: {match_test}")

    # Shortcut probe: same test images with the centre or the border hidden
    te_paths = [base / p for p in man.loc[te, "relative_path"]]
    probe = {}
    for keep_part in ("centre", "border"):
        e = bb.embed([deep.occlude(deep.load_rgb(p), keep_part) for p in te_paths])
        probe[f"keep_{keep_part}_only"] = float((head.predict(e) == y[te]).mean())
    log(f"shortcut probe (test accuracy): full {new_test['accuracy']:.3f} | {probe}")

    # OOD probe: clearly non-scalp images shipped with scikit-image (never used for tuning)
    from skimage import data as skdata
    ood_imgs = {}
    for name in ("astronaut", "coffee", "chelsea", "rocket", "camera", "coins", "brick", "grass", "gravel", "hubble_deep_field"):
        try:
            im = getattr(skdata, name)()
        except Exception:  # noqa: BLE001 - optional sample files may be missing
            continue
        ood_imgs[name] = np.dstack([im] * 3) if im.ndim == 2 else im[..., :3]
    rng = np.random.default_rng(seed)
    ood_imgs["random_noise"] = (rng.random((400, 600, 3)) * 255).astype(np.uint8)
    ood_emb = bb.embed(list(ood_imgs.values()))
    ood_sim, _ = deep.max_similarity(ood_emb, emb[tr])
    ood_top = head.predict_proba(ood_emb).max(1)
    ood_match = scorer(ood_emb).max(1)
    ood_healthy = scorer.healthy_probability(ood_emb)
    ood_rows = [{"image": k, "nearest_train_similarity": float(s), "flagged_ood": bool(s < ood_thr),
                 "top_score": float(t), "would_be_confident": bool(t >= conf_thr and s >= ood_thr),
                 "highest_match_score": float(m), "p_healthy": float(h)}
                for k, s, t, m, h in zip(ood_imgs, ood_sim, ood_top, ood_match, ood_healthy)]
    sim_te, _ = deep.max_similarity(emb[te], emb[tr])
    ood_summary = {"ood_images_flagged": f"{sum(r['flagged_ood'] for r in ood_rows)}/{len(ood_rows)}",
                   "test_images_wrongly_flagged_ood": float((sim_te < ood_thr).mean())}
    log(f"OOD probe: {ood_summary}")

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(5 + 2.6 * len(classes) / 2, 1.8 + 0.6 * len(classes)))
    confusion_plot(old_test["confusion_matrix"], classes, f"BEFORE: feature RF (acc {old_test['accuracy']:.2f})", ax[0])
    confusion_plot(new_test["confusion_matrix"], classes, f"AFTER: MobileNetV2 + linear head (acc {new_test['accuracy']:.2f})", ax[1])
    fig.tight_layout(); fig.savefig(fig_dir / "confusion_before_after.png", dpi=120); plt.close(fig)
    te_rows = man[te].reset_index(drop=True)
    right = np.nonzero(pred_te == y[te])[0]
    wrong = np.nonzero(pred_te != y[te])[0]
    pick = lambda idx: np.random.default_rng(seed).choice(idx, min(12, len(idx)), replace=False) if len(idx) else []
    for tag, idx in (("correct", pick(right)), ("incorrect", pick(wrong))):
        example_grid([base / te_rows.loc[i, "relative_path"] for i in idx],
                     [f"true: {y[te][i]}\npred: {pred_te[i]} ({top_te[i]:.2f})" for i in idx],
                     fig_dir / f"test_examples_{tag}.png", suptitle=f"Held-out test examples - {tag}")
    per_class = pd.DataFrame(new_test["per_class"]).T.loc[classes, ["precision", "recall", "f1-score", "support"]]
    per_class.to_csv(res_dir / "per_class_test.csv")

    # ---- save bundle (head + thresholds + train embeddings for OOD) and metrics
    model_dir = project_path(cfg, "models", source)
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "bundle_version": "2.0", "model_type": "frozen_mobilenetv2_linear_head", "backbone": deep.BACKBONE,
        "model": head, "classes": list(head.classes_), "best_C": best_C, "data_provenance": "real", "trained_on": source,
        "confidence_threshold": conf_thr, "ood_similarity_threshold": ood_thr,
        "match_scorer": scorer, "match_level": match_level,
        "healthy_label": HEALTHY if len(H) else None, "healthy_threshold": HEALTHY_THRESHOLD, "seed": seed, "versions": package_versions(),
        "model_id": f"{source}__mobilenetv2_linear_C{best_C:g}__seed{seed}",
        "split": "v2 (copy groups verified by ORB, grouped 70/15/15)",
    }
    joblib.dump(bundle, model_dir / "deep_bundle.joblib", compress=3)
    report = {
        "dataset": {"source": source, "usable_images": int(len(man)), "split_v2_counts": split_counts.to_dict()},
        "leakage": {**dup_stats, f"test_with_train_neighbour_cos>={COPY_SIM_REPORT}_v1": leak_v1[0],
                    f"test_with_train_neighbour_cos>={COPY_SIM_REPORT}_v2": leak_v2[0],
                    "median_nearest_train_cos_v1": leak_v1[1], "median_nearest_train_cos_v2": leak_v2[1]},
        "before_old_rf_v2_split": {k: old_test[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
        "after_embedding_head_v2_split": {k: new_test[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
        "after_per_class": new_test["per_class"], "after_confusion_matrix": new_test["confusion_matrix"], "classes": classes,
        "memorisation_1nn_test_accuracy": knn_test["accuracy"],
        "validation_selection": selection.to_dict("records"), "selected_C": best_C,
        "thresholds": {"confidence": conf_thr, "ood_similarity": ood_thr, "val_accuracy_when_confident": conf_val_acc},
        "test_confidence_gating": conf_test,
        "match_scores": {"weight_selection_val_macro_f1": match_sel, "selected_weight": best_w, "match_level": match_level,
                         "recall_target_val": MATCH_RECALL_TARGET, "test": match_test},
        "healthy": healthy_eval,
"shortcut_probe_test_accuracy": {"full_image": new_test["accuracy"], **probe},
        "ood_probe": {"summary": ood_summary, "images": ood_rows},
    }
    (res_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log(f"saved {model_dir / 'deep_bundle.joblib'} and {res_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
