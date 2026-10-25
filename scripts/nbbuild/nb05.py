from .common import code, md, notebook, save_cell_fn, setup_cell, status_cell

TITLE = "05_classification"


def build():
    c = []
    c.append(md("""
        # 05 · Scalp-condition classification (classical baseline)

        **Owner:** Member B · **Requirements:** REQ-03, REQ-04, REQ-12; spec Sections 14–15 (P0 classifier, P1 error
        analysis and controlled ablation).

        ### Task definition
        * **Image-level multiclass classification** of the publisher's folder labels when the audited primary source
          (Kaggle *Hair Diseases*) is used; **annotated-region classification** of ground-truth box crops if the Roboflow
          export is used instead. The task mode is read from the manifest, never assumed.
        * Predictions are *dataset-label scores* from an experimental image classifier — **not a medical diagnosis** and
          not calibrated probabilities of clinical correctness.

        ### Leakage rules implemented
        1. Splits come from the audit's grouped split manifest (duplicates/augmented copies share one split).
        2. Imputer → scaler → model are one scikit-learn `Pipeline` fitted on **train only**.
        3. Model family and regularisation are selected on **validation** macro-F1.
        4. **Test** is scored once, after selection is locked; demo images never drive tuning.

        ### Models
        Majority-class baseline (trained on train labels) · `StandardScaler + LogisticRegression` with balanced class
        weights and a small C grid · `RandomForest` fallback. Optional MobileNetV2 (P1) only if TensorFlow is installed.
    """))
    c.append(setup_cell(TITLE, """
        import json, platform, time
        from scalptrack import classification as clf, synthetic as syn
        from scalptrack.audit import grouped_stratified_split, load_manifest, split_count_table, SPLITS
        from scalptrack.experiments import load_real_feature_table, run_experiment, output_dirs
        from scalptrack.features import feature_names, FEATURE_GROUPS, TEXTURE_GROUPS
        from scalptrack.io_utils import load_image, crop_roi
        from scalptrack import transfer
        from scalptrack.viz import confusion_plot, class_score_bar
    """))
    c.append(save_cell_fn())
    c.append(status_cell())

    c.append(md("""
        ## 1. Data mode
        If the audited real manifest exists, the real experiment runs. Otherwise a **synthetic software check** runs on
        generated three-class textures (with augmentation-like grouped variants). Its labels are *not* scalp conditions and
        its scores say nothing about real performance; all of its files are written under `synthetic_demo/`.
    """))
    c.append(code("""
        if REAL:
            PROVENANCE, SOURCE = "real", STATUS["source"]
            t0 = time.perf_counter()
            MANIFEST, TABLE = load_real_feature_table(cfg, SOURCE)
            FEATURE_SECONDS = time.perf_counter() - t0
            TASK = MANIFEST.loc[MANIFEST["split"].isin(SPLITS), "task_type"].mode().iat[0]
            base_dir = ROOT / cfg["datasets"]["sources"][SOURCE]["dir"]
            def get_image(image_id):
                r = MANIFEST.set_index("image_id").loc[image_id]
                rgb = load_image(base_dir / r["relative_path"]).rgb
                return crop_roi(rgb, tuple(r["bbox_xyxy"])) if isinstance(r["bbox_xyxy"], list) else rgb
        else:
            PROVENANCE, SOURCE, TASK, MANIFEST = "synthetic", "synthetic_fixture", "image_classification", None
            SAMPLES = syn.synthetic_classification_set(n_sources_per_class=30, variants_per_source=3, seed=cfg["seed"])
            t0 = time.perf_counter()
            TABLE = clf.feature_table_from_arrays(SAMPLES, cfg)
            FEATURE_SECONDS = time.perf_counter() - t0
            TABLE["split"] = grouped_stratified_split(TABLE["duplicate_group_id"], TABLE["canonical_label"],
                                                      cfg["classification"]["split_fractions"], cfg["seed"]).values
            _imgs = {s["image_id"]: s["image"] for s in SAMPLES}
            def get_image(image_id):
                return _imgs[image_id]
            print("*** SYNTHETIC SOFTWARE CHECK - labels are generated texture classes, not scalp conditions ***")
        print(f"provenance={PROVENANCE} source={SOURCE} task={TASK} | {len(TABLE)} samples x {len(feature_names())} features "
              f"| feature extraction {FEATURE_SECONDS:.1f}s ({1000 * FEATURE_SECONDS / max(len(TABLE), 1):.1f} ms/image incl. cache)")
    """))
    c.append(md("## 2. Label and split validation"))
    c.append(code("""
        CLASSES = sorted(TABLE.loc[TABLE["split"] == "train", "canonical_label"].unique())
        print("label -> index (training order):", {k: i for i, k in enumerate(CLASSES)})
        counts = TABLE.pivot_table(index="canonical_label", columns="split", values="duplicate_group_id", aggfunc="count", fill_value=0)
        groups = TABLE.pivot_table(index="canonical_label", columns="split", values="duplicate_group_id", aggfunc="nunique", fill_value=0)
        split_tab = counts.add_suffix(" images").join(groups.add_suffix(" groups"))
        display(split_tab)
        crossing = (TABLE.groupby("duplicate_group_id")["split"].nunique() > 1).sum()
        assert crossing == 0, "a duplicate group spans several splits - leakage!"
        print(f"independent groups: {TABLE['duplicate_group_id'].nunique()} | groups crossing splits: {crossing}")
        missing = [c for c in CLASSES if (TABLE[(TABLE['split'] == 'test')]['canonical_label'] == c).sum() == 0]
        print("classes without test support:", missing or "none")
    """))

    c.append(md("""
        ## 3. Model selection on validation, then one locked test evaluation
        `run_experiment` fits every candidate on train, ranks them by validation macro-F1, locks the best non-baseline
        model, scores test once, runs the texture ablation, and saves the model bundle (model + class order + feature
        schema + config + package versions) and metrics.
    """))
    c.append(code("""
        OUT = run_experiment(cfg, TABLE, PROVENANCE, SOURCE, TASK, MANIFEST)
        SEL, EV, DIRS = OUT["selection"], OUT["evaluation"], OUT["dirs"]
        print("selected model:", SEL["best_name"], "| bundle:", rel(OUT["bundle_path"]))
        SEL["selection_table"].round(4)
    """))
    c.append(code("""
        t = EV["selected"]; b = EV["majority_baseline"]
        summary = pd.DataFrame({
            SEL["best_name"]: {k: t[k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
            "majority_baseline": {k: b[k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
        }).T
        print(f"TEST SET ({PROVENANCE}) - evaluated once after selection was locked")
        display(summary.round(4))
        per_class = pd.DataFrame(t["per_class"]).T.loc[CLASSES, ["precision", "recall", "f1-score", "support"]]
        per_class.to_csv(DIRS["results"] / "per_class_report.csv")
        per_class.round(3)
    """))
    c.append(code("""
        fig, axes = plt.subplots(1, 2, figsize=(5 + 1.3 * len(CLASSES), 2.5 + 0.6 * len(CLASSES)))
        confusion_plot(t["confusion_matrix"], CLASSES, f"{SEL['best_name']} (test)", axes[0])
        confusion_plot(b["confusion_matrix"], CLASSES, "majority baseline (test)", axes[1])
        plt.tight_layout(); save_fig(fig, f"confusion_matrices_{SOURCE}", PROVENANCE); plt.show()
    """))

    c.append(md("""
        ## 4. Controlled comparison: with vs without texture descriptors (Lab 6 contribution)
        Same groups, same split. Each variant chooses its settings on validation and is then scored once on test. The
        "texture" groups are histogram entropy, the 12 GLCM statistics and the 4 Laplacian-pyramid band energies.
    """))
    c.append(code("""
        ab = OUT["ablation"]
        display(ab.round(4))
        fig, ax = plt.subplots(figsize=(6, 3))
        ab.set_index("variant")[["val_macro_f1", "test_macro_f1"]].plot.bar(ax=ax, rot=0)
        ax.set_ylim(0, 1); ax.set_title("Texture ablation (macro-F1)")
        save_fig(fig, f"texture_ablation_{SOURCE}", PROVENANCE); plt.show()
        d = ab.set_index("variant")
        delta = d.loc["with_texture", "test_macro_f1"] - d.loc["without_texture", "test_macro_f1"]
        print(f"texture descriptors change test macro-F1 by {delta:+.3f} on this {PROVENANCE} task")
    """))

    c.append(md("""
        ## 5. Why grouping matters: measured leakage inflation (synthetic, always runs)
        Same features, same model family, two split protocols on the synthetic grouped set: a naive per-image random split
        (flipped copies of one source can sit in train and test) versus the grouped split.
    """))
    c.append(code("""
        samples = syn.synthetic_classification_set(n_sources_per_class=12, variants_per_source=4, seed=cfg["seed"] + 1)
        leak_table = clf.feature_table_from_arrays(samples, cfg)
        # Add per-source nuisance variation so that memorising a source is actually rewarded
        rng_l = np.random.default_rng(cfg["seed"])
        offsets = {g: rng_l.normal(0, 1, len(feature_names())) for g in leak_table["duplicate_group_id"].unique()}
        names = feature_names()
        stds = leak_table[names].std().replace(0, 1).to_numpy()
        leak_table[names] = leak_table[names].to_numpy() + np.stack([offsets[g] for g in leak_table["duplicate_group_id"]]) * stds * 1.5
        res = []
        for protocol in ("naive per-image", "grouped by source"):
            accs = []
            for rep in range(5):
                tab = leak_table.copy()
                if protocol.startswith("naive"):
                    tab["split"] = np.random.default_rng(rep).choice(["train", "validation", "test"], len(tab), p=[.6, .2, .2])
                else:
                    tab["split"] = grouped_stratified_split(tab["duplicate_group_id"], tab["canonical_label"], (.6, .2, .2), rep).values
                s = clf.select_on_validation(tab, cfg)
                accs.append(clf.final_test_evaluation(s, tab)["selected"]["macro_f1"])
            res.append({"protocol": protocol, "test_macro_f1_mean": np.mean(accs), "test_macro_f1_std": np.std(accs)})
        leak_res = pd.DataFrame(res); leak_res.to_csv(project_path(cfg, "results", "classification", "synthetic_demo", "leakage_experiment.csv", create=True), index=False)
        leak_res.round(3)
    """))
    c.append(md("""
        Each synthetic source received a random feature offset (a stand-in for "same person / same photo session"), so a
        model can partly recognise *sources*. With a naive split this memorisation is rewarded on test; with the grouped
        split it is not. The gap is the leakage inflation the grouped manifest prevents.

        ## 6. Exploratory grouped cross-validation
        The fallback protocol when independent test support is too small; reported as exploratory.
    """))
    c.append(code("""
        cv = clf.exploratory_group_cv(TABLE, cfg, n_splits=5, model_name="logreg_C1")
        print(json.dumps(cv, indent=2))
    """))
    c.append(md("""
        ## 7. Optional low-confidence decision (validation only)
        "Low confidence" is a decision state, not a ground-truth class. A threshold on the top score is chosen on
        **validation** only; the curve shows the coverage/accuracy trade-off there.
    """))
    c.append(code("""
        Xva = TABLE[TABLE["split"] == "validation"][SEL["feature_names"]].to_numpy()
        yva = TABLE[TABLE["split"] == "validation"]["canonical_label"].to_numpy()
        proba = SEL["best_model"].predict_proba(Xva); top = proba.max(1); pred = SEL["best_model"].classes_[proba.argmax(1)]
        rows = []
        for thr in np.linspace(0, 0.95, 20):
            keep = top >= thr
            rows.append({"threshold": thr, "coverage": keep.mean(), "accuracy_on_kept": (pred[keep] == yva[keep]).mean() if keep.any() else np.nan})
        curve = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(curve["coverage"], curve["accuracy_on_kept"], marker="o"); ax.set_xlabel("coverage (validation)"); ax.set_ylabel("accuracy on kept")
        ax.set_title("Coverage vs accuracy (validation only)")
        save_fig(fig, f"uncertainty_curve_{SOURCE}", PROVENANCE); plt.show()
        print("threshold keeping 90% of validation as 'predicted':", round(clf.choose_uncertainty_threshold(SEL, TABLE, 0.9), 3),
              "| configured:", cfg["classification"]["uncertainty_threshold"])
    """))
    c.append(md("## 8. Error analysis — misclassified test examples"))
    c.append(code("""
        preds = EV["predictions"]
        wrong = clf.misclassified(preds)
        print(f"{len(wrong)} of {len(preds)} test samples misclassified")
        display(pd.crosstab(wrong["true"], wrong["pred"]) if len(wrong) else "no errors")
        show = wrong.head(8)
        if len(show):
            fig, axes = plt.subplots(1, len(show), figsize=(2.6 * len(show), 3.0), squeeze=False)
            for ax, r in zip(axes[0], show.itertuples()):
                ax.imshow(get_image(r.image_id)); ax.axis("off")
                ax.set_title(f"true: {r.true[:18]}\\npred: {r.pred[:18]}", fontsize=7)
            plt.tight_layout(); save_fig(fig, f"misclassified_{SOURCE}", PROVENANCE); plt.show()
        right = preds[preds["true"] == preds["pred"]].groupby("true").head(1).head(6)
        if len(right):
            fig, axes = plt.subplots(1, len(right), figsize=(2.6 * len(right), 3.0), squeeze=False)
            for ax, r in zip(axes[0], right.itertuples()):
                ax.imshow(get_image(r.image_id)); ax.axis("off"); ax.set_title(f"correct: {r.true[:18]}", fontsize=7)
            plt.tight_layout(); save_fig(fig, f"correct_examples_{SOURCE}", PROVENANCE); plt.show()
    """))
    c.append(code("""
        # Which features drive the linear model? (standardised coefficients; only meaningful for logistic regression)
        model = SEL["best_model"]
        if "scale" in model.named_steps:
            coef = pd.DataFrame(model.named_steps["clf"].coef_, index=model.named_steps["clf"].classes_, columns=SEL["feature_names"])
            top_feats = coef.abs().mean().sort_values(ascending=False).head(12)
            display(top_feats.round(3).to_frame("mean |standardised coefficient|"))
        else:
            imp = pd.Series(model.named_steps["clf"].feature_importances_, index=SEL["feature_names"]).sort_values(ascending=False)
            display(imp.head(12).round(4).to_frame("random-forest importance"))
    """))
    c.append(md("## 9. Inference timing, bundle round-trip and the reusable prediction function"))
    c.append(code("""
        bundle = clf.load_bundle(OUT["bundle_path"])
        example_id = TABLE.index[TABLE["split"] == "test"][0]
        img = get_image(example_id)
        per_image = clf.time_per_image(lambda: clf.predict_condition(img, bundle, cfg), repeats=7)
        print(f"hardware: {platform.processor() or platform.machine()} | {platform.platform()}")
        print(f"per-image inference (features + model), input {img.shape[1]}x{img.shape[0]}, batch size 1, median of 7: {1000 * per_image:.1f} ms")
        pred = clf.predict_condition(img, bundle, cfg)
        print(json.dumps({k: v for k, v in pred.items() if k != "class_scores"}, indent=2, default=str))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
        axes[0].imshow(img); axes[0].axis("off"); axes[0].set_title(f"true: {TABLE.loc[example_id, 'canonical_label']}", fontsize=9)
        class_score_bar(axes[1], pred["class_scores"])
        plt.tight_layout(); save_fig(fig, f"prediction_example_{SOURCE}", PROVENANCE); plt.show()
        print("bundle metadata keys:", sorted(k for k in bundle if k != "model"))
    """))
    c.append(md("## 10. Optional P1: frozen MobileNetV2 embeddings"))
    c.append(code("""
        tstat = transfer.status()
        print(tstat)
        if tstat["status"] == "ok" and REAL:
            print("TensorFlow available: run transfer.embed on the same split and compare with the classical model (not executed by default).")
        else:
            print("MobileNetV2 comparison NOT run:", tstat["reason"] if tstat["status"] != "ok" else "no real dataset")
    """))
    c.append(md("""
        ## 11. Limitations
        * Publisher folder labels are not clinically adjudicated; some classes (e.g. alopecia areata vs telogen effluvium)
          may be visually ambiguous in web images.
        * Patient identity is unknown; grouping removes copy leakage but not same-person leakage.
        * Very high accuracy on augmented or duplicated web images is not evidence of generalisation.
        * Classical colour/texture features can pick up camera, compression and lighting differences between sources.
        * The synthetic run validates the software (training, locking, evaluation, saving, prediction) only.

        ## 12. Conclusion
        The complete classification workflow — label validation, grouped leakage-safe splits, train-only preprocessing,
        validation-based selection, a majority baseline, one locked test evaluation, a controlled texture ablation,
        grouped CV, error analysis, timing and a reusable saved bundle — is implemented and executed. Real scalp-condition
        metrics appear here only after the dataset is downloaded and notebook 00 has produced the manifest.
    """))
    return notebook(c)
