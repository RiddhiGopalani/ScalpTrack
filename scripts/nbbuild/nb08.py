from .common import code, md, notebook, save_cell_fn, setup_cell, status_cell

TITLE = "08_integrated_demo"


def build():
    c = []
    c.append(md("""
        # 08 · ScalpTrack integrated analyzer (Lab 10)

        **Owners:** Member C (integration), Member B (model loading & result schema) · **Lab:** 10 (acquisition →
        processing → feature extraction → result visualisation in one CV application) · **Requirements:** REQ-10, REQ-11,
        REQ-12; spec Sections 16–17.

        This is the examiner-facing notebook. It **loads a saved model** (it never trains during the presentation), runs
        `scalptrack.pipeline.analyze_image` on one image and ROI, shows the single-image result panel with capability
        badges, exports JSON/CSV/PNG, and keeps the stereo demonstration in a separate, clearly synthetic panel.

        Nothing unavailable is made to look operational: a missing dataset, model, scale or stereo pair is reported with
        its reason.
    """))
    c.append(setup_cell(TITLE, """
        import json
        from scalptrack import classification as clf, synthetic as syn, face_roi
        from scalptrack.pipeline import analyze_image, capability_badges
        from scalptrack.exports import export_result
        from scalptrack.audit import load_manifest
        from scalptrack.io_utils import load_image
        from scalptrack.viz import result_panel, synthetic_banner, show_grid
        from scalptrack import preprocessing as pp, edges, segmentation as sg, pyramids
    """))
    c.append(save_cell_fn())
    c.append(status_cell())
    c.append(md("## 1. Load the saved model bundle (real first, synthetic demo second, otherwise none)"))
    c.append(code("""
        real_model = project_path(cfg, "models", cfg["datasets"]["primary"], "baseline_bundle.joblib")
        syn_model = project_path(cfg, "models", "synthetic_demo", "baseline_bundle.joblib")
        BUNDLE = None
        for path in (real_model, syn_model):
            try:
                BUNDLE = clf.load_bundle(path); break
            except clf.ModelMissingError as exc:
                print("model not found:", rel(path))
        if BUNDLE is None:
            print("No model bundle: classification will be reported as unavailable. Train with notebook 05.")
        else:
            print(f"loaded {BUNDLE['model_id']} | provenance={BUNDLE['data_provenance']} | classes={BUNDLE['classes']}")
            if BUNDLE["data_provenance"] == "synthetic":
                print("*** This bundle was trained on SYNTHETIC fixtures: its labels are NOT scalp conditions. ***")
    """))
    c.append(md("""
        ## 2. Select the input image and native-pixel ROI
        With the real dataset: a fixed test-split image (the model is already locked, so this cannot influence tuning).
        Without it: the synthetic flaky close-up. Edit `IMAGE_CHOICE` / `ROI` to analyse something else;
        ROI = `(x0, y0, x1, y1)` in original pixels, max exclusive, `None` = whole image.
    """))
    c.append(code("""
        if REAL:
            man = load_manifest(STATUS["manifest_path"])
            demo_rows = man[(man["split"] == "test") & man["roi_id"].isna()].sort_values("image_id")
            pick = demo_rows.iloc[0]
            IMAGE_CHOICE = ROOT / cfg["datasets"]["sources"][STATUS["source"]]["dir"] / pick["relative_path"]
            PROVENANCE, IMAGE_ID = "real", pick["image_id"]
            print("demo image:", pick["relative_path"], "| publisher label:", pick["raw_label"])
        else:
            IMAGE_CHOICE, PROVENANCE, IMAGE_ID = SYN["flakes"]["rgb"], "synthetic", "SYNTHETIC_flaky_closeup"
            print("demo image: synthetic flaky close-up (no real dataset)")
        ROI = None
    """))
    c.append(md("## 3. Run the integrated analysis"))
    c.append(code("""
        RESULT = analyze_image(IMAGE_CHOICE, ROI, BUNDLE, cfg, provenance=PROVENANCE, image_id=IMAGE_ID)
        print("status:", RESULT["status"], "| timings (s):", {k: round(v, 3) for k, v in RESULT["timings_s"].items()})
        if PROVENANCE == "synthetic":
            print("known synthetic truth: 7 separated shafts (compare with the exploratory crossing count below)")
        BADGES = capability_badges(RESULT)
        pd.DataFrame(BADGES)
    """))
    c.append(code("""
        fig = result_panel(RESULT, BADGES, f"ScalpTrack - {RESULT['image_id']} [{PROVENANCE}]")
        if PROVENANCE == "synthetic":
            synthetic_banner(fig)
        plt.show()
        out_dir = project_path(cfg, "demo") / ("real" if PROVENANCE == "real" else "synthetic")
        PATHS = export_result(RESULT, out_dir, RESULT["image_id"], {"result_panel": fig})
        print(json.dumps({k: str(rel(Path(v))) for k, v in PATHS.items()}, indent=2))
    """))
    c.append(code("""
        record = json.loads(Path(PATHS["json"]).read_text(encoding="utf-8"))
        print(json.dumps({k: record[k] for k in ("schema_version", "image_id", "roi_xyxy", "input_hash", "run_config_id", "quality",
                                                 "classification", "hair_count", "shaft_width", "stereo")}, indent=2)[:4000])
    """))
    c.append(md("## 4. Technique gallery on the same ROI (Labs 1–7)"))
    c.append(code("""
        roi = RESULT["_arrays"]["roi_rgb"]; gray = pp.to_gray(roi)
        filt = pp.compare_filters(gray, cfg)
        ed = edges.compare_edges(gray, cfg)
        thr = sg.compare_thresholds(gray, cfg)
        lp = pyramids.laplacian_pyramid(pyramids.gaussian_pyramid(gray, 4, 16))
        lab = pp.to_lab_float(roi); hsv = pp.to_hsv(roi)
        panels = [roi, gray, filt["gaussian_5x5_s1.0"], pp.to_display_uint8(filt[next(k for k in filt if k.startswith("sharpened"))]),
                  hsv[..., 1], lab[..., 1], ed["sobel"]["magnitude"], ed["binary_maps"]["canny_50_150"],
                  thr["masks"]["otsu"], thr["masks"][next(k for k in thr["masks"] if k.startswith("adaptive_gaussian"))], lp[0], lp[2]]
        titles = ["ROI (Lab 1)", "grayscale (Lab 1/4)", "Gaussian 5x5 (Lab 2)", "Laplacian-sharpened (Lab 2)", "HSV S (Lab 4)",
                  "LAB a* signed (Lab 4)", "Sobel |g| (Lab 3)", "Canny 50/150 (Lab 3)", "Otsu (Lab 5)", "adaptive Gaussian (Lab 5)",
                  "Laplacian L0 (Lab 7)", "Laplacian L2 (Lab 7)"]
        fig = show_grid(panels, titles, ncols=6, size=2.7, signed={10, 11}, cmap="gray",
                        suptitle=f"Technique gallery - {RESULT['image_id']} [{PROVENANCE}]")
        save_fig(fig, "technique_gallery", PROVENANCE); plt.show()
        tf = RESULT["texture_features"]
        pd.Series({k: tf[k] for k in tf if k.startswith(("glcm_contrast", "glcm_energy", "glcm_entropy", "hist_entropy"))}).round(4).to_frame("Lab 6 texture")
    """))
    c.append(md("""
        ## 5. Robustness: inputs that must not crash the application
        Unreadable file, unsupported format, invalid ROI, a blank (featureless) image and missing model — each returns a
        structured status instead of an exception.
    """))
    c.append(code("""
        tmp = project_path(cfg, "results", NB); tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "corrupt.jpg").write_bytes(b"this is not a jpeg")
        (tmp / "notes.gif").write_bytes(b"GIF89a")
        cases = {
            "corrupt file": (tmp / "corrupt.jpg", None, BUNDLE),
            "unsupported extension": (tmp / "notes.gif", None, BUNDLE),
            "missing file": (tmp / "does_not_exist.png", None, BUNDLE),
            "ROI outside image": (SYN["clean"]["rgb"], (0, 0, 5000, 50), BUNDLE),
            "blank image": (np.full((160, 160, 3), 190, np.uint8), None, BUNDLE),
            "no model": (SYN["clean"]["rgb"], None, None),
        }
        rows = []
        for name, (src, roi_, bundle) in cases.items():
            r = analyze_image(src, roi_, bundle, cfg, provenance="synthetic", run_faces=False)
            rows.append({"case": name, "status": r.get("status"), "error": r.get("error"),
                         "classification": r["classification"]["status"], "crossings": r["hair_count"]["value"],
                         "crossings_status": r["hair_count"]["status"], "quality_flags": ",".join(r["quality"]["flags"]) or "-"})
        pd.DataFrame(rows)
    """))
    c.append(md("""
        ## 6. Separate stereo panel — SYNTHETIC
        Stereo needs a pair and calibration, so it is never invented from the single image above. The synthetic scene is
        passed explicitly; the figure carries the synthetic caption and filename.
    """))
    c.append(code("""
        sc = syn.render_stereo_scene(**{k: cfg["stereo"]["synthetic"][k] for k in ("width", "height", "focal_px")},
                                     baseline_mm=cfg["stereo"]["synthetic"]["baseline_mm"], seed=cfg["seed"])
        pair = {"left": sc["left"], "right": sc["right"], "calibration": sc["calibration"],
                "meta": {"pair_id": sc["pair_id"], "same_static_region_confirmed": True, "provenance": "synthetic"}}
        st_result = analyze_image(SYN["clean"]["rgb"], None, None, cfg, provenance="synthetic", image_id="SYNTHETIC_stereo_pair",
                                  run_faces=False, stereo_pair=pair)
        print(json.dumps(st_result["stereo"], indent=2))
        from scalptrack.stereo import compute_stereo
        s = compute_stereo(sc["left"], sc["right"], sc["calibration"], cfg, pair_meta=pair["meta"])
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
        axes[0].imshow(sc["left"], cmap="gray"); axes[0].set_title("left (synthetic)")
        axes[1].imshow(sc["right"], cmap="gray"); axes[1].set_title("right (synthetic)")
        im = axes[2].imshow(s["disparity_px"], cmap="viridis"); axes[2].set_title("disparity (px)"); plt.colorbar(im, ax=axes[2])
        im = axes[3].imshow(s["depth"], cmap="magma_r", vmax=1600); axes[3].set_title(f"depth ({s['depth_unit']}), valid only"); plt.colorbar(im, ax=axes[3])
        for a in axes: a.axis("off")
        fig.suptitle(f"pair {sc['pair_id']} | calibration {sc['calibration']['calibration_id']} | {sc['label']}")
        plt.tight_layout(); save_fig(fig, "stereo_panel", "synthetic"); plt.show()
    """))
    c.append(md("""
        ## 7. Optional interactive controls
        Widgets change only this demonstration, never the locked evaluation configuration. If `ipywidgets` is unavailable
        (or the notebook is executed headless), the static results above remain the reference.
    """))
    c.append(code("""
        import os
        roi_rgb = RESULT["_arrays"]["roi_rgb"]

        def _view(method="adaptive_gaussian", block=15, C=8, canny_low=50, canny_high=150):
            g = pp.to_gray(roi_rgb)
            if method == "otsu":
                m, _ = sg.threshold_otsu(g)
            elif method == "fixed":
                m = sg.threshold_fixed(g, cfg["segmentation"]["fixed_threshold"])
            else:
                b = min(block if block % 2 else block + 1, min(g.shape) - (1 - min(g.shape) % 2))
                m = sg.threshold_adaptive(g, method.split("_")[1], b, C)
            e = edges.canny(g, canny_low, max(canny_high, canny_low + 1))
            fig, ax = plt.subplots(1, 2, figsize=(9, 4))
            ax[0].imshow(sg.overlay_mask(roi_rgb, m, (255, 0, 0), .5)); ax[0].set_title(f"{method} mask"); ax[0].axis("off")
            ax[1].imshow(e, cmap="gray"); ax[1].set_title("Canny"); ax[1].axis("off"); plt.show()

        if os.environ.get("SCALPTRACK_HEADLESS") == "1":
            print("Headless execution (scripts/run_notebooks.py): widgets skipped. Open the notebook in Jupyter to use them.")
        else:
            try:
                import ipywidgets as W
                display(W.interactive(_view, method=["adaptive_gaussian", "adaptive_mean", "otsu", "fixed"], block=(3, 101, 2),
                                      C=(0, 20), canny_low=(0, 200, 5), canny_high=(10, 300, 5)))
            except ImportError:
                print("ipywidgets not installed - interactive controls skipped (static results above).")
    """))
    c.append(md("""
        ## 8. What this demo shows — and does not
        * **Shown:** acquisition/validation → quality → preprocessing → candidate masks/edges → texture features →
          saved-model classification → exploratory crossings and pixel width → capability badges → JSON/CSV/PNG export.
        * **Not shown as real:** micrometre widths (no scale), follicle counts or density (no labels), real scalp depth
          (no pair/calibration), and — while only the synthetic bundle exists — any scalp-condition prediction.
        * Classification scores are dataset-label model scores, not diagnoses.

        ## 9. Conclusion
        The integrated analyzer runs end-to-end and degrades gracefully. Once the Kaggle export is downloaded and
        notebooks 00 and 05 are re-run, this notebook automatically switches to the real model and a real test image.
    """))
    return notebook(c)
