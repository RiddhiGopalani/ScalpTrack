from .common import code, md, notebook, save_cell_fn, setup_cell, status_cell

TITLE = "04_hair_measurements"


def build():
    c = []
    c.append(md("""
        # 04 · Exploratory hair counting and apparent shaft width

        **Owner:** Member C (with mask/skeleton helpers from Member A) · **Requirements:** REQ-05, REQ-06, REQ-07, REQ-11;
        spec Sections 10–11 and 15.3.

        ### Quantities and their status in this submission

        | Quantity | Definition | Status here |
        |---|---|---|
        | `isolated_strand_candidates` | separate shaft-like mask components after documented filtering | exploratory (synthetic-verified) |
        | `reference_line_crossings` | distinct mask runs crossing a line perpendicular to the dominant shaft direction | exploratory (synthetic-verified) |
        | follicle count | actual follicular openings | **not established** — no suitable labels |
        | hairs per cm² | instances / calibrated area | **unavailable** — no instance labels or scale |
        | apparent shaft width (px) | 2·distance-to-background on the skeleton, native ROI | exploratory (synthetic-verified) |
        | shaft width (µm) | `width_px × s_um_per_px` | **unavailable** — no documented scale |

        A connected component is not automatically one hair, a branched skeleton is not automatically several hairs, one
        curling shaft can cross the line more than once, and Canny's two boundaries of a shaft are never counted as two hairs.
    """))
    c.append(setup_cell(TITLE, """
        import cv2
        from scalptrack import measurements as ms, synthetic as syn
        from scalptrack.segmentation import segment_candidates, overlay_mask
        from scalptrack.quality import assess_quality
        from scalptrack.preprocessing import to_gray
        from scalptrack.viz import draw_count, draw_width, show_grid
    """))
    c.append(save_cell_fn())
    c.append(status_cell())
    c.append(code("""
        print("measurement parameters:", cfg["measurements"])
        RES = project_path(cfg, "results", NB)
        RES.mkdir(parents=True, exist_ok=True)
    """))

    c.append(md("""
        ## 1. Count verification on simple separated fixtures
        Deterministic straight shafts (1–8, several orientations) and gently curved shafts. "Separated" means gaps wider
        than the 2-px crossing merge rule (`crossing_merge_gap_px`); on these the component count and crossing count
        should equal the known truth. Shafts closer than the merge gap are merged on the line — the price of not
        double-counting a shaft split by a one-pixel mask hole.
    """))
    c.append(code("""
        rows = []
        for n in (1, 3, 5, 8):
            for ang in (0, 30, 60, 90, 135):
                f = syn.straight_shafts(n, 4.0, ang, size=200)
                r = ms.estimate_visible_hairs(None, f["mask"], cfg, "synthetic")
                rows.append(
                    {
                        "fixture": "straight",
                        "n_true": n,
                        "angle_deg": ang,
                        "components": r["components"]["value"],
                        "crossings": r["crossings"]["value"],
                    }
                )
        for n in (3, 6, 9):
            for seed in (0, 1, 2):
                f = syn.curved_shafts(n, 4.0, amplitude=8, size=200, seed=seed)
                r = ms.estimate_visible_hairs(None, f["mask"], cfg, "synthetic")
                rows.append(
                    {
                        "fixture": "curved",
                        "n_true": n,
                        "angle_deg": "~90",
                        "components": r["components"]["value"],
                        "crossings": r["crossings"]["value"],
                    }
                )
        counts = pd.DataFrame(rows)
        counts["components_ok"] = counts["components"] == counts["n_true"]
        counts["crossings_ok"] = counts["crossings"] == counts["n_true"]
        counts.to_csv(RES / "SYNTHETIC_count_verification.csv", index=False)
        print(
            f"component count exact on {counts['components_ok'].mean():.0%} of {len(counts)} simple fixtures; "
            f"crossing count exact on {counts['crossings_ok'].mean():.0%}"
        )
        counts.groupby("fixture")[["components_ok", "crossings_ok"]].mean()
    """))
    c.append(code("""
        f = syn.curved_shafts(6, 4.0, amplitude=8, size=200, seed=1)
        img = syn.render_hair_image(f["mask"], seed=1)
        r = ms.estimate_visible_hairs(img, f["mask"], cfg, "synthetic")
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        draw_count(
            axes[0],
            img,
            f["mask"],
            r,
            f"reference-line crossings = {r['crossings']['value']} (truth {f['true_crossings']})",
        )
        lab = r["labels"]
        axes[1].imshow(lab, cmap="tab20", interpolation="nearest")
        for row in r["component_rows"]:
            axes[1].text(*row["centroid_xy"], str(row["id"]), color="white", fontsize=9, ha="center")
        axes[1].set_title(f"numbered candidate components = {r['components']['value']}")
        axes[1].axis("off")
        plt.tight_layout()
        save_fig(fig, "count_overlay_simple", "synthetic")
        plt.show()
        pd.DataFrame(r["component_rows"])[
            ["id", "area_px", "skeleton_length_px", "n_endpoints", "n_junction_px", "orientation_deg", "elongation"]
        ].round(2)
    """))

    c.append(md("""
        ## 2. Challenging fixtures — errors are shown, not omitted
        Crossing, touching, fragmented, curling and noisy masks with their *intended* number of hairs.
    """))
    c.append(code("""
        hard = syn.challenging_fixtures(width=5.0, size=160, seed=cfg["seed"])
        rows = []
        fig, axes = plt.subplots(1, len(hard), figsize=(3.6 * len(hard), 3.9))
        for ax, (name, fx) in zip(axes, hard.items()):
            r = ms.estimate_visible_hairs(None, fx["mask"], cfg, "synthetic", line_xyxy=fx.get("reference_line"))
            w = ms.estimate_shaft_width(fx["mask"], cfg, "synthetic")
            rows.append(
                {
                    "fixture": name,
                    "true_hairs": fx["true_hairs"],
                    "components": r["components"]["value"],
                    "crossings": r["crossings"]["value"],
                    "flags": ", ".join(r["flags"]) or "-",
                    "width_median_px": w["median_px"],
                    "true_width_px": fx["true_width_px"],
                    "width_samples": w["valid_samples"],
                }
            )
            rgb = np.dstack([np.where(fx["mask"], 40, 220)] * 3).astype(np.uint8)
            draw_count(
                ax,
                rgb,
                fx["mask"],
                r,
                f"{name}\\ncomp={r['components']['value']} cross={r['crossings']['value']} true={fx['true_hairs']}",
            )
        plt.tight_layout()
        save_fig(fig, "challenging_fixtures", "synthetic")
        plt.show()
        hard_tab = pd.DataFrame(rows)
        hard_tab.to_csv(RES / "SYNTHETIC_challenging_fixtures.csv", index=False)
        hard_tab
    """))
    c.append(md("""
        * **crossing_X / touching:** two hairs merge into one component → under-count; junction/tip flags raise ambiguity.
        * **fragmented:** one hair broken by gaps → over-count of components; a line placed exactly in a gap reports
          0 crossings, so line placement matters.
        * **curl_multi_crossing:** one hair crosses the horizontal line several times → crossings ≠ hairs.
        * **noisy:** isolated noise pixels are removed by the minimum-area rule.
        Component counting is only appropriate when shafts are separated and branching is limited — the flags say when not.

        ### 2.1 Non-shaft components: dark halos around bright flakes
        Adaptive thresholding marks the ring of pixels *around* a bright flake as darker than its local mean, producing
        short, compact "hair" candidates. A documented shape rule removes a component only if it is **both** compact
        (elongation < `shaft_min_elongation`) **and** short (skeleton < `shaft_min_length_px`), so long compact shapes
        such as two crossing shafts are kept.
    """))
    c.append(code("""
        rgb = SYN["flakes"]["rgb"]
        seg = segment_candidates(rgb, cfg)
        kept, n_ex = ms.shaft_like_mask(seg["hair"], cfg)
        r = ms.estimate_visible_hairs(rgb, seg["hair"], cfg, "synthetic")
        fig = show_grid(
            [
                rgb,
                overlay_mask(rgb, seg["hair"], (255, 0, 0), 0.6),
                overlay_mask(rgb, seg["hair"] & ~kept, (255, 255, 0), 0.9),
                overlay_mask(rgb, kept, (255, 0, 0), 0.6),
            ],
            [
                "flaky close-up (7 true shafts)",
                "hair candidates",
                f"excluded non-shaft components ({n_ex})",
                "shaft-like mask used",
            ],
            ncols=4,
            size=3.3,
        )
        save_fig(fig, "flake_halo_filter", "synthetic")
        plt.show()
        print(
            f"components {r['components']['value']}, crossings {r['crossings']['value']} (truth 7) after excluding {n_ex} halo components"
        )
    """))
    c.append(md("""
        ## 3. Width verification — known widths, orientations and the estimator offset
        `width_px = 2·d − offset`, where *d* is the Euclidean distance transform at a skeleton pixel. Samples within
        `width_exclude_radius_px` of tips, junctions or the ROI border are excluded. The offset is a development choice
        compared on synthetic shafts — 0 (the spec's plain 2·d), 0.5 and 1.0 — with a criterion fixed in advance:
        **lowest MAE** over all widths and orientations. The cell checks that the configuration uses the winner.
    """))
    c.append(code("""
        rows = []
        for offset in (0.0, 0.5, 1.0):
            c2 = {**cfg, "measurements": {**cfg["measurements"], "width_offset_px": offset}}
            for w in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
                for ang in (0, 15, 30, 45, 60, 90, 120, 150):
                    f = syn.straight_shafts(3, w, ang, size=200)
                    r = ms.estimate_shaft_width(f["mask"], c2, "synthetic")
                    rows.append(
                        {
                            "offset": offset,
                            "true_width_px": w,
                            "angle_deg": ang,
                            "median_px": r["median_px"],
                            "iqr_px": r["iqr_px"],
                            "samples": r["valid_samples"],
                        }
                    )
        wv = pd.DataFrame(rows)
        wv["error_px"] = wv["median_px"] - wv["true_width_px"]
        wv.to_csv(RES / "SYNTHETIC_width_verification.csv", index=False)
        summary = wv.groupby("offset")["error_px"].agg(
            MAE=lambda e: e.abs().mean(), bias="mean", max_abs=lambda e: e.abs().max()
        )
        display(summary.round(3))
        best_offset = summary["MAE"].idxmin()
        print(
            "configured offset:",
            cfg["measurements"]["width_offset_px"],
            "| lowest-MAE offset:",
            best_offset,
            "| MATCH"
            if best_offset == cfg["measurements"]["width_offset_px"]
            else "| MISMATCH - update configs/default.yaml",
        )
    """))
    c.append(code("""
        sel = wv[wv["offset"] == cfg["measurements"]["width_offset_px"]]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for w, g in sel.groupby("true_width_px"):
            axes[0].plot(g["angle_deg"], g["error_px"], marker="o", label=f"{w:g}px")
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_xlabel("shaft orientation (deg)")
        axes[0].set_ylabel("median error (px)")
        axes[0].set_title("Orientation-dependent bias")
        axes[0].legend(fontsize=7, ncol=4)
        piv = sel.pivot(index="true_width_px", columns="angle_deg", values="error_px")
        im = axes[1].imshow(piv, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        axes[1].set_xticks(range(len(piv.columns)), piv.columns)
        axes[1].set_yticks(range(len(piv.index)), piv.index)
        axes[1].set_xlabel("orientation (deg)")
        axes[1].set_ylabel("true width (px)")
        axes[1].set_title("error (px)")
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        save_fig(fig, "width_bias", "synthetic")
        plt.show()
        print(
            f"configured estimator: MAE {sel['error_px'].abs().mean():.2f} px, max |error| {sel['error_px'].abs().max():.2f} px "
            f"(development target from the spec: ~1-2 px on clearly resolved fixtures)"
        )
    """))
    c.append(md("""
        The residual error alternates with width parity: the skeleton of an even-width shaft lies half a pixel off-centre,
        so a single distance sample under-reads it. Diagonal shafts are rasterised with a slightly different
        cross-section. These are properties of discrete images, documented rather than hidden.

        ### 3.1 Robustness to blur, noise, contrast and the segmentation threshold
        Now the full chain is used — render → threshold (configured method) → measure — so segmentation bias is included.
    """))
    c.append(code("""
        rows = []
        f = syn.straight_shafts(4, 6.0, 75, size=200)
        for blur in (0.5, 1.0, 2.0, 3.0):
            for noise in (2.0, 8.0):
                for contrast in (1.0, 0.5):
                    img = syn.render_hair_image(
                        f["mask"], contrast=contrast, blur_sigma=blur, noise_sigma=noise, seed=3
                    )
                    q = assess_quality(img, cfg)
                    seg = segment_candidates(img, cfg)
                    # measured without the quality gate so the effect of blur is visible; 'eligible' shows what the gate decides
                    w = ms.estimate_shaft_width(seg["hair"], cfg, "synthetic")
                    rows.append(
                        {
                            "blur_sigma": blur,
                            "noise_sigma": noise,
                            "contrast": contrast,
                            "laplacian_var": round(q["laplacian_var"], 1),
                            "quality_flags": ",".join(q["flags"]) or "-",
                            "eligible": q["measurement_eligible"],
                            "width_median_px": w["median_px"],
                            "true_px": 6.0,
                        }
                    )
        rob = pd.DataFrame(rows)
        rob["error_px"] = rob["width_median_px"] - rob["true_px"]
        rob.to_csv(RES / "SYNTHETIC_width_robustness.csv", index=False)
        rob.round(2)
    """))
    c.append(code("""
        img = syn.render_hair_image(f["mask"], blur_sigma=1.0, seed=3)
        rows = []
        for bs in (15, 31, 51):
            for C in (3, 5, 8, 12):
                c2 = {**cfg, "segmentation": {**cfg["segmentation"], "hair_block_size": bs, "adaptive_c": C}}
                w = ms.estimate_shaft_width(segment_candidates(img, c2)["hair"], cfg, "synthetic")
                rows.append({"block_size": bs, "C": C, "width_median_px": w["median_px"]})
        stab = pd.DataFrame(rows).pivot(index="C", columns="block_size", values="width_median_px")
        stab.to_csv(RES / "SYNTHETIC_width_threshold_sweep.csv")
        print("true width 6 px - median width under a development-only threshold sweep:")
        stab.round(2)
    """))
    c.append(md("""
        Width depends on the threshold and block size: a larger C keeps only the darkest core of a blurred shaft, a
        larger block admits more of the blurred fringe. The pixel width is a **method-dependent apparent width**.

        The robustness table also exposes a limitation of the blur gate: low-contrast, blurred shafts are under-measured,
        and **added noise raises the Laplacian variance**, so blurred-but-noisy images can pass the blur check (rows with
        noise 8). Laplacian variance is an empirical indicator, not a guarantee of shaft resolution.

        ### 3.2 Width overlay and distribution on a synthetic close-up
    """))
    c.append(code("""
        rgb = SYN["clean"]["rgb"]
        seg = segment_candidates(rgb, cfg)
        w = ms.estimate_shaft_width(seg["hair"], cfg, "synthetic")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        draw_width(
            axes[0],
            rgb,
            w,
            f"apparent width samples - median {w['median_px']:.2f} px, IQR {w['iqr_px']:.2f} (true 4 px)",
        )
        axes[1].hist(w["_samples"]["widths_px"], bins=np.arange(0, 12, 0.5))
        axes[1].axvline(4.0, color="r", label="true width")
        axes[1].set_xlabel("apparent width (px)")
        axes[1].legend()
        axes[1].set_title(f"{w['valid_samples']} valid samples")
        plt.tight_layout()
        save_fig(fig, "width_overlay_histogram", "synthetic")
        plt.show()
        print(
            {
                k: w[k]
                for k in (
                    "median_px",
                    "iqr_px",
                    "valid_samples",
                    "flags",
                    "status",
                    "median_um",
                    "physical_status",
                    "physical_reason",
                )
            }
        )
    """))
    c.append(md("""
        ## 4. Physical units are gated on a documented scale
        With no documented `pixel_scale_um_per_px`, micrometres stay **null** while the pixel width remains available.
        The second call below only exercises the conversion code with an explicitly **fictional** scale on a synthetic
        fixture — it is not a measurement of anything.
    """))
    c.append(code("""
        print("no scale      :", ms.to_physical_width(4.0, None))
        print("no provenance :", ms.to_physical_width(4.0, 2.5))
        print("non-isotropic :", ms.to_physical_width(4.0, 2.5, "fictional", isotropic=False))
        print(
            "API check only:",
            ms.to_physical_width(4.0, 2.5, "FICTIONAL scale for a code-path check on synthetic data"),
        )
    """))
    c.append(md("""
        ## 5. Real-image status
        On real public images there are no count labels, masks or scales, so results are **exploratory** overlays only.
        Images that fail the quality gate return `unavailable` with the reason.
    """))
    c.append(code("""
        rows = []
        if DEV:
            for d in DEV:
                q = assess_quality(d["rgb"], cfg)
                seg = segment_candidates(d["rgb"], cfg)
                r = ms.estimate_visible_hairs(d["rgb"], seg["hair"], cfg, "real", quality=q)
                w = ms.estimate_shaft_width(seg["hair"], cfg, "real", quality=q)
                rows.append(
                    {
                        "image_id": d["image_id"],
                        "label": d["label"],
                        "quality_flags": ",".join(q["flags"]) or "-",
                        "crossings": r["crossings"]["value"],
                        "crossings_status": r["crossings"]["status"],
                        "components": r["components"]["value"],
                        "count_flags": ",".join(r["flags"]) or "-",
                        "width_median_px": w["median_px"],
                        "width_status": w["status"],
                        "width_um": w["median_um"],
                    }
                )
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
                draw_count(
                    axes[0],
                    d["rgb"],
                    seg["hair"],
                    r,
                    f"{d['label']}: crossings={r['crossings']['value']} [{r['crossings']['status']}]",
                )
                draw_width(axes[1], d["rgb"], w, f"apparent width median={w['median_px']} px [{w['status']}]")
                plt.tight_layout()
                save_fig(fig, f"real_measurement_{d['image_id']}", "real")
                plt.show()
        else:
            rows.append(
                {
                    "image_id": None,
                    "label": None,
                    "quality_flags": "-",
                    "crossings": None,
                    "crossings_status": "unavailable",
                    "components": None,
                    "count_flags": "-",
                    "width_median_px": None,
                    "width_status": "unavailable",
                    "width_um": None,
                }
            )
            print(
                "No real development images: real-image measurement status is 'unavailable' (dataset not downloaded)."
            )
        real_status = pd.DataFrame(rows)
        real_status.to_csv(RES / "real_image_status.csv", index=False)
        real_status
    """))
    c.append(md("""
        ## 6. Limitations
        * Verification is synthetic; real count MAE is not reported because no count labels exist in the linked exports.
        * Width is an apparent projected 2-D width, biased by rasterisation, blur and the threshold; it is not the true
          cylindrical diameter, and micrometres require a documented scale in the same image plane.
        * Component counts fail on touching/crossing shafts; crossings over-count curling shafts.
        * Classifier-resized images must never be used for these measurements (see notebook 00, Section 5).

        ## 7. Conclusion
        Both count quantities are exact on separated synthetic fixtures and fail in the documented, flagged ways on hard
        ones. The configured width estimator stays within the spec's 1–2 px development target on clean synthetic shafts,
        with its orientation/parity bias measured. Physical width, follicle count and density remain unavailable.
        Code: `scalptrack.measurements`.
    """))
    return notebook(c)
