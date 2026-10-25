from .common import code, md, notebook, save_cell_fn, setup_cell

TITLE = "06_stereo_depth"


def build():
    c = []
    c.append(md("""
        # 06 · Stereo disparity and calibrated depth

        **Owner:** Member C · **Lab:** 8 (stereo disparity and depth estimation) · **Requirement:** REQ-08; spec Section 12.

        ### Real-data gate first
        A single photo is not stereo; two unrelated disease images are not stereo; a shifted copy of one photo is not
        evidence of depth. Metric depth needs intrinsics, distortion, the baseline and their units. The linked public
        exports contain no verified stereo pairs, so the expected status is:

        > **real scalp stereo: unavailable — no verified stereo pair/calibration**

        The algorithm is then demonstrated on a **deterministic synthetic 3-D scene** rendered from two pinhole cameras with
        a known baseline. Every figure and file from that branch is captioned and named *SYNTHETIC*.
        **Faculty acceptance of synthetic-only Lab 8 coverage is unconfirmed** and is disclosed as such.
    """))
    c.append(setup_cell(TITLE, """
        import json
        import cv2
        from scalptrack import stereo, synthetic as syn
        from scalptrack.exports import write_json
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    """))
    c.append(save_cell_fn())
    c.append(md("## 1. Real-data gate"))
    c.append(code("""
        real_pairs = sorted((ROOT / "data" / "raw").rglob("*stereo*")) if (ROOT / "data" / "raw").is_dir() else []
        calib_files = (
            sorted((ROOT / "data" / "raw").rglob("*calib*.json")) if (ROOT / "data" / "raw").is_dir() else []
        )
        gate = stereo.validate_stereo_inputs(None, None, None, None)
        real_status = {
            "real_scalp_stereo": gate["status"],
            "reason": "no verified stereo pair/calibration in the linked exports"
            if not real_pairs
            else "candidate files found - verify pair metadata and calibration",
            "candidate_pair_files": [str(rel(p)) for p in real_pairs][:10],
            "calibration_files": [str(rel(p)) for p in calib_files][:10],
            "gate_failed_checks": gate["failed_checks"],
        }
        write_json(real_status, project_path(cfg, "results", NB, "real_stereo_status.json"))
        print(json.dumps(real_status, indent=2))
    """))
    c.append(md("""
        ## 2. Synthetic scene: geometry, cameras and ground truth
        Background plane (Z = 1500 mm), two textured rectangles (900 mm, 650 mm) and a spherical dome (a curved *synthetic*
        surface). Both views are ray-cast from the same world texture; occlusions arise naturally.
    """))
    c.append(code("""
        sc_cfg = cfg["stereo"]["synthetic"]
        SCENE = syn.render_stereo_scene(
            sc_cfg["width"],
            sc_cfg["height"],
            sc_cfg["focal_px"],
            sc_cfg["baseline_mm"],
            seed=cfg["seed"],
            include_dome=True,
        )
        cal = SCENE["calibration"]
        prov = {
            "label": SCENE["label"],
            "pair_id": SCENE["pair_id"],
            "scene": SCENE["scene"],
            "calibration": {k: v for k, v in cal.items()},
            "seed": cfg["seed"],
        }
        p = write_json(prov, project_path(cfg, "results", NB, "SYNTHETIC_scene_provenance.json"))
        print("saved", rel(p))
        np.savez_compressed(
            project_path(cfg, "results", NB, "SYNTHETIC_scene_arrays.npz"),
            left=SCENE["left"],
            right=SCENE["right"],
            gt_disparity_px=SCENE["gt_disparity_px"],
            gt_depth_mm=SCENE["gt_depth"],
            visible=SCENE["visible_in_right"],
        )
        print("K =\\n", cal["K1"], "\\nbaseline =", cal["baseline"], cal["units"])
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
        axes[0].imshow(SCENE["left"], cmap="gray")
        axes[0].set_title("left camera")
        axes[1].imshow(SCENE["right"], cmap="gray")
        axes[1].set_title(f"right camera (+{cal['baseline']} mm in X)")
        im = axes[2].imshow(SCENE["gt_disparity_px"], cmap="viridis")
        axes[2].set_title("ground-truth disparity (px)")
        plt.colorbar(im, ax=axes[2])
        axes[3].imshow(SCENE["visible_in_right"], cmap="gray")
        axes[3].set_title("visible in right view (white)")
        for a in axes:
            a.axis("off")
        plt.tight_layout()
        save_fig(fig, "scene_inputs_ground_truth", "synthetic")
        plt.show()
    """))
    c.append(md("""
        ## 3. Matching: StereoSGBM, fixed-point scaling and validity masks
        `numDisparities` is a positive multiple of 16 and the block size is odd. OpenCV returns fixed-point disparities
        (×16); they are divided by 16. Validity = matched ∧ left-right consistent (|d_L(x) − d_R(x − d_L)| ≤ 1 px) ∧
        not in the border band (including the leftmost `numDisparities` columns, where the full search range is unavailable).
    """))
    c.append(code("""
        meta = {"pair_id": SCENE["pair_id"], "same_static_region_confirmed": True, "provenance": "synthetic"}
        RES = stereo.compute_stereo(SCENE["left"], SCENE["right"], cal, cfg, pair_meta=meta)
        print("status:", RES["status"], "|", RES["reason"])
        print("matching parameters:", RES["params"])
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
        im = axes[0].imshow(RES["disparity_px"], cmap="viridis", vmin=0, vmax=np.nanmax(SCENE["gt_disparity_px"]))
        axes[0].set_title("SGBM disparity (px)")
        plt.colorbar(im, ax=axes[0])
        axes[1].imshow(RES["matched"], cmap="gray")
        axes[1].set_title("matched")
        axes[2].imshow(RES["lr_consistent"], cmap="gray")
        axes[2].set_title("left-right consistent")
        axes[3].imshow(RES["valid"], cmap="gray")
        axes[3].set_title(f"final valid ({RES['valid_fraction']:.1%})")
        for a in axes:
            a.axis("off")
        plt.tight_layout()
        save_fig(fig, "disparity_validity", "synthetic")
        plt.show()
    """))
    c.append(md("""
        ## 4. Depth from the calibrated Q matrix
        For the rectified synthetic pair, Q is built from f, cₓ, c_y and the baseline; `cv2.reprojectImageTo3D` then gives
        Z = f·B/d in **millimetres** (the baseline's unit). The cell first verifies Q against the closed-form equation.
    """))
    c.append(code("""
        Q = RES["Q"]
        print("Q =\\n", np.round(Q, 4))
        d_test = 40.0
        z_q = cv2.reprojectImageTo3D(np.full((2, 2), d_test, np.float32), Q)[0, 0, 2]
        print(f"d = {d_test} px -> Z(Q) = {z_q:.2f} mm, f*B/d = {cal['focal_px'] * cal['baseline'] / d_test:.2f} mm")
        depth = RES["depth"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        vmin, vmax = np.nanmin(SCENE["gt_depth"]), 1600
        im = axes[0].imshow(SCENE["gt_depth"], cmap="magma_r", vmin=vmin, vmax=vmax)
        axes[0].set_title("ground-truth depth (mm)")
        plt.colorbar(im, ax=axes[0])
        im = axes[1].imshow(depth, cmap="magma_r", vmin=vmin, vmax=vmax)
        axes[1].set_title("estimated depth (mm), invalid = blank")
        plt.colorbar(im, ax=axes[1])
        err = np.abs(RES["disparity_px"] - SCENE["gt_disparity_px"])
        err[~RES["valid"]] = np.nan
        im = axes[2].imshow(err, cmap="inferno", vmin=0, vmax=2)
        axes[2].set_title("|disparity error| (px) on valid pixels")
        plt.colorbar(im, ax=axes[2])
        for a in axes:
            a.axis("off")
        plt.tight_layout()
        save_fig(fig, "depth_and_error", "synthetic")
        plt.show()
    """))
    c.append(md("""
        ## 5. Evaluation — coverage and error reported together
        The evaluation region is every pixel visible in both views and outside the border band. Hiding difficult pixels
        would lower the error, so coverage is always printed next to it.
    """))
    c.append(code("""
        ev = stereo.evaluate_against_truth(
            RES,
            SCENE["gt_disparity_px"],
            SCENE["gt_depth"],
            SCENE["visible_in_right"],
            cfg["stereo"]["error_threshold_px"],
        )
        rows = [{"matcher": "SGBM + LR check", **ev}]
        # StereoBM comparison on the same pair and validity rules
        bm = stereo.disparity_bm(SCENE["left"], SCENE["right"], cfg, block_size=15)
        bm_res = {**RES, "disparity_px": bm, "valid": np.isfinite(bm) & ~RES["border"], "depth": None}
        rows.append(
            {
                "matcher": "StereoBM (no LR check)",
                **stereo.evaluate_against_truth(
                    bm_res,
                    SCENE["gt_disparity_px"],
                    SCENE["gt_depth"],
                    SCENE["visible_in_right"],
                    cfg["stereo"]["error_threshold_px"],
                ),
            }
        )
        # Per-surface breakdown for SGBM
        for sid, s in enumerate(SCENE["scene"]["surfaces"]):
            m = SCENE["surface_id"] == sid
            sub = {**RES, "valid": RES["valid"] & m}
            r = stereo.evaluate_against_truth(
                sub,
                SCENE["gt_disparity_px"],
                SCENE["gt_depth"],
                SCENE["visible_in_right"] & m,
                cfg["stereo"]["error_threshold_px"],
            )
            rows.append({"matcher": f"SGBM on {s['name']}", **r})
        table = pd.DataFrame(rows)
        table.to_csv(project_path(cfg, "results", NB, "SYNTHETIC_stereo_evaluation.csv", create=True), index=False)
        table.round(3)
    """))
    c.append(code("""
        v = RES["valid"] & SCENE["visible_in_right"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        axes[0].scatter(SCENE["gt_disparity_px"][v][::25], RES["disparity_px"][v][::25], s=2, alpha=0.4)
        lim = [0, np.nanmax(SCENE["gt_disparity_px"]) * 1.05]
        axes[0].plot(lim, lim, "r--", lw=1)
        axes[0].set_xlabel("true disparity (px)")
        axes[0].set_ylabel("estimated (px)")
        axes[0].set_title("disparity: estimate vs truth")
        axes[1].scatter(SCENE["gt_depth"][v][::25], RES["depth"][v][::25], s=2, alpha=0.4)
        lim = [500, 1600]
        axes[1].plot(lim, lim, "r--", lw=1)
        axes[1].set_xlabel("true depth (mm)")
        axes[1].set_ylabel("estimated depth (mm)")
        axes[1].set_title("depth: estimate vs truth")
        plt.tight_layout()
        save_fig(fig, "truth_vs_estimate", "synthetic")
        plt.show()
    """))
    c.append(md("""
        Depth error grows with distance: with Z = fB/d, a fixed disparity error δ gives a depth error ≈ Z²δ/(fB), so the
        far background (small disparity) has the largest depth error even when its disparity error is small.

        ## 6. Sparse point cloud (synthetic geometry, labelled axes)
    """))
    c.append(code("""
        pts = stereo.sparse_point_cloud(RES["points"], RES["valid"], step=5, max_points=7000, seed=cfg["seed"])
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=pts[:, 2], cmap="magma_r", s=1)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z depth (mm)")
        ax.set_zlabel("-Y (mm)")
        ax.set_title(f"{len(pts)} reconstructed points of the SYNTHETIC scene")
        ax.view_init(elev=18, azim=-62)
        save_fig(fig, "point_cloud", "synthetic")
        plt.show()
    """))
    c.append(md("""
        ## 7. A shifted copy is not stereo
        Shifting one image by 12 px gives a perfectly "valid" constant disparity — the matcher works, but the scene has no
        depth variation to recover. This is why a single public photo can never yield real scalp depth.
    """))
    c.append(code("""
        left = SCENE["left"]
        shifted = np.zeros_like(left)
        shifted[:, :-12] = left[:, 12:]
        fake = stereo.compute_stereo(left, shifted, None, cfg, pair_meta={"pair_id": "shifted_copy"})
        d = fake["disparity_px"][fake["valid"]]
        print(
            f"shifted copy: median disparity {np.median(d):.2f} px, std {d.std():.2f} px -> a flat plane; status {fake['status']}"
        )
        print("gate reasons:", fake["gate"]["failed_checks"])
    """))
    c.append(md("""
        ## 8. Limitations
        * Synthetic only: no real scalp pair or calibration exists in the linked data, so real scalp 3-D is **unavailable**.
        * The synthetic cameras are perfectly rectified and noise is mild; real pairs add distortion, rectification error,
          glare, defocus and repetitive hair texture that breaks matching.
        * A point cloud describes reconstructed visible surfaces — not hidden follicle depth, hair volume or 3-D shaft diameter.
        * Faculty acceptance of synthetic-only stereo coverage is unconfirmed.

        ## 9. Conclusion
        The complete Lab 8 pipeline — gate, SGBM matching, fixed-point conversion, LR/border validity masks, Q-matrix depth,
        point cloud and ground-truth evaluation — runs and is quantified on a calibrated synthetic scene. Code:
        `scalptrack.stereo`, scene generator `scalptrack.synthetic.render_stereo_scene`.

        **Reference:** OpenCV camera calibration and 3-D reconstruction (spec Section 24.2, ref 6).
    """))
    return notebook(c)
