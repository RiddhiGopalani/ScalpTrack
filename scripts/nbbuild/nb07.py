from .common import code, md, notebook, save_cell_fn, setup_cell, status_cell

TITLE = "07_face_roi"


def build():
    c = []
    c.append(md("""
        # 07 · Haar face detection and region of interest

        **Owner:** Member C · **Lab:** 9 (Haar cascade face detection) · **Requirement:** REQ-09; spec Section 13.

        ### What Haar can and cannot do here
        * It detects **frontal faces** in full-head photographs and can suggest a rough upper-head crop.
        * It does not standardise pose, detect hairs, or find the scalp in a trichoscopic close-up — most linked images are
          close-ups, where "no face" is the expected, correct result.
        * Scalp analysis always needs a **manually confirmed ROI**; the face-based crop is only a proposal.
        * A positive example is included only if one exists in the approved linked data and is detected. No unrelated face
          dataset is imported.

        **Privacy:** saved figures blur detected face boxes by default; only box coordinates are exported.
    """))
    c.append(setup_cell(TITLE, """
        import cv2, json
        from scalptrack import face_roi
        from scalptrack.io_utils import load_image, validate_roi, crop_roi, ROIError
        from scalptrack.audit import load_manifest
        from scalptrack.exports import write_json
    """))
    c.append(save_cell_fn())
    c.append(status_cell())
    c.append(md("## 1. Load and verify the cascade"))
    c.append(code("""
        CASCADE = face_roi.load_cascade()
        print("cascade loaded from", Path(cv2.data.haarcascades).name, "| empty():", CASCADE.empty())
        print("detectMultiScale parameters:", cfg["face"])
        BLUR_FACES_IN_FIGURES = True
    """))
    c.append(md("""
        ## 2. Search the approved linked data for face-containing images
        Every audited train/validation image (or a capped sample) is scanned. Test images are not used for development.
    """))
    c.append(code("""
        found, scanned = [], 0
        if REAL:
            man = load_manifest(STATUS["manifest_path"])
            cand = man[man["split"].isin(["train", "validation"]) & man["roi_id"].isna()]
            cand = cand.sample(min(len(cand), 1500), random_state=cfg["seed"])
            base = ROOT / cfg["datasets"]["sources"][STATUS["source"]]["dir"]
            for r in cand.itertuples(index=False):
                rgb = load_image(base / r.relative_path).rgb
                det = face_roi.detect_faces(rgb, CASCADE, cfg); scanned += 1
                if det["n_faces"]:
                    found.append({"image_id": r.image_id, "relative_path": r.relative_path, "label": r.raw_label, "boxes_xywh": det["boxes_xywh"]})
        positives = pd.DataFrame(found)
        summary = {"images_scanned": scanned, "images_with_detections": len(found),
                   "positive_case_status": ("available (detections still need visual confirmation: Haar false positives occur)"
                                            if found else ("data-limited: no face detected in the scanned linked images" if REAL
                                                           else "data-limited: dataset not downloaded"))}
        write_json(summary | {"detections": found[:50]}, project_path(cfg, "results", NB, "face_search_summary.json"))
        print(json.dumps(summary, indent=2))
        positives.head(10) if len(positives) else None
    """))
    c.append(code("""
        if found:
            show = found[:4]
            fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.2), squeeze=False)
            for ax, f in zip(axes[0], show):
                rgb = load_image(base / f["relative_path"]).rgb
                prop = face_roi.propose_upper_head_roi(rgb.shape, max(f["boxes_xywh"], key=lambda b: b[2] * b[3]))
                vis = face_roi.blur_faces(rgb, f["boxes_xywh"]) if BLUR_FACES_IN_FIGURES else rgb
                ax.imshow(face_roi.draw_boxes(vis, f["boxes_xywh"], roi_xyxy=prop["roi_xyxy"])); ax.axis("off")
                ax.set_title(f"{f['label'][:20]}: {len(f['boxes_xywh'])} face(s)\\nyellow = UNCONFIRMED upper-head proposal", fontsize=8)
            plt.tight_layout(); save_fig(fig, "haar_positive_examples", "real"); plt.show()
        else:
            print("No positive detections to display - positive-case verification is data-limited.")
    """))
    c.append(md("""
        ## 3. The no-face path on a scalp close-up and the manual ROI
        On a close-up the detector correctly returns zero faces; the analysis continues through a manual ROI.
    """))
    c.append(code("""
        closeups = [(d["rgb"], d["image_id"], "real") for d in DEV[:2]] + [(SYN["flakes"]["rgb"], "closeup", "synthetic")]
        rows = []
        for rgb, name, prov in closeups:
            det = face_roi.detect_faces(rgb, CASCADE, cfg)
            h, w = rgb.shape[:2]
            manual = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)  # the examiner's chosen ROI (here: central half)
            choice_auto = face_roi.choose_roi(rgb.shape, det)
            choice_manual = face_roi.choose_roi(rgb.shape, det, manual)
            rows.append({"image": name, "provenance": prov, "faces": det["n_faces"], "detector_status": det["status"],
                         "reason": det["reason"], "auto_roi_source": choice_auto["source"], "manual_roi": choice_manual["roi_xyxy"]})
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].imshow(face_roi.draw_boxes(rgb, det["boxes_xywh"], roi_xyxy=choice_manual["roi_xyxy"])); axes[0].axis("off")
            axes[0].set_title(f"{name}: faces={det['n_faces']} -> manual ROI (yellow)", fontsize=9)
            axes[1].imshow(crop_roi(rgb, choice_manual["roi_xyxy"])); axes[1].axis("off"); axes[1].set_title("native-resolution ROI crop", fontsize=9)
            plt.tight_layout(); save_fig(fig, f"no_face_manual_roi_{name}", prov); plt.show()
        pd.DataFrame(rows)
    """))
    c.append(md("## 4. Error handling: invalid ROIs and images too small for detection"))
    c.append(code("""
        rgb = SYN["clean"]["rgb"]
        for roi in [(0, 0, 50, 50), (-5, 0, 50, 50), (10, 10, 10, 40), (0, 0, 999, 20)]:
            try:
                print(roi, "->", validate_roi(rgb.shape, roi))
            except ROIError as exc:
                print(roi, "-> ROIError:", exc)
        print(face_roi.detect_faces(rgb[:30, :30], CASCADE, cfg))
        print(face_roi.propose_upper_head_roi(rgb.shape, (60, 5, 80, 80)))
    """))
    c.append(md("""
        ## 5. Limitations
        * Haar frontal-face detection fails on profiles, tilted heads, occlusion by hair and poor lighting, and gives false
          positives on textured regions — detections must be reviewed.
        * The upper-head proposal is a geometric heuristic above the face box, clipped at the image top; it is not scalp
          segmentation and is never used without confirmation.
        * Positive-detection accuracy is not evaluated (no face labels); if no linked image contains a face, Lab 9 positive
          verification remains data-limited and is disclosed as such.

        ## 6. Conclusion
        The cascade loads and is verified, detection parameters are documented, the no-face path and manual ROI work, ROI
        errors are explicit, and saved figures blur faces. Code: `scalptrack.face_roi`.

        **Reference:** OpenCV cascade classifier tutorial (spec Section 24.2, ref 7).
    """))
    return notebook(c)
