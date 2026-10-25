# Experiment and requirement mapping

The table maps every lab from spec Section 5.2, and every requirement from Section 5.1, to the code
and notebook that implements it. "Evidence" is what the executed notebook actually produces.
`(S)` marks evidence produced on synthetic fixtures because the real dataset has not been downloaded yet.

## Labs 1–10

| Lab | Spec technique | Notebook (owner) | Package code | Evidence produced |
|---|---|---|---|---|
| 1 | Reading, displaying, resizing, grayscale, pixel analysis | `00_data_audit_basics` (B) | `io_utils.load_image`, `describe_array`, `pixel_neighbourhood`, `compare_resize_methods`, `resize_with_record` | BGR-vs-RGB panel, dtype/range table, 5×5 pixel table, histograms, nearest/bilinear/area comparison, width-after-resize table |
| 2 | Mean filter, Gaussian blur, Laplacian sharpening | `01_filters_color` (A) | `preprocessing.compare_filters`, `mean_filter`, `gaussian_filter`, `signed_laplacian`, `laplacian_sharpen` (+ median, bilateral as extras) | contact sheets, 60×60 zooms, parameter table, downstream gradient/Otsu table, MSE/PSNR table vs a known clean image (S), alpha sweep (S) |
| 3 | Sobel, Prewitt, Canny | `02_edges_segmentation` (A) | `edges.sobel_gradients`, `prewitt_gradients` (explicit kernels via `filter2D`), `canny`, `compare_edges`, `edge_summary` | signed gradient maps, kernels, 3 Canny threshold pairs, edge fraction/fragmentation CSV, "two edges per shaft" profile |
| 4 | RGB → grayscale, HSV, LAB | `01_filters_color` (A) | `preprocessing.to_gray/to_hsv/to_lab_uint8/to_lab_float/lab_uint8_to_signed/channel_stack/patch_statistics` | channel panels with native scales, uint8 vs float LAB check, hair/scalp patch-statistics CSV, region histograms, glare-vs-flake table |
| 5 | Global and adaptive thresholding | `02_edges_segmentation` (A) | `segmentation.threshold_fixed/threshold_otsu/threshold_adaptive/compare_thresholds/cleanup_mask/segment_candidates/coverage_descriptors` | fixed/Otsu/adaptive mean/Gaussian (block 15/31/51) overlays, parameter-selection CSV, cleanup before/after, candidate masks, coverage table, failure-case table with known truth (S) |
| 6 | Statistical texture: contrast, energy, entropy | `03_texture_pyramids` (A); used in `05` | `texture.glcm/glcm_properties/glcm_features/histogram_entropy/interior_patches`, `features.extract_features` | per-offset GLCM table, check against scikit-image, histogram-vs-GLCM entropy demo, masked-GLCM demo, feature CSV, distribution plots, cost table; **texture ablation** in notebook 05 |
| 7 | Gaussian and Laplacian pyramids | `03_texture_pyramids` (A) | `pyramids.gaussian_pyramid/laplacian_pyramid/reconstruct/reconstruction_error/band_energies` | level figures (odd sizes), signed residuals, reconstruction error table, band energy by shaft width (S) |
| 8 | Stereo disparity and depth | `06_stereo_depth` (C) | `stereo.validate_stereo_inputs/compute_stereo/q_matrix_rectified/rectify_pair/left_right_consistency/evaluate_against_truth`, `synthetic.render_stereo_scene` | real-data gate status JSON, scene provenance JSON, disparity/validity/depth/error figures, Q check, SGBM vs StereoBM table, per-surface table, point cloud, shifted-copy counter-example (all S) |
| 9 | Haar cascade face detection | `07_face_roi` (C) | `face_roi.load_cascade/detect_faces/propose_upper_head_roi/choose_roi/blur_faces` | cascade verification, dataset face search summary, no-face + manual ROI panels, ROI error handling; positive case **data-limited** until data exists |
| 10 | Integrated CV application | `08_integrated_demo` (C, B) | `pipeline.analyze_image/capability_badges`, `exports.export_result`, `viz.result_panel` | result panel, capability badges, JSON/CSV/PNG under `outputs/demo/`, technique gallery, robustness table, separate synthetic stereo panel |

Owner slots follow spec Section 19: **A** = classical processing and texture, **B** = data audit and
classification, **C** = measurement, stereo, face and integration.

## Additional project experiments

| Topic | Notebook | Code | Evidence |
|---|---|---|---|
| Data audit and manifests (P0) | `00` | `audit.build_manifest/save_manifest/assign_splits/duplicate_groups/capability_audit` | publisher listing table (real metadata), manifest/split/report files (after download), grouped-vs-naive split demo (S) |
| Hair counting (REQ-05) | `04` | `measurements.estimate_visible_hairs/component_table/perpendicular_line/count_runs` | count verification CSV (S), challenging-fixture table (S), numbered overlays, real-image status table |
| Shaft width (REQ-06/07) | `04` | `measurements.estimate_shaft_width/width_samples/to_physical_width` | width verification CSV and offset selection (S), orientation bias plot (S), blur/noise/contrast robustness (S), threshold sweep (S), µm gating demo |
| Classification (REQ-04) | `05` | `classification.*`, `experiments.run_experiment` | validation selection table, locked test metrics, majority baseline, per-class report, confusion matrices, texture ablation, measured leakage inflation (S), grouped CV, uncertainty curve, misclassified gallery, timing, saved bundle |
| MobileNetV2 (P1) | `05` §10 | `transfer` | status only: TensorFlow not installed, so not run |

## Requirements REQ-01 to REQ-12

| ID | Where it is satisfied |
|---|---|
| REQ-01 load/validate, keep geometry | `io_utils.load_image` (EXIF, RGB, corrupt/unsupported/empty checks), `validate_roi`, `crop_roi`; tests in `test_io_utils.py` |
| REQ-02 every method of Labs 1–7 | notebooks 00–03 (table above) |
| REQ-03 documented numerical features | `features.FEATURE_GROUPS` (43 named features), notebook 03 CSV and distributions |
| REQ-04 classifier with appropriate split | `audit.assign_splits` (grouped), `classification.select_on_validation` / `final_test_evaluation`, notebook 05 |
| REQ-05 exploratory hair counting | `measurements.estimate_visible_hairs`, notebook 04 |
| REQ-06 apparent shaft width (px) | `measurements.estimate_shaft_width`, notebook 04 |
| REQ-07 physical units only with a scale | `measurements.to_physical_width` (refuses without a scale, without provenance, or for non-isotropic resizes) |
| REQ-08 stereo disparity and calibrated depth | `stereo.compute_stereo`, notebook 06 (synthetic) |
| REQ-09 Haar with a no-face/manual ROI path | `face_roi`, notebook 07 |
| REQ-10 integrated results and exports | `pipeline.analyze_image`, `exports.export_result`, notebook 08, `scripts/run_pipeline.py` |
| REQ-11 explicit missing/unsupported status | `schema.Status` (`ok`/`exploratory`/`synthetic_demo`/`unavailable`/`failed`/`not_requested`), null-not-zero rule, badges |
| REQ-12 reproducibility and saved model | `configs/default.yaml`, fixed seed, `requirements.txt` (tested pins), model bundle with schema/classes/config/versions, `scripts/run_notebooks.py` |

## Spec Section 6.2 interface names → implementation

| Spec interface | Implementation |
|---|---|
| `load_image(path)` | `scalptrack.io_utils.load_image` |
| `validate_roi(image_shape, roi)` | `scalptrack.io_utils.validate_roi` |
| `assess_quality(rgb_roi, config)` | `scalptrack.quality.assess_quality` |
| `compare_filters(gray_roi, config)` | `scalptrack.preprocessing.compare_filters` |
| `compare_edges(gray_roi, config)` | `scalptrack.edges.compare_edges` |
| `segment_candidates(rgb_roi, config)` | `scalptrack.segmentation.segment_candidates` |
| `extract_features(rgb_roi, config)` | `scalptrack.features.extract_features` |
| `estimate_visible_hairs(rgb_roi, hair_mask, config)` | `scalptrack.measurements.estimate_visible_hairs` |
| `estimate_shaft_width(hair_mask, config)` | `scalptrack.measurements.estimate_shaft_width` |
| `predict_condition(rgb_roi, model_bundle)` | `scalptrack.classification.predict_condition` (also takes `cfg`) |
| `detect_faces(rgb_image, cascade, config)` | `scalptrack.face_roi.detect_faces` |
| `compute_stereo(left, right, calibration, config)` | `scalptrack.stereo.compute_stereo` |
| `analyze_image(path, roi, model_bundle, config)` | `scalptrack.pipeline.analyze_image` |
| `export_result(result, output_dir)` | `scalptrack.exports.export_result` |

## Deviations from the spec's proposed layout (Section 18.1)

* `detailed_spec.md` is not duplicated. The original `fcv_miniProj.pdf` is kept as the specification.
* Notebook sources live in `scripts/nbbuild/`. `python scripts/build_notebooks.py` regenerates
  `notebooks/*.ipynb`, and `python scripts/run_notebooks.py` executes them.
* Real and synthetic outputs are separated as `outputs/figures/<notebook>/{real,synthetic}/`, with
  `SYNTHETIC_` filename prefixes, and as `models/synthetic_demo/` versus `models/<source>/`.
