# Limitations, assumptions and incomplete work

This file states what the repository does **not** establish. It is kept in sync with the executed
evidence in `reports/capability_status.md`.

## 1. Data status

* The Kaggle *Hair Diseases* export (12,000 images, verified against the publisher's listing) is
  audited and used for training and evaluation. Images stay in the git-ignored `data/raw/`.
* **The publisher split leaks.** 562 confirmed copy groups (4,032 images) crossed train/val/test, so
  ScalpTrack re-splits by copy group. 151 images that are confirmed copies of an image with a
  *different* label are excluded.
* **Residual leakage remains.** The detector confirms flip/shift/zoom copies (dHash candidates →
  thumbnail correlation → ORB geometric verification). It does not find copies made by arbitrary
  rotation, colour jitter or heavy cropping. On the real split a 1-nearest-neighbour lookup scores 0.973
  test accuracy, above the selected random forest (0.951). The test score is therefore an **upper
  bound** for this dataset, not evidence of generalisation to new patients.
* The synthetic model in `models/synthetic_demo/` is a software check only. Its labels are not scalp
  conditions.

## 2. Classification

* The labels are publisher folder labels with no clinical adjudication. This is an experimental image
  classifier, **not a diagnostic tool**. Its scores are model scores, not calibrated probabilities.
* No patient identity is available. Grouping by exact or near-duplicate cluster prevents copy leakage,
  but not two different photos of the same person landing in different splits.
* Near-duplicate detection (dHash under 8 flips/rotations, confirmed by 32×32 thumbnail correlation
  ≥ 0.9) misses crops, colour jitter, strong rescaling and other augmentations. Clusters should be
  reviewed visually. A dHash match alone was shown to merge distinct look-alike textures, which is why
  the verification step exists.
* Colour and texture features can pick up camera, compression or lighting differences between
  publisher folders instead of condition-related appearance.
* The app's model is a frozen ImageNet MobileNetV2 (PyTorch) with a linear head (`scripts/train_deep.py`).
  It reaches 0.55 test accuracy on the strict copy-grouped split. Grad-CAM and VGG19 (P2) were **not
  run**, and the TensorFlow path in `scalptrack.transfer` is guarded and untested.
* The **healthy class** is learned from 9 user-added photos, some possibly of the same person, and is
  evaluated only by leave-one-out (6/9 recognised). It is a rough indication, not a reliable healthy check.
* The match level (5%) is chosen for high recall: the true condition is reported for 95% of validation
  photos and 84% of test photos. The price is that several conditions often pass the level together.

## 3. Measurements

* Counts are **exploratory**. `reference_line_crossings` and `isolated_strand_candidates` are verified
  only on synthetic fixtures. They are not follicle counts, and density per cm² is unavailable.
* Known failure modes are demonstrated, not hidden: crossing or touching shafts merge into one component,
  fragmented masks over-count, curling shafts cross the line several times, and a line can fall into a
  mask gap. Shafts closer than the 2 px merge gap are merged on the line.
* Adaptive thresholding creates dark halos around bright flakes. Components that are both compact
  (elongation < 3) and short (skeleton < 40 px) are excluded as non-shaft. This rule also removes
  genuinely short, compact hair fragments, and it was tuned on synthetic images.
* Width is an **apparent 2-D width in native pixels**. The estimator offset (0, the spec's plain
  `2·d`) was selected by lowest MAE on synthetic fixtures. A width-parity and orientation bias of up to
  1 px remains. The result depends on the threshold method and its parameters.
* **Micrometres are never produced** without a documented scale and its provenance. The linked data
  supplies no scale.
* The hair-mask parameters (adaptive Gaussian, block 15, C 8, closing 3) were selected on synthetic
  3–4 px shafts. Real close-ups with wider shafts need re-selection on development images, with a block
  of at least about 3× the apparent shaft width.
* The blur gate (Laplacian variance ≥ 60) is empirical. Image noise inflates it, so blurred but noisy
  images can pass. This was shown in notebook 04.

## 4. Segmentation

* No ground-truth masks exist for public images. Real overlays are qualitative. Precision/recall is
  reported only against synthetic truth.
* Global dark thresholds fail on shadows and light/grey hair. Brightness-based flake candidates fire on
  glare and on bright noise. Coverage fractions are pixel-area descriptors, not clinical dandruff
  percentages.

## 5. Stereo / 3-D

* **Real scalp 3-D is unavailable.** There is no verified stereo pair or calibration in the linked data.
  The Lab 8 demonstration uses a synthetic, perfectly rectified scene. Faculty acceptance of
  synthetic-only coverage is unconfirmed.
* The real-pair path (`stereo.rectify_pair` with `cv2.stereoRectify`) is implemented but has not been
  exercised on real calibrated images.
* Depth error grows roughly with Z², and the evaluation reports coverage next to error.

## 6. Haar / ROI

* No positive face example has been verified. The dataset has not been downloaded, and scalp close-ups
  are not expected to contain faces. Lab 9 positive detection is **data-limited**.
* The upper-head crop is a heuristic proposal that always needs manual confirmation.

## 7. Environment

* Tested on Windows 11, Python 3.13.7, CPU only. Pinned versions are in `requirements.txt`.
  `opencv-python-headless` must stay below 5, because OpenCV 5 removed `CascadeClassifier` from the main
  wheel.
* Notebook widgets are skipped during headless execution (`scripts/run_notebooks.py`) and work in an
  interactive Jupyter session. The interactive widget path was not exercised in this environment.
* RandomForest (`n_jobs=-1`) is seeded. Thread scheduling does not change its result, but very small
  floating-point differences across platforms are possible.

## 8. Out of scope (per the specification)

Scalp health index, treatment or serum recommendations, longitudinal tracking, general skin analysis and
cloud hosting. (A local Streamlit frontend was added later, in `app/`.)
