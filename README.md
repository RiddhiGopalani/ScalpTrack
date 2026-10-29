# ScalpTrack

**Explainable scalp-condition classification and experimental hair measurement with classical computer vision**

ScalpTrack is a reproducible, notebook-first computer-vision project (FCV mini-project, Labs 1–10). It
classifies scalp photographs into ten dataset-defined conditions, explains *why* with interpretable
colour, edge and texture features, and investigates hair counting, hair-shaft width and stereo depth.
Every number is labelled with what it means and how far it can be trusted.

> **Not a medical device.** Predictions are scores for the publisher's folder labels, produced by an
> experimental classifier. They are not diagnoses.

**Quick start:** see [HOW_TO_RUN.md](HOW_TO_RUN.md).

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Solution](#2-solution)
3. [What makes it different (USP)](#3-what-makes-it-different-usp)
4. [Results on the real dataset](#4-results-on-the-real-dataset)
5. [Architecture](#5-architecture)
6. [What happens in each notebook](#6-what-happens-in-each-notebook)
7. [Technologies](#7-technologies)
8. [Repository structure](#8-repository-structure)
9. [Limitations](#9-limitations)
10. [Team and references](#10-team-and-references)

---

## 1. Problem statement

Hair and scalp conditions such as alopecia, psoriasis, folliculitis, seborrheic dermatitis and lice are
common, and they look alike to an untrained eye. Public image datasets exist, but projects built on them
often share three problems:

* **Black-box predictions.** A deep network outputs a label, with no visible evidence for it.
* **Inflated accuracy.** Public datasets are heavily augmented. Flipped or shifted copies of one photo
  end up in both the training and the test set, so a model scores highly by memorising.
* **Unsupported measurements.** Hair "density in hairs/cm²" or "thickness in µm" gets reported from
  ordinary photos that contain no physical scale.

**Goal:** develop a reproducible computer-vision system that analyses public hair/scalp images,
predicts their dataset-defined condition labels, visualises the image-processing evidence behind each
prediction, and reports hair and 3-D measurements only in the units the data actually supports. Along
the way it applies the whole CV lab syllabus: filtering, colour spaces, edges, thresholding, texture,
pyramids, stereo and Haar detection.

## 2. Solution

ScalpTrack is a Python package (`src/scalptrack/`) and nine Jupyter notebooks (Labs 1–10). They form a
pipeline from raw images to an explained result:

1. **Audit the data first.** Every image is decoded and checked. Augmented copies are found with a
   three-stage detector (perceptual hash → thumbnail correlation → ORB geometric verification), copies
   that carry conflicting labels are excluded, and the data is re-split so that **no group of copies
   crosses train/validation/test**.
2. **Explainable features.** Classical operators (Gaussian/Laplacian filters, Sobel/Prewitt/Canny,
   HSV/LAB statistics, GLCM texture, Laplacian-pyramid energies) produce a fixed, named 43-feature
   vector.
3. **An honest classifier.** A majority baseline, logistic regression and a random forest are compared
   on the **validation** set. The chosen model is scored **once** on the test set. Two extra checks
   follow: a texture ablation (with vs without texture features) and a memorisation check (1-nearest
   neighbour).
4. **Measurements with validity gates.** Hair counts are named precisely ("reference-line crossings",
   not "hairs"). Shaft width is given in pixels; micrometres are reported only with a documented scale.
   Every algorithm is verified against synthetic images with known truth.
5. **One integrated analyzer.** `analyze_image()` takes an image and an optional region and returns
   a result panel, capability badges and JSON/CSV/PNG exports. Missing inputs give a clear status
   instead of a crash.

## 3. What makes it different (USP)

| | Typical public-dataset project | ScalpTrack |
|---|---|---|
| Data leakage | Uses the publisher's split as-is | Detects augmented copies across 12,000 images. **34% of images** sat in copy groups that crossed the publisher's split, so the data was re-split by group |
| Label quality | Trusts the folder labels | Finds the *same photo* filed under two diseases and excludes it (151 images) |
| Explainability | A label from a CNN | Named features, ablation, per-class errors and a visual technique gallery |
| Accuracy claims | Headline number | Majority baseline, a locked test set, and a memorisation check that flags residual leakage |
| Measurements | Reports µm / hairs per cm² from plain photos | Pixel units with a validity status; physical units only with a documented scale |
| 3-D | Implied from single photos | A real-data gate; stereo is shown on a calibrated synthetic scene with ground truth |
| Reproducibility | Ad-hoc notebook cells | Tested package (150+ tests), one YAML config, fixed seeds, saved model bundles, notebooks re-executable in one command |

## 4. Results on the real dataset

Dataset: Kaggle **Hair Diseases** (`sundarannamalai/hair-diseases`, v1, CC BY-NC-SA 4.0). It has 12,000
JPEGs, ten classes and 1,200 images per class. All files were verified byte-for-byte against the
publisher's listing.

### 4.1 Data audit (notebook 00)

| Check | Result |
|---|---|
| Unreadable / unlabelled files | 0 / 0 |
| Confirmed near-duplicate pairs (flip/shift/zoom copies) | 9,451, forming 1,218 copy groups |
| Copy groups crossing the publisher's train/val/test | **562 groups, 4,032 images** → publisher split rejected |
| Same photo under two different labels | 285 pairs → **151 images excluded** |
| Final split (grouped, about 70/15/15) | 11,849 usable images: train 8,240 · validation 1,811 · test 1,798 |

### 4.2 Classification (notebook 05)

| Model (selected on validation) | Validation macro-F1 | Test accuracy | Test macro-F1 | Balanced accuracy |
|---|---|---|---|---|
| Majority-class baseline | 0.018 | 0.099 | 0.018 | 0.100 |
| Logistic regression (best C) | 0.665 | not scored (not selected) | — | — |
| **Random forest (selected)** | **0.978** | **0.951** | **0.951** | **0.954** |

Per-class test F1 ranges from 0.87 (alopecia areata) to 0.997 (lichen planus, male-pattern baldness).
The texture ablation changes test macro-F1 by less than 0.001 (0.951 with texture vs 0.952 without).

**How to read the 0.95.** A 1-nearest-neighbour lookup on the same features reaches **0.973** test
accuracy, higher than the forest. Simple matching of 43 global statistics should not separate ten
visually similar conditions this well. The likely explanation is that the dataset was built from fewer
original photos with heavy augmentation, and copies made by arbitrary rotation, colour jitter or
cropping are not caught by the duplicate detector. **Treat 0.95 as an upper bound for this dataset, not
as evidence that the model generalises to new patients.**

### 4.3 Verified algorithms (synthetic ground truth)

| Module | Result |
|---|---|
| Hair counting (separated shafts) | Exact on 100% of 29 fixtures (components and line crossings) |
| Apparent shaft width | Mean absolute error 0.31 px, max 1.0 px, over 56 width/orientation fixtures |
| Stereo depth (SGBM + calibrated Q) | Disparity error 0.20 px at 84.6% coverage; depth MAE 6.1 mm |
| GLCM texture | Matches scikit-image exactly |
| Pyramid reconstruction | Exact (error at machine precision) |

All current numbers, generated from files on disk, are in
[reports/capability_status.md](reports/capability_status.md).

## 5. Architecture

```mermaid
flowchart TD
    A[Raw Kaggle images<br/>data/raw] --> B[Audit<br/>decode · dHash · thumbnail · ORB<br/>label conflicts · grouped split]
    B --> M[(Manifest<br/>data/manifests)]
    M --> C[Quality gate<br/>blur · exposure · ROI size]
    C --> D[Preprocessing<br/>filters · HSV/LAB]
    D --> E[Edges + segmentation<br/>Sobel · Prewitt · Canny<br/>Otsu · adaptive · masks]
    D --> F[Features<br/>43 named: colour · edge<br/>GLCM · histogram · pyramid]
    F --> G[Classifier<br/>baseline · LogReg · RF<br/>select on val · test once]
    G --> K[(Model bundle<br/>models/)]
    E --> H[Measurements<br/>crossings · components<br/>width in px · µm gate]
    S[Stereo pair + calibration] --> T[Stereo<br/>gate · SGBM · LR check · Q depth]
    I[Full image] --> J[Haar face detection<br/>manual ROI fallback]
    K --> P[analyze_image]
    H --> P
    T --> P
    J --> P
    C --> P
    P --> R[Result panel · badges<br/>JSON / CSV / PNG]
```

**Design principles**

* **All computation lives in the package.** Notebooks only present results, so every algorithm is unit-tested.
* **One configuration file.** `configs/default.yaml` holds every threshold, kernel, seed and path.
* **Explicit statuses.** Every output is `ok`, `exploratory`, `synthetic_demo`, `unavailable`, `failed`
  or `not_requested`. An unknown value is `null`, never `0`.
* **Real and synthetic outputs are kept apart.** They go to separate folders, and synthetic files carry
  a `SYNTHETIC_` prefix and caption.

## 6. What happens in each notebook

Run them in order. Each one starts with its objective, the labs and requirements it covers, and its
inputs, and ends with limitations and a conclusion.

| # | Notebook | Labs | What it does | Key outputs |
|---|---|---|---|---|
| 00 | `00_data_audit_basics` | 1 | Registers the sources and runs the full audit (corrupt files, sizes, duplicates, label conflicts, grouped split). Shows class distributions, sample grids and duplicate clusters. Demonstrates image reading/display, BGR vs RGB, pixel tables and resizing methods, and why measurement needs native resolution | manifest, split file, audit report, capability table |
| 01 | `01_filters_color` | 2, 4 | Mean, Gaussian, median and bilateral filters and Laplacian sharpening on the same image; MSE/PSNR against a known clean image; RGB/gray/HSV/LAB channels with their correct numeric encodings; hair vs scalp colour statistics | filter contact sheets, PSNR table, patch statistics CSV |
| 02 | `02_edges_segmentation` | 3, 5 | Sobel, Prewitt (explicit kernels) and Canny at three threshold pairs. Fixed, Otsu, adaptive-mean and adaptive-Gaussian thresholds with parameter selection; mask clean-up; hair/scalp/flake/unknown candidate masks. Measured failure cases: shadow, glare, light hair, blur | edge and threshold overlays, failure-case table |
| 03 | `03_texture_pyramids` | 6, 7 | GLCM contrast, energy, entropy, homogeneity and correlation (checked against scikit-image); histogram vs GLCM entropy; masked GLCM; feature distributions per class; Gaussian/Laplacian pyramids with exact reconstruction | feature CSV, distribution plots, pyramid figures |
| 04 | `04_hair_measurements` | — | Hair counting by connected components and reference-line crossings; shaft width by distance transform; verification on synthetic shafts, including hard cases (crossing, touching, broken, curly); flake-halo filter; robustness to blur and noise; µm gating; exploratory overlays on real images | verification CSVs, count and width overlays |
| 05 | `05_classification` | 6 (ablation) | Trains on the real dataset: label checks, grouped split, majority baseline, model selection on validation, a single locked test evaluation, per-class report, confusion matrices, texture ablation, synthetic leakage experiment, **1-NN memorisation check**, grouped cross-validation, error gallery, timing, saved model | model bundle, `metrics.json`, confusion matrices |
| 06 | `06_stereo_depth` | 8 | Real-data gate (no real stereo pairs exist). Renders a calibrated synthetic 3-D scene, then runs StereoSGBM, left-right consistency checks, Q-matrix depth and a point cloud, and evaluates against ground truth. Also shows why a shifted copy of one photo is not stereo | evaluation table, depth and point-cloud figures |
| 07 | `07_face_roi` | 9 | Loads the Haar frontal-face cascade, searches the real dataset for faces, proposes an upper-head region (always needs confirmation), and supports a manual ROI. Faces are blurred in saved figures | face-search summary, ROI overlays |
| 08 | `08_integrated_demo` | 10 | The examiner-facing application: loads the saved model, analyses a test image end to end, and shows the result panel with capability badges, a technique gallery, a robustness table and a separate synthetic stereo panel | `outputs/demo/real/*.json, *.csv, *.png` |

## 7. Technologies

| Area | Tools |
|---|---|
| Language | Python 3.11–3.13 |
| Computer vision | OpenCV 4.x (filters, edges, thresholds, pyramids, ORB, StereoSGBM, Haar cascades), scikit-image (skeletons, GLCM check), SciPy, Pillow |
| Machine learning | scikit-learn (Pipeline, StandardScaler, LogisticRegression, RandomForest, metrics, StratifiedGroupKFold) |
| Data and plots | NumPy, pandas, Matplotlib |
| Notebooks | Jupyter, nbformat/nbclient (headless execution), ipywidgets (optional controls) |
| Quality | pytest (150+ tests), ruff (lint and format), YAML configuration, joblib model bundles |
| Optional | TensorFlow / MobileNetV2 (`requirements-training.txt`, not used for the reported results) |

## 8. Repository structure

```
ScalpTrack/
├── README.md · HOW_TO_RUN.md · requirements.txt · pyproject.toml
├── configs/default.yaml          every parameter, seed and path
├── src/scalptrack/               the package (all computation)
│   ├── io_utils · quality · preprocessing · edges · segmentation      Labs 1–5
│   ├── texture · pyramids · features                                  Labs 6–7, classifier inputs
│   ├── audit · classification · experiments · transfer                data and models
│   ├── measurements · stereo · face_roi                               hair, Lab 8, Lab 9
│   ├── pipeline · exports · viz · schema                              Lab 10 analyzer and outputs
│   └── synthetic · demo_data · config                                 fixtures and helpers
├── notebooks/00…08_*.ipynb       executed notebooks (sources in scripts/nbbuild/)
├── scripts/                      download · audit · train · run_pipeline · run/check/build notebooks · report
├── tests/                        unit and integration tests (no download needed)
├── data/  raw/ (git-ignored) · manifests/ · derived/ · synthetic/
├── models/                       saved model bundles
├── outputs/  figures/ · results/ · demo/
├── reports/capability_status.md  generated evidence summary
└── docs/                         experiment mapping · dataset setup · limitations · Member C migration
```

To read the code, start with `configs/default.yaml` and `schema.py`, then the lab modules in lab order,
then `pipeline.analyze_image`, which is written as eight numbered steps.
[docs/experiment_mapping.md](docs/experiment_mapping.md) maps every lab and requirement to code.

## 9. Limitations

* **Residual augmentation leakage.** Copies made by arbitrary rotation, colour jitter or cropping are not
  detected, so the test score is an upper bound (see §4.2). There are no patient identities, so
  different photos of the same person can still fall into different splits.
* **Labels are the publisher's folders**, without clinical adjudication. Some classes overlap visually.
* **Hair counts and widths on real photos are exploratory.** There are no count labels and no physical
  scale, so µm widths, follicle counts and density are reported as unavailable.
* **Real scalp 3-D is unavailable.** The data has no stereo pairs; Lab 8 is shown on a synthetic scene.
* **Haar detects frontal faces only.** Most images are scalp close-ups, where finding no face is the
  correct result.

The full list is in [docs/limitations.md](docs/limitations.md).

## 10. Team and references

| Member | Responsibility | Notebooks |
|---|---|---|
| Member A | Classical processing and texture | 01, 02, 03 |
| Member B | Data audit and condition classification | 00, 05 |
| Member C | Measurement, stereo, face detection, integration | 04, 06, 07, 08 |

**Dataset:** Sundar Annamalai, *Hair Diseases*, Kaggle, CC BY-NC-SA 4.0 (non-commercial, attribution
required). Images are not redistributed in this repository.
**References:** OpenCV documentation (filtering, colour conversion, thresholding, pyramids, calib3d,
cascade classifier); scikit-image GLCM; Rublee et al., *ORB: an efficient alternative to SIFT or SURF*
(ICCV 2011). The specification is the course PDF `fcv_miniProj.pdf`, which is not included in this
repository.
