# ScalpTrack: Explainable Scalp Condition Classification and Experimental Hair Measurement

A reproducible, notebook-first computer-vision mini-project (FCV, Labs 1–10). It analyses public
hair/scalp images, predicts **dataset-defined scalp-condition labels** with an explainable classical
classifier, and investigates visible hair counting, apparent shaft width and stereo depth. Every
measurement reports its units and validity status.

> **This is not a medical tool.** Predictions are scores for publisher labels from an experimental image
> classifier. Physical widths (µm), follicle density and real scalp 3-D are reported **only** when the
> required scale, labels or calibration exist. In the linked public data they do not.

The authoritative specification is [`fcv_miniProj.pdf`](fcv_miniProj.pdf).

---

## Current status (from executed evidence, see `reports/capability_status.md`)

| Item | Status |
|---|---|
| Automated tests | **147 passed** (`pytest`, Windows 11, Python 3.13.7) |
| Notebooks 00–08 | **All executed** top-to-bottom in fresh kernels, 0 errors |
| Real dataset | **Not downloaded.** Kaggle needs your account (see [docs/dataset_setup.md](docs/dataset_setup.md)). Only the public file listing was recorded: 10 classes × 1,200 jpg |
| Real-data code paths | Executed end to end on a throwaway toy export of generated images in the Kaggle layout. That run proves the code paths only, and its outputs were discarded |
| Scalp-condition classifier | **Not trained on real data yet.** A *synthetic software-check* model exists in `models/synthetic_demo/`. Its labels are not scalp conditions |
| Hair count / width (px) | Verified on synthetic fixtures. Exploratory on real images |
| Width in µm, follicle density, real 3-D | **Unavailable.** The linked data has no scale, instance labels or stereo pairs |
| Stereo (Lab 8) | Calibrated **synthetic** scene: disparity MAE 0.20 px at 84.6 % coverage |
| Haar (Lab 9) | Implemented, with a no-face/manual-ROI path. Positive case **data-limited** |
| MobileNetV2 (P1) | Not run (TensorFlow not installed). Optional profile in `requirements-training.txt` |

---

## Experiments → notebooks

| Notebook | Labs | Owner | What it shows |
|---|---|---|---|
| `00_data_audit_basics` | 1 | Member B | source registry, publisher listing, **audit → manifest → leakage-safe grouped split**, sample grids, duplicate clusters, grouped-vs-naive split demo, BGR/RGB, pixel tables, resize comparison |
| `01_filters_color` | 2, 4 | Member A | mean/Gaussian/median/bilateral/Laplacian sharpening, MSE/PSNR vs a known clean image, alpha sweep, RGB/Gray/HSV/LAB with correct encodings, hair/scalp patch statistics |
| `02_edges_segmentation` | 3, 5 | Member A | Sobel/Prewitt (explicit kernels)/Canny, fixed/Otsu/adaptive mean & Gaussian (15/31/51), parameter selection, cleanup, hair/scalp/flake/unknown masks, coverage, measured failure cases |
| `03_texture_pyramids` | 6, 7 | Member A | GLCM (32 levels, d = 1/2/4, 4 angles) checked against scikit-image, histogram vs GLCM entropy, masked GLCM, feature CSV, Gaussian/Laplacian pyramids with exact reconstruction |
| `04_hair_measurements` | – | Member C | crossing/component counts and pixel width on known fixtures, hard cases, flake-halo filter, estimator selection, robustness, µm gating, real-image status |
| `05_classification` | 6 (ablation) | Member B | majority baseline, LogisticRegression/RandomForest selected on **validation**, one locked **test** evaluation, per-class report, confusion matrices, texture ablation, **measured leakage inflation**, grouped CV, error analysis, timing, saved bundle |
| `06_stereo_depth` | 8 | Member C | real-data gate, synthetic calibrated scene, SGBM, LR/border validity, Q-matrix depth, SGBM vs BM, point cloud, shifted-copy counter-example |
| `07_face_roi` | 9 | Member C | cascade verification, dataset face search, no-face + manual ROI, ROI errors, face blurring |
| `08_integrated_demo` | 10 | Members C + B | `analyze_image`: result panel, capability badges, JSON/CSV/PNG export, technique gallery, robustness table, separate synthetic stereo panel |

The full requirement-by-requirement mapping is in [docs/experiment_mapping.md](docs/experiment_mapping.md).

---

## Repository structure

```
ScalpTrack/
├── fcv_miniProj.pdf               specification (source of truth)
├── README.md  requirements.txt  requirements-training.txt  pyproject.toml  .gitignore
├── configs/default.yaml           every parameter, seed, path and source (no absolute paths)
├── src/scalptrack/                reusable package (all computation lives here)
│   ├── io_utils.py     load/validate images, ROI rules, resize records
│   ├── quality.py      blur / exposure / resolution flags
│   ├── preprocessing.py colour spaces, filters, MSE/PSNR
│   ├── edges.py        Sobel, Prewitt, Canny, edge summaries
│   ├── segmentation.py thresholds, cleanup, candidate masks, coverage
│   ├── texture.py      masked GLCM, histogram entropy
│   ├── pyramids.py     Gaussian/Laplacian pyramids, reconstruction
│   ├── features.py     fixed 43-feature schema (grouped for the ablation)
│   ├── measurements.py crossings, components, shaft width, µm gating
│   ├── audit.py        scanning, decoding checks, duplicates, manifests, grouped splits
│   ├── classification.py  models, selection, evaluation, bundles, predict_condition
│   ├── experiments.py  end-to-end classification run (notebook 05 + script)
│   ├── transfer.py     optional MobileNetV2 (guarded)
│   ├── stereo.py       gate, SGBM, validity, Q depth, evaluation
│   ├── face_roi.py     Haar detection, ROI proposal, face blurring
│   ├── synthetic.py    labelled verification fixtures (hairs, textures, stereo scene)
│   ├── demo_data.py    consistent notebook inputs (train/validation only)
│   ├── pipeline.py     analyze_image + capability badges
│   ├── exports.py      JSON / flattened CSV / PNG
│   ├── schema.py       statuses and result schema
│   └── viz.py          Matplotlib helpers, result panel
├── notebooks/00_…08_*.ipynb        executed notebooks (generated from scripts/nbbuild/)
├── scripts/
│   ├── download_dataset.py  audit_dataset.py  train_classifier.py  run_pipeline.py
│   ├── build_notebooks.py  run_notebooks.py  check_notebooks.py  generate_report.py
│   └── nbbuild/             notebook source definitions
├── tests/                  147 pytest tests on tiny synthetic fixtures (no download needed)
├── data/{raw,manifests,derived,synthetic}/   see data/README.md (raw data is git-ignored)
├── models/                 saved bundles (models/synthetic_demo/ = software check only)
├── outputs/{figures,results,demo}/            real/ and synthetic/ kept apart, SYNTHETIC_ prefixes
├── reports/capability_status.md               generated from evidence on disk
└── docs/  experiment_mapping.md  dataset_setup.md  limitations.md
```

---

## Setup on Windows

Prerequisites: Python 3.11–3.13 (tested with 3.13.7), Git, and about 1 GB of free disk space for the dataset.

```bash
cd C:\Riddhi_Projects\ScalpTrack
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
```

```bash
python -m pip install -e .
```

Optionally register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name scalptrack --display-name "Python (ScalpTrack)"
```

`opencv-python-headless` must stay below 5, because OpenCV 5 removed `CascadeClassifier` (Haar) from
the main wheel.

---

## Get the data (manual, your account)

Follow [docs/dataset_setup.md](docs/dataset_setup.md). In short: download Kaggle
`sundarannamalai/hair-diseases` (CC BY-NC-SA 4.0), extract it to `data\raw\kaggle_hair_diseases\`, then run:

```bash
python scripts/download_dataset.py --source kaggle_hair_diseases --record-only --version 1
```

```bash
python scripts/audit_dataset.py
```

Without data, every notebook still runs. It prints exactly where to put the files and uses clearly
labelled synthetic fixtures instead.

---

## Run

**Tests**

```bash
python -m pytest
```

**Notebooks**, in order 00 → 08. Either open them in Jupyter (`jupyter lab notebooks`) or execute all of
them headless:

```bash
python scripts/run_notebooks.py
```

```bash
python scripts/check_notebooks.py
```

**Train and evaluate** (after the audit). Use `--max-per-class 300` for a quicker CPU run:

```bash
python scripts/train_classifier.py
```

**Integrated analyzer on one image** (ROI is `x0 y0 x1 y1` in original pixels, max exclusive):

```bash
python scripts/run_pipeline.py --image "data\raw\kaggle_hair_diseases\Hair Diseases - Final\test\Psoriasis\psoriasis_0026.jpg"
```

```bash
python scripts/run_pipeline.py --synthetic-demo
```

**Evidence report**

```bash
python scripts/generate_report.py --run-tests
```

### Model training and prediction workflow

1. `audit_dataset.py` writes the manifest with exclusion reasons and `duplicate_group_id`. It keeps the
   publisher split only if no duplicate group crosses it, and otherwise re-splits about 70/15/15 by group.
2. `train_classifier.py` (or notebook 05) extracts the 43 features (cached in `data/derived/`). Inside one
   `Pipeline` it fits the imputer, scaler and model on **train**, selects the model on **validation**
   macro-F1, scores **test** once, runs the texture ablation, and saves
   `models/<source>/baseline_bundle.joblib` plus `.json` metadata (classes, feature schema, config, versions).
3. `classification.predict_condition(rgb_roi, bundle, cfg)`, `run_pipeline.py` and notebook 08 load the
   bundle. Nothing is retrained during the demo.

### Expected outputs

* `data/manifests/<source>_manifest.csv`, `_splits.csv`, `_audit_report.json`, `capability_audit.csv`
* `outputs/figures/<notebook>/{real,synthetic}/*.png` (synthetic files carry a `SYNTHETIC_` prefix and caption)
* `outputs/results/...`: verification CSVs (counts, widths, stereo), classification metrics JSON, predictions
* `outputs/demo/{real,synthetic}/<image>.json|.csv|_result_panel.png`
* `reports/capability_status.md`

---

## What was verified, and what still needs you

**Verified here:** the 147 unit and integration tests; execution of all nine notebooks; every CLI script,
including the clean exit code 2 when data is missing; and the real-data branches of every notebook and
script, run on a toy export of generated images.

**Still requires manual action:**

1. Download the Kaggle export (your account), then run `audit_dataset.py`, re-run the notebooks
   (`run_notebooks.py`) and `generate_report.py`. Only then do real class counts, duplicate statistics,
   classifier metrics and Haar positives exist.
2. Re-check the hair-mask parameters on real development images. They were selected on synthetic
   3–4 px shafts (docs/limitations.md §3).
3. Optionally, install TensorFlow from `requirements-training.txt` for the MobileNetV2 comparison.
4. Ask faculty whether synthetic-only stereo (Lab 8) and a data-limited Haar positive case (Lab 9) are
   acceptable.

Known limitations are in [docs/limitations.md](docs/limitations.md).

---

## Team (spec Section 19)

| Slot | Responsibility | Notebooks |
|---|---|---|
| Member A | classical processing and texture | 01, 02, 03 |
| Member B | data audit and condition classification | 00, 05 |
| Member C | measurement, stereo, face, integration | 04, 06, 07, 08 |

## References

OpenCV documentation (filtering, colour conversions, thresholding, pyramids, calib3d, cascade classifier,
histograms); scikit-image GLCM; TensorFlow transfer-learning tutorial. These are spec Section 24.2,
refs 1–9. Dataset: Sundar Annamalai, *Hair Diseases*, Kaggle, CC BY-NC-SA 4.0.
