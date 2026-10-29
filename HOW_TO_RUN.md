# How to run ScalpTrack

Step-by-step instructions for a Windows machine. The commands also work on macOS/Linux; there, use
`source .venv/bin/activate` instead of `.venv\Scripts\activate`.

Expected total time on a typical laptop CPU: about **25–35 minutes** the first time (most of it is
feature extraction on 12,000 images). Later runs reuse cached features and take a few minutes.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11–3.13 | Tested with 3.13.7. Check with `python --version` |
| Git | To clone the repository |
| ~2 GB free disk | ~600 MB dataset, plus the environment and outputs |
| Kaggle account | Only to download the dataset (free) |

No GPU is needed.

## 2. Get the code and create the environment

```bash
git clone https://github.com/RiddhiGopalani/ScalpTrack.git
```

```bash
cd ScalpTrack
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

Check the installation (it should finish with "passed" and no failures):

```bash
python -m pytest
```

> Keep `opencv-python-headless` below version 5. OpenCV 5 removed the Haar face detector from the main
> package, and `requirements.txt` already pins a compatible version.

## 3. Download the dataset

1. Sign in at kaggle.com, open <https://www.kaggle.com/datasets/sundarannamalai/hair-diseases> and click
   **Download** (about 592 MB, license CC BY-NC-SA 4.0).
2. Extract the zip into `data\raw\kaggle_hair_diseases\`. The result must look like this:

   ```
   data\raw\kaggle_hair_diseases\Hair Diseases - Final\train\<10 class folders>\*.jpg
   data\raw\kaggle_hair_diseases\Hair Diseases - Final\val\<10 class folders>\*.jpg
   data\raw\kaggle_hair_diseases\Hair Diseases - Final\test\<10 class folders>\*.jpg
   ```

3. Record where the data came from:

```bash
python scripts/download_dataset.py --source kaggle_hair_diseases --record-only --version 1
```

Alternatively, with a Kaggle API token in `%USERPROFILE%\.kaggle\kaggle.json`, run
`pip install kaggle` and then `python scripts/download_dataset.py`. More detail is in
[docs/dataset_setup.md](docs/dataset_setup.md).

## 4. Run the pipeline

Run these in order. Each step writes files that the next one reads.

### 4.1 Audit the dataset (about 2 minutes)

```bash
python scripts/audit_dataset.py
```

This decodes every image, finds duplicates and augmented copies, excludes copies that carry
conflicting labels, and writes a leakage-safe train/validation/test split to
`data/manifests/kaggle_hair_diseases_manifest.csv`.

### 4.2 Train and evaluate the classifier (about 15–25 minutes the first time)

```bash
python scripts/train_classifier.py
```

For a quick trial run, cap the images per class and split:

```bash
python scripts/train_classifier.py --max-per-class 200
```

Outputs:
- the model in `models/kaggle_hair_diseases/baseline_bundle.joblib` (plus `.json` metadata);
- metrics in `outputs/results/classification/kaggle_hair_diseases/metrics.json`;
- confusion matrices in `outputs/figures/05_classification/`.

Features are cached in `data/derived/`, so re-running is fast.

### 4.2b Train the v2 model used by the app (about 15 minutes on CPU)

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
python scripts/embed_dataset.py
```

```bash
python scripts/train_deep.py
```

`embed_dataset.py` extracts frozen MobileNetV2 features for all images (about 5 minutes, cached in
`data/derived/`). `train_deep.py` builds the strict copy-grouped split, trains and selects the model on
validation, evaluates the test set once, and saves `models/kaggle_hair_diseases/deep_bundle.joblib`
plus `outputs/results/model_v2/metrics.json`.

### 4.3 Run all notebooks (about 5–10 minutes)

```bash
python scripts/run_notebooks.py
```

```bash
python scripts/check_notebooks.py
```

The second command lists each notebook with its executed-cell count and errors (it should report 0 errors).
To work interactively instead, run `jupyter lab notebooks` and open the notebooks **in order, 00 → 08**,
using *Run All*.

### 4.4 Analyse a single image

```bash
python scripts/run_pipeline.py --image "data\raw\kaggle_hair_diseases\Hair Diseases - Final\test\Psoriasis\psoriasis_0026.jpg"
```

To analyse only part of the image, add `--roi X0 Y0 X1 Y1` (pixel coordinates in the original image).
The result panel, JSON and CSV are written to `outputs/results/pipeline/`.

### 4.5 Web app

```bash
python -m streamlit run app/streamlit_app.py
```

Open <http://localhost:8501>, then upload a scalp photo or choose a sample image. It needs the v2 model from step 4.2b.

### 4.6 Summary report

```bash
python scripts/generate_report.py --run-tests
```

This writes `reports/capability_status.md`, which lists which capabilities are verified, exploratory,
synthetic-only or unavailable, based on the files on disk.

## 5. Where to find the results

| What | Where |
|---|---|
| Dataset audit (counts, duplicates, splits) | `data/manifests/kaggle_hair_diseases_audit_report.json`, notebook 00 |
| Classifier metrics | `outputs/results/classification/kaggle_hair_diseases/metrics.json`, notebook 05 |
| All figures, per notebook | `outputs/figures/<notebook>/real/` and `.../synthetic/` |
| Integrated demo output | `outputs/demo/real/` and notebook 08 |
| Capability summary | `reports/capability_status.md` |

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Dataset 'kaggle_hair_diseases' was not found` (exit code 2) | Check the folder layout in step 3 |
| `No manifest at ...` | Run step 4.1 before 4.2 |
| `No trained model bundle` in notebook 08 | Run step 4.2 (notebook 08 then falls back to the synthetic demo model) |
| `This OpenCV build has no CascadeClassifier` | `pip install "opencv-python-headless<5"` |
| A notebook seems to hang on the widget cell | Widgets need interactive Jupyter. `run_notebooks.py` skips them automatically |
| `ModuleNotFoundError: scalptrack` | Activate the environment and run `pip install -e .` |

## 7. For developers

| Task | Command |
|---|---|
| Lint and format | `ruff check src scripts tests` and `ruff format src scripts tests` |
| Edit a notebook | Change its source in `scripts/nbbuild/nbXX.py`, run `python scripts/build_notebooks.py XX`, then `python scripts/run_notebooks.py XX` |
| Change a parameter | Edit `configs/default.yaml`. No code changes are needed |
