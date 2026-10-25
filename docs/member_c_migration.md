# Member C migration record

## Background

Member C first built their part (measurement, stereo, Haar face detection, integration) as a separate,
standalone prototype called `scalptrack_member_c`. That prototype is **not part of this repository**.
It is kept outside as a reference and evidence bundle, and nothing here imports it.

`ScalpTrack` (this repository, package `src/scalptrack/`) is the one canonical implementation. This file
records where each piece of the prototype ended up, so nobody has to compare the two codebases.

## Where each prototype component went

| Prototype source | Now lives in | Decision |
|---|---|---|
| `core.py::measure_mask` | `src/scalptrack/measurements.py` | Already covered by the component, crossing, skeleton, junction and width functions. The canonical result schema was kept. |
| `stereo.py::compute_stereo` | `src/scalptrack/stereo.py` | Already covered: input gate, SGBM, left-right check, Q-matrix depth, point cloud and ground-truth evaluation. |
| `core.py::detect_faces` | `src/scalptrack/face_roi.py` | Canonical detector kept. **New:** the cascade XML can now come from an explicit or offline path (see below). |
| `core.py::analyze_image` | `src/scalptrack/pipeline.py` | Canonical pipeline kept. No second result schema was introduced. |
| `fixtures.py` | `src/scalptrack/synthetic.py` and `tests/` | The existing deterministic fixtures are richer and remain the source of truth for the tests. |
| `audit_data.py`, `data/audit.json` | `docs/dataset_setup.md`, `docs/limitations.md`, `reports/capability_status.md` | Provenance and limitations are documented. The prototype's bundled demo images were **not** merged into the primary dataset. |

## Configuration changes that came with the migration

Both live in `configs/default.yaml`:

* **`face.cascade_path`** (default `null`). Set it to a relative path such as
  `assets/haarcascade_frontalface_default.xml` to ship your own cascade file for offline use. With `null`,
  OpenCV's installed cascade is used. `face_roi.load_cascade_from_config` searches `<project>/assets/`,
  then the project root, then OpenCV's folder, and its error message lists every place it looked.
* **`classification.rf_n_jobs`** (default `1`). RandomForest trains on a single CPU worker so runs are
  reproducible and work in restricted environments. Increase it for faster training on your own machine.

## How to verify

Run from the project root:

```bash
python -m pytest
```

```bash
python scripts/run_notebooks.py
```

```bash
python scripts/check_notebooks.py
```

```bash
python scripts/run_pipeline.py --synthetic-demo
```

Synthetic results stay explicitly synthetic. They do not establish real scalp measurements or clinical
performance.
