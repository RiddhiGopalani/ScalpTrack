# Member C migration record

`ScalpTrack` is the canonical runtime package. The standalone
`scalptrack_member_c` folder remains a reference/evidence bundle and is not a
runtime dependency.

## Migrated behavior

| Member C source | Canonical destination | Decision |
|---|---|---|
| `core.py::measure_mask` | `src/scalptrack/measurements.py` | Already represented by component, crossing, skeleton, junction, and width APIs; canonical schema retained. |
| `stereo.py::compute_stereo` | `src/scalptrack/stereo.py` | Already represented by calibrated input gates, SGBM, LR consistency, Q/depth, point-cloud, and truth evaluation. |
| `core.py::detect_faces` | `src/scalptrack/face_roi.py` | Canonical detector retained; cascade loading now accepts explicit/offline asset paths. |
| `core.py::analyze_image` | `src/scalptrack/pipeline.py` | Canonical pipeline retained; no second result schema introduced. |
| `fixtures.py` | `src/scalptrack/synthetic.py` and `tests/` | Existing deterministic fixtures are richer and remain the source of truth for regression tests. |
| `audit_data.py` and `data/audit.json` | `docs/dataset_setup.md`, `docs/limitations.md`, `reports/capability_status.md` | Provenance and limitations are documented; bundled demo images are not merged into the primary dataset. |

## Configuration and verification

All runtime parameters remain in `configs/default.yaml`. `face.cascade_path`
may point to a relative `assets/haarcascade_frontalface_default.xml`; when it
is null, OpenCV's installed cascade is used. The RandomForest default is
single-worker for reproducible CPU/offline execution; it can be increased in
the YAML configuration outside restricted environments.

Run from `ScalpTrack`:

```text
python -m pytest
python scripts/run_notebooks.py
python scripts/check_notebooks.py
python scripts/run_pipeline.py --synthetic-demo
```

Synthetic results remain explicitly synthetic and do not establish real scalp
measurement or clinical performance.