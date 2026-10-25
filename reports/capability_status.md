# ScalpTrack capability status

Generated 2026-10-25T04:49:00 by `scripts/generate_report.py` from files on disk.

## Automated tests

`pytest`: **147 passed in 22.30s** (exit code 0)

## Notebooks

| notebook | code_cells | executed | errors | status |
|---|---|---|---|---|
| 00_data_audit_basics.ipynb | 13 | 13 | 0 | executed |
| 01_filters_color.ipynb | 13 | 13 | 0 | executed |
| 02_edges_segmentation.ipynb | 12 | 12 | 0 | executed |
| 03_texture_pyramids.ipynb | 12 | 12 | 0 | executed |
| 04_hair_measurements.ipynb | 15 | 15 | 0 | executed |
| 05_classification.ipynb | 16 | 16 | 0 | executed |
| 06_stereo_depth.ipynb | 10 | 10 | 0 | executed |
| 07_face_roi.ipynb | 8 | 8 | 0 | executed |
| 08_integrated_demo.ipynb | 12 | 12 | 0 | executed |

## Capability table (from executed evidence)

| capability | status | evidence |
|---|---|---|
| Real dataset downloaded & audited | NO | manual Kaggle download required - see docs/dataset_setup.md |
| Scalp-condition classifier (real data) | NOT trained | requires the real dataset |
| Classifier software check (synthetic) | trained | random_forest: test acc 0.733, macro-F1 0.739 (majority baseline macro-F1 0.167, n=45) - synthetic labels, not scalp conditions |
| Texture ablation | synthetic only | with_texture: test macro-F1 0.739; without_texture: test macro-F1 0.866 |
| Hair count (crossings / components) | exploratory | synthetic: components exact on 100%, crossings exact on 100% of 29 separated fixtures; real images unvalidated (no count labels) |
| Hair width in pixels | exploratory | synthetic: MAE 0.31 px, max 1.00 px over 56 width/orientation fixtures |
| Hair width in micrometres | unavailable | no documented pixel scale in the linked data |
| Follicle count / hairs per cm2 | unavailable | no instance labels or scale |
| Stereo disparity / depth | synthetic demo | synthetic: coverage 84.6%, disparity MAE 0.20 px, depth MAE 6.1 mm / RMSE 45.1 mm |
| Real scalp 3-D | unavailable | no verified stereo pair/calibration |
| Haar face detection | implemented | data-limited: dataset not downloaded |
| MobileNetV2 transfer learning (P1) | not run | TensorFlow not installed in the tested environment |
| Scalp health index / treatment advice | out of scope | deferred by the specification |
