# data/

| Folder | Contents | Committed? |
|---|---|---|
| `raw/` | Immutable downloaded exports, one folder per source (`kaggle_hair_diseases/`, `maheshwariss_hair_and_scalp/`, `litmers_scalp/`) plus `SOURCE_INFO.json` | **No** (git-ignored, licensed data) |
| `manifests/` | Audit outputs: `<source>_manifest.csv` (one row per original image plus linked crop rows), `<source>_splits.csv`, `<source>_audit_report.json`, `capability_audit.csv`; also `kaggle_hair_diseases_remote_inventory.csv` (publisher file listing, metadata only) | yes (small text) |
| `derived/` | Cached feature tables (`<source>_features.csv`) with image ids that link back to the manifest | No (regenerated) |
| `synthetic/` | Reserved for saved synthetic fixtures. Fixtures are normally generated deterministically in memory by `scalptrack.synthetic` and never mixed with real splits | yes |

Download instructions: `docs/dataset_setup.md`.
