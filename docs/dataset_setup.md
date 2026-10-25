# Dataset setup

ScalpTrack uses **only the public sources linked in the specification** (Section 4). No dataset is
committed to this repository, and none was downloaded while building it: every source needs your own
account. The code runs without data and tells you exactly what is missing.

## What is known without downloading

The Kaggle public API exposes the *file listing* of the primary dataset without logging in. It was
recorded on 2026-10-25 in `data/manifests/kaggle_hair_diseases_remote_inventory.csv`:

| Property | Value (publisher listing, not an audit) |
|---|---|
| Dataset | `sundarannamalai/hair-diseases` (version 1) |
| License | CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution required) |
| Files | 12,000 `.jpg`, about 592 MB |
| Layout | `Hair Diseases - Final/{train,val,test}/<Class>/<file>.jpg` |
| Classes (10) | Alopecia Areata, Contact Dermatitis, Folliculitis, Head Lice, Lichen Planus, Male Pattern Baldness, Psoriasis, Seborrheic Dermatitis, Telogen Effluvium, Tinea Capitis |
| Per class | 960 train / 120 val / 120 test, exactly |

The perfectly equal counts suggest curation or augmentation. Notebook 00 therefore checks for exact and
near-duplicates (including flips and rotations) **before** it trusts the publisher split.

## Option A (recommended): Kaggle "Hair Diseases", the primary source

1. Sign in at <https://www.kaggle.com> and open
   <https://www.kaggle.com/datasets/sundarannamalai/hair-diseases>. Read and accept the terms.
2. Either:
   * **Browser:** click *Download*, then extract the zip into
     `data\raw\kaggle_hair_diseases\` so that `data\raw\kaggle_hair_diseases\Hair Diseases - Final\train\...` exists.
     Then record the provenance:
     ```bash
     python scripts/download_dataset.py --source kaggle_hair_diseases --record-only --version 1
     ```
   * **API:** Account → Settings → API → *Create New Token*. Save `kaggle.json` to
     `%USERPROFILE%\.kaggle\kaggle.json`, never inside this repository. Then:
     ```bash
     python -m pip install kaggle
     ```
     ```bash
     python scripts/download_dataset.py --source kaggle_hair_diseases
     ```
3. Audit it, which writes the manifest, splits and report:
   ```bash
   python scripts/audit_dataset.py
   ```
4. Re-run the notebooks (see the README). Notebooks 00–08 switch automatically to the real data.

Expected layout. Any wrapper folder name works, because the split folder is found anywhere in the path:

```
data/raw/kaggle_hair_diseases/
├── SOURCE_INFO.json              (written by download_dataset.py: url, version, license, date)
└── Hair Diseases - Final/
    ├── train/<Class Name>/*.jpg
    ├── val/<Class Name>/*.jpg     ("val", "valid" and "validation" are all recognised)
    └── test/<Class Name>/*.jpg
```

A flat `data/raw/kaggle_hair_diseases/<Class Name>/*.jpg` layout without a publisher split also works.
The audit then creates a grouped 70/15/15 split.

## Option B: Roboflow "Maheshwariss HAIR AND SCALP" (fallback, region classification)

Use this source if the Kaggle export is unusable. It is an **object-detection** export with four labels:
`Contact Dermatitis`, `Hair-and-Scalp-Diseases-MPB-`, `Lichen-Planus` and `Psoriasis`. The raw names are
preserved. ScalpTrack classifies **ground-truth box crops**, which is *annotated-region classification*,
not whole-photo diagnosis.

1. Open <https://universe.roboflow.com/maheshwariss-workspace-nn7d1/hair-and-scalp-w3ih8> and sign in.
2. Pick a version, then *Download Dataset* → format **YOLOv8** → *Download zip to computer*.
3. Extract it into `data\raw\maheshwariss_hair_and_scalp\` so that `data.yaml` sits directly in that folder.
4. Record the provenance and audit:
   ```bash
   python scripts/download_dataset.py --source maheshwariss_hair_and_scalp --record-only --version <N>
   ```
   ```bash
   python scripts/audit_dataset.py --source maheshwariss_hair_and_scalp
   ```
5. To make it the classifier source, set `datasets.primary: maheshwariss_hair_and_scalp` in
   `configs/default.yaml`.

All crops from one original image share that image's split, so crops never leak across splits.

## Option C: Roboflow "Litmers scalp" (demonstrations only)

37 listed images and 10 overlapping sign/condition labels. Use them only for demonstration and
failure-case figures. They are too few for a classifier. Same steps as Option B, folder
`data\raw\litmers_scalp\`.

## Not used

* The Aditi "hair-problem" Roboflow listing could not be verified. It stays unselected (spec 4.2).
* No other dataset is added. Linked sources are never merged just to increase the image count.

## Privacy and licensing

* Keep `kaggle.json` and Roboflow keys outside the repository (`.gitignore` also excludes `kaggle.json`).
* `data/raw/` is git-ignored. Do not commit images, because the CC BY-NC-SA terms require attribution and
  non-commercial use.
* Notebook 07 blurs detected faces in saved figures and exports only box coordinates.
