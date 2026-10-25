import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scalptrack import audit
from scalptrack.config import load_config


def _cfg_for(root, extra_source=None):
    cfg = load_config()
    cfg = {**cfg, "root": str(root)}
    cfg["datasets"] = {"primary": "toy", "sources": {"toy": {"dir": "data/raw/toy", "url": "local", "license": "test",
                                                            "task_type": "image_classification",
                                                            "layout": "split_class_folders"}}}
    if extra_source:
        cfg["datasets"]["sources"].update(extra_source)
    return cfg


def test_manifest_detects_corrupt_duplicates_and_regroups(class_folder_dataset):
    root, base, _ = class_folder_dataset
    cfg = _cfg_for(root)
    man, rep = audit.build_manifest(cfg, "toy", progress=False)
    assert rep["n_image_files"] == 26 and rep["n_unreadable"] == 1 and rep["n_other_files"] == 1
    assert rep["class_names_raw"] == ["Class A", "Class B"]
    assert rep["duplicates"]["exact_duplicate_files"] >= 1
    broken = man[man["relative_path"].str.endswith("broken.jpg")].iloc[0]
    assert broken["exclusion_reason"] == "unreadable_or_corrupt" and broken["split"] == "excluded"
    # the duplicate crossed train/test -> the publisher split must NOT be kept
    assert rep["split"]["groups_crossing_publisher_split"] >= 1
    assert "re-split" in rep["split"]["decision"]
    usable = man[man["split"].isin(audit.SPLITS)]
    assert (usable.groupby("duplicate_group_id")["split"].nunique() == 1).all()
    assert set(audit.MANIFEST_COLUMNS) <= set(man.columns)
    assert man["patient_id"].isna().all() and man["pixel_scale_um_per_px"].isna().all()


def test_publisher_split_kept_when_clean(class_folder_dataset):
    root, base, _ = class_folder_dataset
    (base / "Wrapper" / "test" / "Class A" / "dup.png").unlink()
    man, rep = audit.build_manifest(_cfg_for(root), "toy", progress=False)
    assert rep["split"]["groups_crossing_publisher_split"] == 0
    assert rep["split"]["decision"].startswith("publisher split preserved")
    ok = man[man["split"].isin(audit.SPLITS)]
    assert (ok["split"] == ok["publisher_split"]).all()


def test_flipped_copy_is_near_duplicate(tmp_path):
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64)) * 255).astype(np.uint8)
    img = np.asarray(Image.fromarray(img).resize((64, 64)))
    a = np.array([audit.dihedral_dhashes(img)], dtype=np.uint64)
    b = np.array([audit.dihedral_dhashes(np.ascontiguousarray(img[:, ::-1]))], dtype=np.uint64)
    other = np.array([audit.dihedral_dhashes((rng.random((64, 64)) * 255).astype(np.uint8))], dtype=np.uint64)
    pairs = audit.near_duplicate_pairs(np.concatenate([a, b, other]), threshold=4)
    assert [0, 1] in pairs.tolist()
    assert not any(2 in p for p in pairs.tolist())


def test_dhash_lookalike_rejected_by_thumbnail_check():
    from scalptrack import synthetic as syn
    # Two DIFFERENT images with strands at the same positions collide under dHash alone.
    a = np.asarray(Image.fromarray(syn.synthetic_texture_sample("synthetic_sparse_strands", 103, 64)).convert("L"))
    b = np.asarray(Image.fromarray(syn.synthetic_texture_sample("synthetic_sparse_strands", 201, 64)).convert("L"))
    flipped = np.ascontiguousarray(a[:, ::-1])
    hashes = np.array([audit.dihedral_dhashes(x) for x in (a, b, flipped)], dtype=np.uint64)
    thumbs = [audit.thumbnail(x) for x in (a, b, flipped)]
    g_hash_only, st0 = audit.duplicate_groups(["h1", "h2", "h3"], hashes, 6)
    g_verified, st1 = audit.duplicate_groups(["h1", "h2", "h3"], hashes, 6, thumbs, 0.9)
    assert g_hash_only[0] == g_hash_only[1]              # false merge without verification
    assert g_verified[0] != g_verified[1]                # rejected by thumbnail correlation
    assert g_verified[0] == g_verified[2]                # true flipped copy still merged
    assert st1["dhash_candidate_pairs"] > st1["near_duplicate_pairs"]


def test_missing_dataset_message(tmp_path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    cfg = _cfg_for(tmp_path)
    with pytest.raises(audit.DatasetMissingError, match="docs/dataset_setup.md"):
        audit.build_manifest(cfg, "toy", progress=False)


def test_save_and_load_manifest(class_folder_dataset, tmp_path):
    root, _, _ = class_folder_dataset
    man, rep = audit.build_manifest(_cfg_for(root), "toy", progress=False)
    paths = audit.save_manifest(man, rep, tmp_path / "manifests")
    assert all(p.is_file() for p in paths.values())
    back = audit.load_manifest(paths["manifest"])
    assert len(back) == len(man)
    t = audit.split_count_table(back)
    assert "train_images" in t.columns and "train_groups" in t.columns


def test_label_conflict_excluded(tmp_path):
    base = tmp_path / "data" / "raw" / "toy"
    for cls in ("A", "B"):
        (base / cls).mkdir(parents=True)
    img = (np.random.default_rng(1).random((40, 40, 3)) * 255).astype(np.uint8)
    Image.fromarray(img).save(base / "A" / "x.png")
    Image.fromarray(img).save(base / "B" / "x.png")  # same picture, two labels
    other = (np.random.default_rng(2).random((40, 40, 3)) * 255).astype(np.uint8)
    Image.fromarray(other).save(base / "A" / "y.png")
    cfg = _cfg_for(tmp_path)
    cfg["datasets"]["sources"]["toy"]["layout"] = "class_folders"
    man, rep = audit.build_manifest(cfg, "toy", progress=False)
    assert rep["n_label_conflict"] == 2 and rep["layout"] == "class_folders"


def test_yolo_export_region_rows(tmp_path):
    base = tmp_path / "data" / "raw" / "yolo"
    (base / "train" / "images").mkdir(parents=True)
    (base / "train" / "labels").mkdir(parents=True)
    (base / "data.yaml").write_text("names: ['Psoriasis', 'Lichen-Planus']\n")
    for k in range(3):
        img = (np.random.default_rng(k).random((100, 80, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(base / "train" / "images" / f"im{k}.jpg")
        (base / "train" / "labels" / f"im{k}.txt").write_text("0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.2 0.2\n")
    cfg = _cfg_for(tmp_path, {"yolo": {"dir": "data/raw/yolo", "license": "CC BY 4.0", "url": "x",
                                       "task_type": "region_classification", "layout": "yolo"}})
    man, rep = audit.build_manifest(cfg, "yolo", progress=False)
    regions = man[man["task_type"] == "region_classification"]
    assert len(regions) == 6 and set(regions["raw_label"]) == {"Psoriasis", "Lichen-Planus"}
    assert regions["parent_image_id"].notna().all()
    # all crops of an original share its split
    assert (regions.groupby("parent_image_id")["split"].nunique() == 1).all()
    assert regions.iloc[0]["bbox_xyxy"] == [24, 30, 56, 70]


def test_capability_audit_reports_unavailable():
    cap = audit.capability_audit(None)
    assert (cap.loc[cap["capability"] == "calibrated stereo pairs", "status"] == "unavailable").all()


def test_canonical_label():
    assert audit.canonical_label("Hair-and-Scalp-Diseases-MPB-") == "hair_and_scalp_diseases_mpb"
    assert audit.canonical_label(None) is None
