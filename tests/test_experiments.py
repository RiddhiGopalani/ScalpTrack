"""Real-data code path (manifest -> features -> train -> bundle) on a tiny on-disk toy export."""

import matplotlib

matplotlib.use("Agg")

from scalptrack import audit, demo_data, experiments
from scalptrack.classification import load_bundle
from scalptrack.config import load_config


def _toy_cfg(root):
    cfg = load_config()
    cfg = {**cfg, "root": str(root)}
    cfg["paths"] = {k: v for k, v in cfg["paths"].items()}
    cfg["datasets"] = {"primary": "toy", "sources": {"toy": {"dir": "data/raw/toy", "url": "local", "license": "test",
                                                            "task_type": "image_classification",
                                                            "layout": "split_class_folders"}}}
    return cfg


def test_real_path_end_to_end(class_folder_dataset):
    root, _, _ = class_folder_dataset
    cfg = _toy_cfg(root)
    assert not demo_data.dataset_status(cfg)["audited"]
    man, rep = audit.build_manifest(cfg, "toy", progress=False)
    audit.save_manifest(man, rep, root / "data" / "manifests")
    st = demo_data.dataset_status(cfg)
    assert st["available"]
    dev = demo_data.development_images(cfg, n_per_class=2)
    assert dev and all(d["split"] in ("train", "validation") for d in dev)

    manifest, table = experiments.load_real_feature_table(cfg, "toy", progress=False)
    assert (root / "data" / "derived" / "toy_features.csv").is_file()
    assert set(table["split"]) <= {"train", "validation", "test"}
    out = experiments.run_experiment(cfg, table, "real", "toy", "image_classification", manifest)
    assert out["bundle_path"].is_file()
    assert (out["dirs"]["results"] / "metrics.json").is_file()
    assert load_bundle(out["bundle_path"])["data_provenance"] == "real"
    # cached features are reused (same rows, no recomputation needed)
    _, again = experiments.load_real_feature_table(cfg, "toy", progress=False)
    assert again.shape == table.shape


def test_missing_manifest_message(tmp_path):
    cfg = _toy_cfg(tmp_path)
    try:
        experiments.load_real_feature_table(cfg, "toy")
    except audit.DatasetMissingError as exc:
        assert "audit_dataset.py" in str(exc)
    else:
        raise AssertionError("expected DatasetMissingError")
    assert "NOT AVAILABLE" in demo_data.dataset_status(cfg)["message"]
