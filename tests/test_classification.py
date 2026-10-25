import numpy as np
import pytest

from scalptrack import classification as clf
from scalptrack import synthetic as syn
from scalptrack.audit import grouped_stratified_split


@pytest.fixture(scope="module")
def synthetic_table(cfg):
    samples = syn.synthetic_classification_set(n_sources_per_class=10, variants_per_source=3, seed=1, size=96)
    table = clf.feature_table_from_arrays(samples, cfg)
    table["split"] = grouped_stratified_split(
        table["duplicate_group_id"], table["canonical_label"], (0.6, 0.2, 0.2), cfg["seed"]
    ).values
    return table


def test_grouped_split_keeps_variants_together(synthetic_table):
    per_group = synthetic_table.groupby("duplicate_group_id")["split"].nunique()
    assert (per_group == 1).all()
    assert set(synthetic_table["split"]) == {"train", "validation", "test"}


def test_selection_uses_validation_and_beats_majority(cfg, synthetic_table):
    sel = clf.select_on_validation(synthetic_table, cfg)
    assert sel["best_name"] != "majority_baseline"
    assert "val_macro_f1" in sel["selection_table"].columns
    ev = clf.final_test_evaluation(sel, synthetic_table)
    assert ev["status"] == "ok"
    assert ev["selected"]["macro_f1"] > ev["majority_baseline"]["macro_f1"]
    cm = np.array(ev["selected"]["confusion_matrix"])
    assert cm.sum() == (synthetic_table["split"] == "test").sum()


def test_preprocessing_fitted_on_train_only(cfg, synthetic_table):
    sel = clf.select_on_validation(synthetic_table, cfg)
    model = sel["best_model"]
    if "scale" in model.named_steps:
        train = synthetic_table[synthetic_table["split"] == "train"][sel["feature_names"]].to_numpy()
        np.testing.assert_allclose(
            model.named_steps["scale"].mean_, np.nanmean(train, axis=0), rtol=1e-9, atol=1e-9
        )


def test_ablation_table(cfg, synthetic_table):
    ab = clf.texture_ablation(synthetic_table, cfg)
    assert list(ab["variant"]) == ["with_texture", "without_texture"]
    assert ab.loc[0, "n_features"] > ab.loc[1, "n_features"]


def test_bundle_roundtrip_and_predict(cfg, synthetic_table, tmp_path, hair_rgb):
    sel = clf.select_on_validation(synthetic_table, cfg)
    bundle = clf.make_bundle(sel, cfg, "image_classification", "synthetic", "synthetic_fixture")
    path = clf.save_bundle(bundle, tmp_path / "m.joblib")
    assert path.with_suffix(".json").is_file()
    loaded = clf.load_bundle(path)
    pred = clf.predict_condition(hair_rgb, loaded, cfg)
    assert pred["status"] == "synthetic_demo" and pred["predicted_label"] in loaded["classes"]
    assert list(pred["class_scores"]) == sorted(pred["class_scores"], key=lambda k: -pred["class_scores"][k])
    assert sum(pred["class_scores"].values()) == pytest.approx(1.0)


def test_missing_model_handled(cfg, hair_rgb, tmp_path):
    with pytest.raises(clf.ModelMissingError):
        clf.load_bundle(tmp_path / "nope.joblib")
    with pytest.raises(clf.ModelMissingError):
        clf.load_bundle(None)
    out = clf.predict_condition(hair_rgb, None, cfg)
    assert out["status"] == "unavailable" and out["predicted_label"] is None


def test_metrics_contents():
    m = clf.metrics(["a", "b", "a"], ["a", "a", "a"], ["a", "b"])
    assert m["accuracy"] == pytest.approx(2 / 3) and m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["confusion_matrix"] == [[2, 0], [1, 0]]


def test_group_cv(cfg, synthetic_table):
    cv = clf.exploratory_group_cv(synthetic_table, cfg, n_splits=3)
    assert cv["status"] == "exploratory" and len(cv["macro_f1_per_fold"]) == 3


def test_uncertainty_threshold_from_validation(cfg, synthetic_table):
    sel = clf.select_on_validation(synthetic_table, cfg)
    thr = clf.choose_uncertainty_threshold(sel, synthetic_table, 0.9)
    assert 0 < thr <= 1


def test_selection_requires_train_and_val(cfg, synthetic_table):
    t = synthetic_table.copy()
    t["split"] = "train"
    with pytest.raises(ValueError):
        clf.select_on_validation(t, cfg)
