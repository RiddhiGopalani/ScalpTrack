import json

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from scalptrack import classification as clf
from scalptrack import synthetic as syn
from scalptrack.audit import grouped_stratified_split
from scalptrack.exports import export_result, flatten, to_jsonable
from scalptrack.pipeline import analyze_image, capability_badges
from scalptrack.viz import result_panel


def test_full_pipeline_without_model(cfg, png_path, tmp_path):
    r = analyze_image(png_path, None, None, cfg, provenance="synthetic")
    assert r["status"] == "ok"
    assert r["classification"]["status"] == "unavailable"
    assert r["stereo"]["status"] == "unavailable"
    assert r["hair_count"]["unit"] == "crossings" and r["hair_count"]["value"] == 4
    assert r["shaft_width"]["median_um"] is None and r["shaft_width"]["physical_status"] == "unavailable"
    assert r["faces"]["n_faces"] == 0
    fig = result_panel(r, capability_badges(r), "test")
    paths = export_result(r, tmp_path / "out", "case", {"panel": fig})
    rec = json.loads(open(paths["json"], encoding="utf-8").read())
    assert "_arrays" not in rec and rec["artifacts"]["figure_panel"] == "case_panel.png"
    assert (tmp_path / "out" / "case.csv").is_file() and (tmp_path / "out" / "case_panel.png").is_file()


def test_pipeline_with_roi_and_model(cfg, hair_rgb):
    samples = syn.synthetic_classification_set(6, 2, seed=2, size=96)
    table = clf.feature_table_from_arrays(samples, cfg)
    table["split"] = grouped_stratified_split(
        table["duplicate_group_id"], table["canonical_label"], (0.6, 0.2, 0.2), 0
    ).values
    bundle = clf.make_bundle(
        clf.select_on_validation(table, cfg), cfg, "image_classification", "synthetic", "syn"
    )
    r = analyze_image(hair_rgb, (10, 10, 110, 120), bundle, cfg, provenance="synthetic", run_faces=False)
    assert r["roi_xyxy"] == [10, 10, 110, 120]
    assert r["classification"]["status"] == "synthetic_demo" and r["classification"]["class_scores"]
    assert r["faces"]["status"] == "not_requested"


def test_pipeline_bad_inputs_fail_gracefully(cfg, tmp_path, hair_rgb):
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"junk")
    assert analyze_image(bad, None, None, cfg)["status"] == "failed"
    assert analyze_image(tmp_path / "missing.png", None, None, cfg)["status"] == "failed"
    r = analyze_image(hair_rgb, (0, 0, 999, 10), None, cfg)
    assert r["status"] == "failed" and "outside" in r["error"]
    assert analyze_image(hair_rgb[..., 0], None, None, cfg)["status"] == "failed"
    with pytest.raises(ValueError):
        analyze_image(hair_rgb, None, None, None)


def test_blank_image_measurements_do_not_crash(cfg):
    blank = np.full((120, 120, 3), 180, np.uint8)
    r = analyze_image(blank, None, None, cfg, run_faces=False)
    assert r["status"] == "ok"
    assert r["hair_count"]["status"] in ("unavailable", "exploratory")  # blurry/flat -> quality gate


def test_pipeline_with_synthetic_stereo(cfg, hair_rgb):
    sc = syn.render_stereo_scene(width=320, height=240, seed=1, include_dome=False)
    pair = {
        "left": sc["left"],
        "right": sc["right"],
        "calibration": sc["calibration"],
        "meta": {"pair_id": sc["pair_id"], "same_static_region_confirmed": True, "provenance": "synthetic"},
    }
    r = analyze_image(hair_rgb, None, None, cfg, provenance="synthetic", run_faces=False, stereo_pair=pair)
    assert r["stereo"]["status"] == "synthetic_demo" and r["stereo"]["depth_unit"] == "mm"


def test_jsonable_and_flatten():
    d = {"a": np.float32(1.5), "b": np.arange(3), "c": {"d": np.nan, "_hidden": 1}, "e": np.zeros((100, 100))}
    j = to_jsonable(d)
    assert j == {
        "a": 1.5,
        "b": [0, 1, 2],
        "c": {"d": None},
        "e": {"array_shape": [100, 100], "dtype": "float64"},
    }
    assert flatten({"x": {"y": 1, "z": [1, 2]}}) == {"x.y": 1, "x.z": "[1, 2]"}
