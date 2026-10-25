import numpy as np
import pytest

from scalptrack import measurements as ms
from scalptrack import synthetic as syn


@pytest.mark.parametrize("n", [1, 3, 6])
@pytest.mark.parametrize("angle", [0, 30, 90, 135])
def test_component_and_crossing_counts_on_separated_shafts(cfg, n, angle):
    f = syn.straight_shafts(n, 4.0, angle, size=160)
    r = ms.estimate_visible_hairs(None, f["mask"], cfg, "synthetic")
    assert r["components"]["value"] == f["true_components"]
    assert r["crossings"]["value"] == f["true_crossings"]
    assert r["components"]["status"] == "synthetic_demo"


def test_curved_shafts_counted(cfg):
    f = syn.curved_shafts(5, 4.0, seed=3)
    r = ms.estimate_visible_hairs(None, f["mask"], cfg, "synthetic")
    assert r["components"]["value"] == 5 and r["crossings"]["value"] == 5


def test_crossing_x_is_flagged_not_hidden(cfg):
    fx = syn.challenging_fixtures()["crossing_X"]
    r = ms.estimate_visible_hairs(None, fx["mask"], cfg, "synthetic")
    assert r["components"]["value"] == 1 != fx["true_hairs"]   # known failure: one component, two hairs
    assert "components_with_more_than_two_tips" in r["flags"]


def test_curl_crosses_reference_line_multiple_times(cfg):
    fx = syn.challenging_fixtures()["curl_multi_crossing"]
    r = ms.estimate_visible_hairs(None, fx["mask"], cfg, "synthetic", line_xyxy=(80.0, 0.0, 80.0, 159.0))
    assert r["components"]["value"] == 1
    assert r["crossings"]["value"] >= 1
    r2 = ms.estimate_visible_hairs(None, fx["mask"], cfg, "synthetic", line_xyxy=fx["reference_line"])
    assert r2["crossings"]["value"] >= 2   # one curling shaft, several crossings


def test_fragmented_shaft_overcounts(cfg):
    fx = syn.challenging_fixtures()["fragmented"]
    r = ms.estimate_visible_hairs(None, fx["mask"], cfg, "synthetic")
    assert r["components"]["value"] > fx["true_hairs"]


def test_empty_mask_is_zero_not_null(cfg):
    r = ms.estimate_visible_hairs(None, np.zeros((50, 50), bool), cfg)
    assert r["components"]["value"] == 0 and r["crossings"]["value"] == 0
    assert r["components"]["status"] == "exploratory"


def test_quality_gate_gives_null(cfg):
    r = ms.estimate_visible_hairs(None, np.ones((50, 50), bool), cfg, quality={"measurement_eligible": False, "flags": ["possible_blur"]})
    assert r["crossings"]["value"] is None and r["crossings"]["status"] == "unavailable"
    w = ms.estimate_shaft_width(np.ones((50, 50), bool), cfg, quality={"measurement_eligible": False, "flags": []})
    assert w["median_px"] is None and w["status"] == "unavailable"


def test_mask_shape_must_match_roi(cfg):
    with pytest.raises(ValueError):
        ms.estimate_visible_hairs(np.zeros((10, 10, 3), np.uint8), np.zeros((11, 10), bool), cfg)


def test_count_runs_merge_rule():
    v = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0], bool)
    assert len(ms.count_runs(v, 0)) == 3
    assert len(ms.count_runs(v, 1)) == 2
    assert ms.count_runs(np.zeros(5, bool)) == []


def test_perpendicular_line_inside_roi():
    x0, y0, x1, y1 = ms.perpendicular_line((100, 80), 90.0)   # vertical hair -> horizontal line
    assert abs(y0 - y1) < 1e-6 and min(x0, x1) == pytest.approx(0) and max(x0, x1) == pytest.approx(79)


@pytest.mark.parametrize("width", [4.0, 5.0, 7.0, 9.0])
@pytest.mark.parametrize("angle", [0, 45, 90, 120])
def test_width_within_tolerance(cfg, width, angle):
    f = syn.straight_shafts(3, width, angle, size=160)
    r = ms.estimate_shaft_width(f["mask"], cfg, "synthetic")
    assert r["valid_samples"] > 50
    assert abs(r["median_px"] - width) <= 1.5   # spec development target: ~1-2 px on resolved fixtures
    assert r["unit"] == "px" and r["median_um"] is None and r["physical_status"] == "unavailable"


def test_narrow_shaft_flagged(cfg):
    f = syn.straight_shafts(3, 2.0, 90, size=120)
    r = ms.estimate_shaft_width(f["mask"], cfg, "synthetic")
    assert "below_min_resolvable_width" in r["flags"]


def test_empty_width_unavailable(cfg):
    r = ms.estimate_shaft_width(np.zeros((40, 40), bool), cfg)
    assert r["status"] == "unavailable" and r["median_px"] is None and r["valid_samples"] == 0


def test_physical_conversion_requires_documented_scale():
    assert ms.to_physical_width(5.0, None)["median_um"] is None
    assert ms.to_physical_width(5.0, 2.0)["physical_status"] == "unavailable"           # no provenance
    assert ms.to_physical_width(5.0, 2.0, "ruler", isotropic=False)["median_um"] is None
    ok = ms.to_physical_width(5.0, 2.0, "calibration target in same plane")
    assert ok["median_um"] == pytest.approx(10.0) and ok["physical_status"] == "ok"
    with pytest.raises(ValueError):
        ms.to_physical_width(5.0, -1.0, "x")


def test_pixel_width_survives_missing_scale(cfg):
    f = syn.straight_shafts(2, 6.0, 90)
    r = ms.estimate_shaft_width(f["mask"], cfg)
    assert r["median_px"] is not None and r["median_um"] is None
    assert r["status"] == "exploratory" and r["physical_status"] == "unavailable"


def test_flake_halos_excluded_but_crossing_kept(cfg):
    from scalptrack.demo_data import synthetic_closeups
    from scalptrack.segmentation import segment_candidates
    flaky = synthetic_closeups(cfg)["flakes"]
    seg = segment_candidates(flaky["rgb"], cfg)
    r = ms.estimate_visible_hairs(flaky["rgb"], seg["hair"], cfg, "synthetic")
    assert r["components"]["value"] == 7 and r["crossings"]["value"] == 7      # 7 true shafts
    assert r["components"]["non_shaft_components_excluded"] > 0              # halos around flakes removed
    x = syn.challenging_fixtures()["crossing_X"]["mask"]
    assert ms.shaft_like_mask(x, cfg)[1] == 0                                # long compact X is kept


def test_skeleton_endpoints_and_junctions():
    m = np.zeros((21, 21), bool)
    m[10, 2:19] = True
    m[2:19, 10] = True
    sk = ms.skeleton(m)
    ends, junc = ms.endpoints_and_junctions(sk)
    assert ends.sum() == 4 and junc.any()
