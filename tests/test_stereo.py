import cv2
import numpy as np
import pytest

from scalptrack import stereo
from scalptrack.synthetic import render_stereo_scene


@pytest.fixture(scope="module")
def scene():
    return render_stereo_scene(width=320, height=240, seed=3, include_dome=False)


def _meta(scene):
    return {"pair_id": scene["pair_id"], "same_static_region_confirmed": True, "provenance": "synthetic"}


def test_scene_ground_truth_consistent(scene):
    c = scene["calibration"]
    np.testing.assert_allclose(scene["gt_disparity_px"], c["focal_px"] * c["baseline"] / scene["gt_depth"])
    assert scene["visible_in_right"].mean() > 0.7
    assert not np.array_equal(scene["left"], scene["right"])


def test_q_matrix_gives_z_equals_fb_over_d():
    Q = stereo.q_matrix_rectified(500.0, 159.5, 119.5, 60.0)
    disp = np.full((4, 4), 30.0, np.float32)
    pts = cv2.reprojectImageTo3D(disp, Q)
    assert pts[0, 0, 2] == pytest.approx(500 * 60 / 30)


def test_synthetic_disparity_accuracy(cfg, scene):
    res = stereo.compute_stereo(scene["left"], scene["right"], scene["calibration"], cfg, pair_meta=_meta(scene))
    assert res["status"] == "synthetic_demo" and res["depth_unit"] == "mm"
    ev = stereo.evaluate_against_truth(res, scene["gt_disparity_px"], scene["gt_depth"], scene["visible_in_right"])
    assert ev["valid_coverage"] > 0.6
    assert ev["disparity_mae_px"] < 0.5
    assert ev["depth_mae"] < 25.0
    assert np.isnan(res["depth"][~res["valid"]]).all()


def test_no_calibration_gives_disparity_but_no_depth(cfg, scene):
    res = stereo.compute_stereo(scene["left"], scene["right"], None, cfg, pair_meta=_meta(scene))
    assert res["status"] == "unavailable" and res["depth"] is None
    assert np.isfinite(res["disparity_px"]).any()


def test_gate_rejects_single_and_identical_images(cfg, scene):
    g = stereo.validate_stereo_inputs(scene["left"], None, None, None)
    assert g["status"] == "unavailable" and any("two images" in r for r in g["failed_checks"])
    g2 = stereo.validate_stereo_inputs(scene["left"], scene["left"], {"pair_id": "x", "same_static_region_confirmed": True},
                                       scene["calibration"])
    assert any("identical" in r for r in g2["failed_checks"])
    res = stereo.compute_stereo(scene["left"], None, None, cfg)
    assert res["status"] == "unavailable"


def test_calibration_validation():
    assert stereo.validate_calibration(None) == ["no calibration supplied"]
    bad = {"K1": np.eye(3), "D1": np.zeros(5), "K2": np.eye(2), "D2": np.zeros(5), "R": np.eye(3), "T": np.zeros(3), "units": "mm"}
    probs = stereo.validate_calibration(bad)
    assert "K2 must be 3x3" in probs and "baseline |T| is zero" in probs


def test_sgbm_parameter_validation(cfg):
    bad = {**cfg, "stereo": {**cfg["stereo"], "num_disparities": 50}}
    with pytest.raises(ValueError):
        stereo.make_sgbm(bad)
    bad = {**cfg, "stereo": {**cfg["stereo"], "block_size": 4}}
    with pytest.raises(ValueError):
        stereo.make_sgbm(bad)


def test_lr_consistency_simple():
    dl = np.full((3, 10), 2.0)
    dr = np.full((3, 10), 2.0)
    ok = stereo.left_right_consistency(dl, dr, 0.5)
    assert ok[:, 2:].all() and not ok[:, :2].any()
