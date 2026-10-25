import numpy as np
import pytest

from scalptrack import edges
from scalptrack import segmentation as sg
from scalptrack import synthetic as syn
from scalptrack.preprocessing import to_gray


def test_four_threshold_methods_present(cfg, hair_rgb):
    res = sg.compare_thresholds(to_gray(hair_rgb), cfg)
    names = set(res["masks"])
    assert {"fixed", "otsu"} <= names
    assert any(n.startswith("adaptive_mean") for n in names) and any(
        n.startswith("adaptive_gaussian") for n in names
    )
    for m in res["masks"].values():
        assert m.dtype == bool and m.shape == hair_rgb.shape[:2]
    assert res["params"]["otsu"]["threshold"] > 0


def test_block_size_validation(cfg):
    g = np.zeros((40, 40), np.uint8)
    with pytest.raises(ValueError):
        sg.threshold_adaptive(g, "mean", 4, 2)
    with pytest.raises(ValueError):
        sg.threshold_adaptive(g, "mean", 51, 2)  # larger than ROI
    res = sg.compare_thresholds(g, cfg)
    assert 51 in res["skipped_block_sizes"]


def test_otsu_separates_dark_shafts(hair_rgb):
    mask, _ = sg.threshold_otsu(to_gray(hair_rgb), inverse=True)
    truth = syn.straight_shafts(4, 5.0, 90.0, size=128)["mask"]
    assert (mask & truth).sum() / truth.sum() > 0.9


def test_cleanup_removes_specks_keeps_shafts():
    truth = syn.straight_shafts(3, 3.0, 90.0, size=100)["mask"]
    noisy = truth.copy()
    noisy[5, 5] = noisy[50, 3] = True
    clean = sg.cleanup_mask(noisy, open_kernel=0, close_kernel=0, min_area=10)
    assert not clean[5, 5] and (clean == truth).all()


def test_opening_erases_thin_shafts():
    thin = syn.straight_shafts(3, 2.0, 90.0, size=80)["mask"]
    opened = sg.cleanup_mask(thin, open_kernel=3, close_kernel=0, min_area=1)
    assert opened.sum() < 0.2 * thin.sum()  # documents why opening is disabled by default


def test_empty_mask_handling():
    empty = np.zeros((30, 30), bool)
    assert not sg.remove_small_components(empty, 5).any()
    assert not sg.cleanup_mask(empty).any()


def test_segment_candidates_disjoint(cfg, hair_rgb):
    img, _ = syn.add_flakes(hair_rgb, n=8, seed=2)
    seg = sg.segment_candidates(img, cfg)
    stack = np.stack([seg["hair"], seg["flake"], seg["scalp"], seg["unknown"]]).astype(int)
    assert (stack.sum(0) == 1).all()
    assert seg["flake"].any() and seg["hair"].any()


def test_glare_and_shadow_go_to_unknown(cfg, hair_rgb):
    img = hair_rgb.copy()
    img[:10, :10] = 255
    img[-10:, -10:] = 5
    seg = sg.segment_candidates(img, cfg)
    assert seg["unknown"][:10, :10].all() and seg["unknown"][-10:, -10:].all()


def test_coverage_descriptors_and_unavailable():
    h = np.zeros((10, 10), bool)
    h[:, :2] = True
    s = ~h
    cov = sg.coverage_descriptors(h, s, np.zeros_like(h))
    assert cov["hair_coverage_fraction"] == pytest.approx(0.2) and cov[
        "scalp_to_hair_area_ratio"
    ] == pytest.approx(4.0)
    assert cov["flake_candidate_coverage"] == 0.0  # valid measurement that found none
    none = sg.coverage_descriptors(np.zeros_like(h), np.zeros_like(h))
    assert none["visible_scalp_fraction"] is None and none["scalp_to_hair_area_ratio"] is None
    with pytest.raises(ValueError):
        sg.coverage_descriptors(h, h)


def test_edges_compare_and_prewitt_kernels(cfg, hair_rgb):
    res = edges.compare_edges(to_gray(hair_rgb), cfg)
    assert {"sobel_binary", "prewitt_binary"} <= set(res["binary_maps"])
    assert sum(k.startswith("canny_") for k in res["binary_maps"]) == len(cfg["edges"]["canny_pairs"])
    assert res["sobel"]["gx"].dtype == np.float64 and res["sobel"]["gx"].min() < 0
    step = np.zeros((9, 9), np.uint8)
    step[:, 5:] = 100
    pr = edges.prewitt_gradients(step)
    assert pr["gx"][4, 4] == pytest.approx(300.0) and abs(pr["gy"][4, 4]) < 1e-9


def test_edge_summary_empty_and_canny_validation():
    s = edges.edge_summary(np.zeros((5, 5), bool))
    assert s["edge_fraction"] == 0 and s["n_fragments"] == 0
    with pytest.raises(ValueError):
        edges.canny(np.zeros((5, 5), np.uint8), 100, 50)
    assert not edges.threshold_magnitude(np.zeros((4, 4)), 0.5).any()


def test_orientation_tools():
    gx = np.ones((5, 5))
    gy = np.zeros((5, 5))
    h = edges.orientation_histogram(gx, gy, 4)
    assert h[0] == pytest.approx(1.0)
    assert edges.orientation_coherence(gx, gy) == pytest.approx(1.0)
    assert edges.orientation_coherence(np.zeros((3, 3)), np.zeros((3, 3))) == 0.0
