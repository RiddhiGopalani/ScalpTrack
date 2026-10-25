import cv2
import numpy as np
import pytest

from scalptrack import preprocessing as pp
from scalptrack.synthetic import clean_noisy_pair


def test_color_conversions_shapes_and_conventions():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 0] = 255  # pure red
    hsv = pp.to_hsv(rgb)
    assert hsv.dtype == np.uint8 and hsv[0, 0, 0] == 0 and hsv[0, 0, 1] == 255
    blue = np.zeros((2, 2, 3), np.uint8)
    blue[..., 2] = 255
    assert pp.hue_degrees(pp.to_hsv(blue))[0, 0] == pytest.approx(240.0)
    labf = pp.to_lab_float(rgb)
    assert labf.dtype == np.float32 and labf[0, 0, 1] > 50  # red has strongly positive a*
    lab8 = pp.to_lab_uint8(rgb)
    np.testing.assert_allclose(pp.lab_uint8_to_signed(lab8)[0, 0], labf[0, 0], atol=1.5)
    assert pp.to_gray(rgb).shape == (4, 4)


def test_gray_rejects_bad_shape():
    with pytest.raises(ValueError):
        pp.to_gray(np.zeros((4, 4, 2), np.uint8))


def test_circular_hue_handles_wraparound():
    hue = np.array([350.0, 10.0])
    st = pp.circular_hue_stats(hue)
    assert min(st["hue_mean_deg"], 360 - st["hue_mean_deg"]) < 1e-6


def test_channel_stack_and_patch_stats(hair_rgb):
    ch = pp.channel_stack(hair_rgb)
    assert len(ch) == 10 and ch["a* (signed)"].dtype == np.float32
    st = pp.patch_statistics(ch["Gray"])
    assert st["p05"] <= st["p50"] <= st["p95"]
    empty = pp.patch_statistics(ch["Gray"], np.zeros(ch["Gray"].shape, bool))
    assert np.isnan(empty["mean"])


def test_compare_filters_types_and_shapes(cfg, hair_rgb):
    gray = pp.to_gray(hair_rgb)
    out = pp.compare_filters(gray, cfg)
    for name, arr in out.items():
        assert arr.shape == gray.shape, name
    assert out["mean_3x3"].dtype == np.uint8
    assert out["laplacian"].dtype == np.float64 and out["laplacian"].min() < 0  # signed preserved
    sharp_name = next(k for k in out if k.startswith("sharpened"))
    assert out[sharp_name].dtype == np.float64  # unclipping is checked in test_laplacian_sharpen_formula


def test_laplacian_sharpen_formula():
    g = np.zeros((7, 7), np.uint8)
    g[3, 3] = 100
    lap = pp.signed_laplacian(g)
    assert lap[3, 3] == -400 and lap[2, 3] == 100  # OpenCV negative-centre convention
    s = pp.laplacian_sharpen(g, 0.5)
    assert s[3, 3] == pytest.approx(300.0) and s[2, 3] == pytest.approx(-50.0)
    assert pp.to_display_uint8(s)[2, 3] == 0
    with pytest.raises(ValueError):
        pp.laplacian_sharpen(g, -1)


def test_even_kernel_rejected():
    with pytest.raises(ValueError):
        pp.mean_filter(np.zeros((5, 5), np.uint8), 4)


def test_denoising_improves_psnr_on_synthetic_fixture(cfg):
    pair = clean_noisy_pair(noise_sigma=20, seed=3)
    base = pp.psnr(pair["clean"], pair["noisy"])
    smoothed = pp.gaussian_filter(pair["noisy"], 5, 1.0)
    assert pp.psnr(pair["clean"], smoothed) > base
    assert pp.psnr(pair["clean"], pair["clean"]) == float("inf")
    with pytest.raises(ValueError):
        pp.mse(pair["clean"], pair["clean"][:-1])


def test_border_policy(cfg):
    assert pp.border_flag(cfg) == cv2.BORDER_REFLECT_101
    with pytest.raises(ValueError):
        pp.border_flag({"filters": {"border": "nope"}})
