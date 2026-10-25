import numpy as np
import pytest
from skimage.feature import graycomatrix

from scalptrack import features as ft
from scalptrack import pyramids, texture


@pytest.mark.parametrize("d", [1, 2, 4])
@pytest.mark.parametrize("angle", [0, 45, 90, 135])
def test_glcm_matches_scikit_image(d, angle):
    rng = np.random.default_rng(0)
    g = rng.integers(0, 256, (40, 37)).astype(np.uint8)
    q = texture.quantize(g, 32)
    mine, n = texture.glcm(q, d, angle, 32)
    ref = graycomatrix(q, [d], [np.deg2rad(angle)], levels=32, symmetric=True, normed=True)[:, :, 0, 0]
    np.testing.assert_allclose(mine, ref, atol=1e-12)
    assert n > 0


def test_glcm_properties_constant_image():
    q = np.full((10, 10), 3, np.uint8)
    P, _ = texture.glcm(q, 1, 0, 8)
    p = texture.glcm_properties(P)
    assert p["contrast"] == 0 and p["asm"] == pytest.approx(1.0) and p["energy"] == pytest.approx(1.0)
    assert p["glcm_entropy_bits"] == pytest.approx(0.0) and p["homogeneity"] == pytest.approx(1.0)


def test_checkerboard_contrast():
    q = (np.indices((8, 8)).sum(0) % 2).astype(np.uint8)
    P, _ = texture.glcm(q, 1, 0, 2)
    assert texture.glcm_properties(P)["contrast"] == pytest.approx(1.0)
    assert texture.glcm_properties(P)["glcm_entropy_bits"] == pytest.approx(1.0)


def test_masked_glcm_ignores_outside_pixels():
    g = np.full((20, 20), 200, np.uint8)
    g[:, 10:] = 0  # "filled" region outside the mask
    mask = np.zeros_like(g, bool)
    mask[:, :10] = True
    masked = texture.glcm_features(g, 32, mask=mask)["aggregate"]
    assert masked["glcm_contrast_mean"] == pytest.approx(0.0)  # only the uniform valid area
    unmasked = texture.glcm_features(g, 32)["aggregate"]
    assert unmasked["glcm_contrast_mean"] > 0  # artificial edge texture


def test_empty_mask_gives_nan():
    g = np.zeros((10, 10), np.uint8)
    agg = texture.glcm_features(g, 8, mask=np.zeros_like(g, bool))["aggregate"]
    assert np.isnan(agg["glcm_contrast_mean"])
    assert np.isnan(texture.histogram_entropy(g, mask=np.zeros_like(g, bool)))


def test_histogram_entropy_values():
    assert texture.histogram_entropy(np.zeros((4, 4), np.uint8)) == 0.0
    two = np.array([[0, 255], [0, 255]], np.uint8)
    assert texture.histogram_entropy(two) == pytest.approx(1.0)


def test_quantize_range():
    q = texture.quantize(np.arange(256, dtype=np.uint8).reshape(16, 16), 32)
    assert q.min() == 0 and q.max() == 31
    with pytest.raises(ValueError):
        texture.quantize(np.zeros((2, 2), np.float32))


@pytest.mark.parametrize("shape", [(64, 64), (61, 47), (33, 100)])
def test_pyramid_exact_reconstruction(shape):
    rng = np.random.default_rng(1)
    g = rng.integers(0, 256, shape).astype(np.uint8)
    gp = pyramids.gaussian_pyramid(g, 4, 8)
    lp = pyramids.laplacian_pyramid(gp)
    rec = pyramids.reconstruct(lp)
    err = pyramids.reconstruction_error(g, rec)
    assert err["max_abs_error"] < 1e-9
    assert lp[0].min() < 0  # signed residuals kept
    assert len(pyramids.band_energies(lp)) == len(gp) - 1


def test_pyramid_stops_at_min_side():
    gp = pyramids.gaussian_pyramid(np.zeros((40, 40), np.uint8), max_levels=4, min_side_px=16)
    assert [g.shape for g in gp] == [(40, 40), (20, 20)]


def test_feature_schema_fixed_and_finite(cfg, hair_rgb):
    f = ft.extract_features(hair_rgb, cfg)
    assert list(f) == ft.feature_names()
    assert all(np.isfinite(v) for v in f.values())
    f2 = ft.extract_features(hair_rgb[:90, :70], cfg)  # different size -> same schema
    assert list(f2) == list(f)


def test_feature_groups_and_vector(cfg, hair_rgb):
    no_tex = [g for g in ft.ALL_GROUPS if g not in ft.TEXTURE_GROUPS]
    names = ft.feature_names(no_tex)
    assert not any(n.startswith("glcm") for n in names)
    f = ft.extract_features(hair_rgb, cfg)
    v = ft.features_to_vector(f, names)
    assert v.shape == (len(names),)
    with pytest.raises(KeyError):
        ft.features_to_vector({}, names)
    with pytest.raises(KeyError):
        ft.feature_names(["nope"])
