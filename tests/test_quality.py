import cv2
import numpy as np
import pytest

from scalptrack.quality import assess_quality, laplacian_variance


def test_sharp_image_eligible(cfg, hair_rgb):
    q = assess_quality(hair_rgb, cfg)
    assert q["measurement_eligible"] and "possible_blur" not in q["flags"]


def test_blur_lowers_laplacian_variance_and_flags(cfg, hair_rgb):
    blurred = cv2.GaussianBlur(hair_rgb, (0, 0), 6)
    assert laplacian_variance(cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)) < laplacian_variance(
        cv2.cvtColor(hair_rgb, cv2.COLOR_RGB2GRAY)
    )
    q = assess_quality(blurred, cfg)
    assert "possible_blur" in q["flags"] and not q["measurement_eligible"]


def test_exposure_flags(cfg):
    dark = np.full((100, 100, 3), 5, np.uint8)
    bright = np.full((100, 100, 3), 252, np.uint8)
    assert "underexposed_or_shadow" in assess_quality(dark, cfg)["flags"]
    qb = assess_quality(bright, cfg)
    assert "overexposed_or_glare" in qb["flags"] and "specular_highlights" in qb["flags"]


def test_small_roi_not_eligible(cfg, hair_rgb):
    q = assess_quality(hair_rgb[:20, :20], cfg)
    assert "roi_too_small" in q["flags"] and not q["measurement_eligible"]


def test_rejects_gray_input(cfg):
    with pytest.raises(ValueError):
        assess_quality(np.zeros((50, 50), np.uint8), cfg)
