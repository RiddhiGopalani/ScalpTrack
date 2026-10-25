import numpy as np
import pytest
from PIL import Image

from scalptrack.io_utils import (
    ImageLoadError,
    ROIError,
    compare_resize_methods,
    crop_roi,
    describe_array,
    load_image,
    pixel_neighbourhood,
    resize_with_record,
    validate_roi,
)


def test_load_png_returns_rgb_uint8(png_path, hair_rgb):
    rec = load_image(png_path)
    assert rec.rgb.dtype == np.uint8 and rec.rgb.shape == hair_rgb.shape
    np.testing.assert_array_equal(rec.rgb, hair_rgb)
    assert len(rec.file_hash) == 64 and rec.width_px == 128


def test_grayscale_file_becomes_three_channels(tmp_path):
    p = tmp_path / "g.png"
    Image.fromarray(np.full((40, 50), 77, np.uint8)).save(p)
    rec = load_image(p)
    assert rec.rgb.shape == (40, 50, 3) and rec.channels_in_file == 1
    assert (rec.rgb == 77).all()


def test_rgba_alpha_noted(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(np.zeros((10, 10, 4), np.uint8)).save(p)
    rec = load_image(p)
    assert rec.rgb.shape == (10, 10, 3) and "alpha channel discarded" in rec.notes


def test_exif_orientation_applied(tmp_path):
    p = tmp_path / "rot.jpg"
    im = Image.fromarray(np.zeros((20, 40, 3), np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    im.save(p, exif=exif)
    rec = load_image(p)
    assert rec.rgb.shape[:2] == (40, 20) and rec.exif_transposed


@pytest.mark.parametrize(
    "name,content,msg",
    [
        ("missing.png", None, "does not exist"),
        ("bad.gif", b"GIF89a", "Unsupported extension"),
        ("empty.png", b"", "Empty file"),
        ("corrupt.jpg", b"\xff\xd8\xff garbage", "Corrupt or unreadable"),
    ],
)
def test_invalid_inputs_raise(tmp_path, name, content, msg):
    p = tmp_path / name
    if content is not None:
        p.write_bytes(content)
    with pytest.raises(ImageLoadError, match=msg):
        load_image(p)


def test_truncated_png_is_rejected(tmp_path, png_path):
    data = png_path.read_bytes()
    p = tmp_path / "trunc.png"
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(ImageLoadError):
        load_image(p)


def test_min_side_enforced(png_path):
    with pytest.raises(ImageLoadError, match="below the minimum"):
        load_image(png_path, min_side_px=500)


def test_directory_rejected(tmp_path):
    with pytest.raises(ImageLoadError, match="Not a file"):
        load_image(tmp_path)


def test_validate_roi_rules():
    assert validate_roi((100, 200, 3), None) == (0, 0, 200, 100)
    assert validate_roi((100, 200), (10, 20, 30, 40)) == (10, 20, 30, 40)
    with pytest.raises(ROIError):
        validate_roi((100, 200), (0, 0, 201, 10))  # outside (max exclusive)
    with pytest.raises(ROIError):
        validate_roi((100, 200), (10, 10, 10, 20))  # empty
    with pytest.raises(ROIError):
        validate_roi((100, 200), (-1, 0, 5, 5))
    with pytest.raises(ROIError):
        validate_roi((100, 200), (0, 0, 5))
    with pytest.raises(ROIError):
        validate_roi((100, 200), (0.5, 0, 5, 5))


def test_crop_is_copy_and_exclusive():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    c = crop_roi(img, (2, 3, 5, 7))
    assert c.shape == (4, 3) and c[0, 0] == 32
    c[:] = 0
    assert img[3, 2] == 32


def test_resize_record_and_methods(hair_rgb):
    out, rec = resize_with_record(hair_rgb, (64, 32))
    assert out.shape == (32, 64, 3) and not rec["isotropic"] and rec["scale_x"] == 0.5
    m = compare_resize_methods(hair_rgb, 0.5)
    assert set(m) == {"nearest", "bilinear", "area"} and all(v.shape == (64, 64, 3) for v in m.values())


def test_pixel_neighbourhood_and_describe(hair_rgb):
    n = pixel_neighbourhood(hair_rgb, 0, 0, 2)
    assert n.shape == (3, 3, 3)
    with pytest.raises(ROIError):
        pixel_neighbourhood(hair_rgb, 500, 0)
    d = describe_array(hair_rgb)
    assert d["channels"] == 3 and d["dtype"] == "uint8"
