from pathlib import Path

import cv2
import numpy as np
import pytest

from scalptrack import face_roi


def test_load_cascade_accepts_explicit_path():
    path = Path(cv2.data.haarcascades) / face_roi.DEFAULT_CASCADE
    cascade = face_roi.load_cascade(str(path))
    assert not cascade.empty()


def test_load_cascade_from_config_uses_opencv_default(cfg):
    assert cfg["face"]["cascade_path"] is None
    assert not face_roi.load_cascade_from_config(cfg).empty()


def test_load_cascade_reports_where_it_searched(tmp_path):
    with pytest.raises(face_roi.CascadeLoadError, match="searched"):
        face_roi.load_cascade("no_such_cascade.xml", search_roots=(tmp_path,))


def test_unparseable_cascade_file_is_an_error(tmp_path):
    (tmp_path / "broken.xml").write_text("not a cascade")
    with pytest.raises(face_roi.CascadeLoadError, match="failed to load"):
        face_roi.load_cascade("broken.xml", search_roots=(tmp_path,))


def test_haar_no_face_path(cfg):
    cascade = face_roi.load_cascade()
    img = (np.random.default_rng(0).random((200, 200, 3)) * 255).astype(np.uint8)
    det = face_roi.detect_faces(img, cascade, cfg)
    assert det["status"] == "ok" and det["n_faces"] == 0 and "manual ROI" in det["reason"]
    tiny = face_roi.detect_faces(img[:20, :20], cascade, cfg)
    assert tiny["status"] == "unavailable"
    choice = face_roi.choose_roi(img.shape, det)
    assert choice["source"] == "full_image_default" and not choice["confirmed"]
    manual = face_roi.choose_roi(img.shape, det, (10, 10, 50, 60))
    assert manual["confirmed"] and manual["roi_xyxy"] == [10, 10, 50, 60]


def test_upper_head_proposal_clipped_and_unconfirmed():
    p = face_roi.propose_upper_head_roi((300, 400), (150, 40, 100, 100))
    assert p["requires_confirmation"] and p["roi_xyxy"][1] == 0 and p["clipped_at_top"]
    x0, y0, x1, y1 = p["roi_xyxy"]
    assert 0 <= x0 < x1 <= 400 and 0 <= y0 < y1 <= 300


def test_blur_faces_changes_only_box():
    img = (np.random.default_rng(0).random((50, 50, 3)) * 255).astype(np.uint8)
    out = face_roi.blur_faces(img, [(10, 10, 20, 20)])
    assert np.array_equal(out[:10], img[:10]) and not np.array_equal(out[10:30, 10:30], img[10:30, 10:30])
