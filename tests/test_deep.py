"""Model-free tests for scalptrack.deep (no network weights needed)."""

import numpy as np
import pytest

from scalptrack import deep


def test_normalise_and_similarity_helpers():
    x = deep.normalise(np.array([[3.0, 4.0], [0.0, 2.0], [3.0, 4.1]]))
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
    best, arg = deep.max_similarity(x[:1], x[1:])
    assert arg[0] == 1 and best[0] > 0.99
    best_self, arg_self = deep.max_similarity(x, x, exclude_self=True)
    assert arg_self[0] == 2 and arg_self[2] == 0
    pairs = deep.similar_pairs(x, 0.99)
    assert pairs.tolist() == [[0, 2]]


def test_preprocess_shape_and_normalisation():
    img = (np.random.default_rng(0).random((300, 450, 3)) * 255).astype(np.uint8)
    t = deep.Backbone.preprocess(img)
    assert t.shape == (3, deep.INPUT_SIZE, deep.INPUT_SIZE) and t.dtype == np.float32
    assert abs(float(t.mean())) < 1.0


def test_occlude_keeps_only_requested_region():
    img = np.full((100, 100, 3), 200, np.uint8)
    border = deep.occlude(img, "border")
    centre = deep.occlude(img, "centre")
    assert (border[50, 50] == 128).all() and (border[0, 0] == 200).all()
    assert (centre[50, 50] == 200).all() and (centre[0, 0] == 128).all()
    with pytest.raises(ValueError):
        deep.occlude(img, "nope")
