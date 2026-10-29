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


def test_match_scorer_is_low_for_photos_unlike_every_class():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    centres = deep.normalise(rng.normal(size=(3, 16)))
    def sample(n):
        y = np.repeat(np.arange(3), n)
        return deep.normalise(centres[y] + 0.15 * rng.normal(size=(len(y), 16))), np.array(["a", "b", "c"])[y]
    xtr, ytr = sample(40)
    xva, yva = sample(20)
    head = LogisticRegression(max_iter=1000).fit(xtr, ytr)
    scorer = deep.MatchScorer(head, xtr, ytr, k=3).calibrate(xva, yva)
    typical = scorer(centres)  # one prototypical photo per class
    assert typical.shape == (3, 3) and np.all(typical.argmax(1) == [0, 1, 2])
    assert np.all(typical.max(1) > 0.5)
    unlike = scorer(deep.normalise(rng.normal(size=(5, 16))))  # unlike any class
    assert np.all(unlike.max(1) < 0.1)
    assert np.all(scorer.nearest_similarity(xtr[:2]) > 0.99)


def test_match_scorer_healthy_class_lowers_every_condition():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    centres = deep.normalise(rng.normal(size=(3, 16)))
    names = np.array(["a", "b", "healthy"])
    def sample(n, classes=(0, 1, 2)):
        y = np.repeat(np.array(classes), n)
        return deep.normalise(centres[y] + 0.15 * rng.normal(size=(len(y), 16))), names[y]
    xtr, ytr = sample(40)
    xva, yva = sample(20, classes=(0, 1))  # no validation photos of the healthy class
    head = LogisticRegression(max_iter=1000).fit(xtr, ytr)
    scorer = deep.MatchScorer(head, xtr, ytr, k=3, healthy_label="healthy").calibrate(xva, yva)
    assert scorer.match_classes == ["a", "b"]
    healthy_like = centres[2:3]
    assert scorer.healthy_probability(healthy_like)[0] > 0.5
    assert scorer(healthy_like).max() < 0.1
    assert scorer(centres[:1]).argmax() == 0
