import inspect

import numpy as np

import source_local_fusion_v263 as local


def test_source_fitting_learns_complementary_regression_forecasts():
    rng = np.random.default_rng(11)
    y = rng.normal(size=100)
    P = np.stack([.5*y, y+rng.normal(scale=.03, size=len(y)), -y])
    for kind in ('convex', 'stack'):
        model = local.fit(P, y, 'regression', 0., 1., kind)
        prediction = local.predict(P, model, 'regression', 0., 1.)
        assert np.mean((prediction-y)**2) < .5*np.mean((P[0]-y)**2)


def test_classification_predicts_normalized_probabilities_without_labels():
    y = np.arange(80) % 2
    good = np.full((len(y), 2), .1)
    good[np.arange(len(y)), y] = .9
    P = np.stack([np.full_like(good, .5), good, 1-good])
    for kind in ('convex', 'stack'):
        model = local.fit(P, y, 'classification', 0., 1., kind)
        prediction = local.predict(P, model, 'classification', 0., 1.)
        np.testing.assert_allclose(prediction.sum(1), 1., atol=1e-12)
        assert (prediction.argmax(1) == y).all()
    assert 'y_target' not in inspect.signature(local.fit).parameters
    assert 'y_source' not in inspect.signature(local.predict).parameters


def test_insufficient_source_support_returns_original_base_exactly():
    P = np.arange(18.).reshape(3, 6)
    model = local.fit(P, np.arange(6), 'regression', 0., 1., 'convex')
    assert model['kind'] == 'base'
    np.testing.assert_array_equal(local.predict(P, model, 'regression', 0., 1.), P[0])
