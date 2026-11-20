import numpy as np
import pytest

import adapt_shared_heads_v119 as heads


def classification():
    rng = np.random.default_rng(0)
    p = rng.random((6, 32, 3))
    p /= p.sum(axis=2, keepdims=True)
    return {'task': 'classification', 'P': p, 'mu': 0., 'sigma': 1., 'y': rng.integers(0, 3, 32)}


def regression():
    return {'task': 'regression', 'P': np.random.default_rng(0).normal(size=(6, 32)),
        'mu': .3, 'sigma': 2., 'y': np.arange(32)}


@pytest.mark.parametrize('kind', ['base', 'mean', 'median', 'adaptive', 'loglinear'])
def test_classifier_prediction_does_not_read_truth_and_is_a_probability(kind):
    c = classification()
    model = {'kind': kind, 'parameters': np.arange(18).tolist(), 'beta': [1., -.2, .1, .1, 0., 0.]}
    before = c['P'].copy()
    expected = heads.predict(c, model)
    del c['y']
    actual = heads.predict(c, model)
    assert np.array_equal(expected, actual)
    assert np.array_equal(c['P'], before)
    assert np.isfinite(actual).all() and (actual >= 0).all()
    assert np.allclose(actual.sum(axis=1), 1.)


@pytest.mark.parametrize('kind,cap', [('adaptive', None), ('adaptive', 2.), ('ridge', None),
    ('base', None), ('mean', None), ('median', None)])
def test_regression_affine_unit_equivariance_without_truth(kind, cap):
    c = regression()
    model = {'kind': kind, 'cap': cap, 'parameters': np.linspace(-1, 1, 18).tolist(),
        'beta': [1.1, -.1, 0, 0, 0, 0], 'intercept': .2}
    before = c['P'].copy()
    y = heads.predict(c, model)
    a, b = 7., -3.
    changed = c | {'P': a*c['P']+b, 'mu': a*c['mu']+b, 'sigma': a*c['sigma']}
    del changed['y']
    assert np.allclose(heads.predict(changed, model), a*y+b, atol=1e-10)
    assert np.array_equal(c['P'], before)


def test_no_adaptive_change_when_every_member_matches_base():
    c = regression()
    c['P'][:] = c['P'][0]
    for cap in (None, 2.):
        model = {'kind': 'adaptive', 'cap': cap, 'parameters': np.arange(18).tolist()}
        assert np.allclose(heads.predict(c, model), c['P'][0])


def test_loglinear_base_identity():
    c = classification()
    p = heads.predict(c, {'kind': 'loglinear', 'beta': [1., 0, 0, 0, 0, 0]})
    assert np.allclose(p, c['P'][0])
