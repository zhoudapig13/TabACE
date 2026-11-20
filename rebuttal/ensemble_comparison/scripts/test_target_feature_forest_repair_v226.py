import inspect

import numpy as np

import target_feature_forest_repair_v226 as repair


def test_non_linear_conditional_numeric_repair_excludes_response_feature():
    rng = np.random.default_rng(4)
    z = rng.uniform(-3, 3, 240)
    Q = np.c_[z, rng.normal(size=len(z)), np.sin(z)+rng.normal(scale=.025, size=len(z))]
    fitted = repair.fit_target(Q, [0, 1, 2])
    for record in fitted['info']['responses_considered']:
        assert not set(record['input_encoded_columns']) & set(record['own_encoding_columns'])
    ids = np.flatnonzero(np.abs(Q[:, 2]) > .7)[:24]
    X = Q[ids].copy()
    X[:, 2] *= -1
    out = repair.repair_source(X, fitted)[1]['X']
    assert np.mean((out[:, 2]-Q[ids, 2])**2) < .7*np.mean((X[:, 2]-Q[ids, 2])**2)
    assert set(inspect.signature(repair.fit_target).parameters) == {'Q', 'numeric_columns'}
    assert set(inspect.signature(repair.repair_source).parameters) == {'X_source', 'state'}


def test_categorical_repair_uses_valid_codes_without_interpolation():
    z = np.linspace(-3., 3., 160)
    Q = np.c_[(z > 0).astype(float), z]
    fitted = repair.fit_target(Q, [1])
    ids = np.r_[np.arange(10), np.arange(150, 160)]
    X = Q[ids].copy()
    X[:, 0] = 1-X[:, 0]
    for output in repair.repair_source(X, fitted):
        assert set(np.unique(output['X'][:, 0])).issubset({0., 1.})
        assert np.mean(output['X'][:, 0] == Q[ids, 0]) > .8
    for fold in fitted['info']['feature_CV_splits']:
        assert not set(fold['train']) & set(fold['valid'])


def test_constant_or_single_feature_context_is_finite_and_repeatable():
    Q = np.ones((12, 1))
    fitted = repair.fit_target(Q, [0])
    assert not fitted['models']
    source = np.full((7, 1), 3.)
    a, b = repair.repair_source(source, fitted), repair.repair_source(source, fitted)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x['X'], y['X'])
        assert np.isfinite(x['X']).all()
    np.testing.assert_array_equal(source, np.full((7, 1), 3.))
