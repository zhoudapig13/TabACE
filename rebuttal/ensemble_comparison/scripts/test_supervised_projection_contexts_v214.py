import inspect

import numpy as np

import supervised_projection_contexts_v214 as constructor


def test_source_labels_preserved_and_no_query_outcome_argument():
    rng = np.random.default_rng(7)
    X, Q = rng.normal(size=(32, 7)), rng.normal(size=(19, 7))
    y = np.arange(32) % 3
    before = X.copy()
    values = constructor.construct(X, y, Q, np.arange(7), 'classification')
    assert set(inspect.signature(constructor.construct).parameters) == {
        'X_source', 'y_source', 'X_target', 'numeric_columns', 'task'}
    for value in values:
        np.testing.assert_array_equal(value['y'], y)
        assert value['X'].shape == (32, 4) and value['Q'].shape == (19, 4)
        assert np.isfinite(value['X']).all() and np.isfinite(value['Q']).all()
    np.testing.assert_array_equal(X, before)


def test_supervision_is_effective_and_rank_view_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(9)
    X, Q = rng.normal(size=(40, 6)), rng.normal(size=(20, 6))
    y = X[:, 0]+.2*X[:, 1]**2
    a = constructor.construct(X, y, Q, np.arange(6), 'regression')
    b = constructor.construct(X, rng.permutation(y), Q, np.arange(6), 'regression')
    assert not np.allclose(a[0]['X'], b[0]['X'])
    c = constructor.construct(np.exp(X), y, np.exp(Q), np.arange(6), 'regression')
    np.testing.assert_array_equal(a[1]['X'], c[1]['X'])
    np.testing.assert_array_equal(a[1]['Q'], c[1]['Q'])


def test_constant_features_and_response_and_categorical_only_are_finite():
    X, Q = np.ones((12, 3)), np.ones((5, 3))
    for task in ('regression', 'classification'):
        for v in constructor.construct(X, np.zeros(12), Q, np.arange(3), task):
            assert v['info']['fallback'] == 'all_source_features_constant'
            assert v['X'].shape == (12, 1) and np.isfinite(v['Q']).all()
    X = np.arange(36).reshape(12, 3) % 4
    Q = np.arange(15).reshape(5, 3) % 5
    for v in constructor.construct(X, np.zeros(12), Q, [], 'regression'):
        assert v['info']['fallback'] == 'constant_source_response_source_PCA'
        assert np.isfinite(v['Q']).all()
