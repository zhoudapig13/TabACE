import inspect

import numpy as np

from shared_quantile_contexts_v122 import align


def test_interface_has_no_labels_or_generator_identity():
    assert list(inspect.signature(align).parameters) == ['X_source', 'X_target', 'numeric_columns']


def test_identity_when_source_matches_target_and_categories_preserved():
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=(64, 4)), rng.integers(0, 3, size=64)].astype(np.float32)
    first, second, _ = align(X, X, np.arange(4))
    assert np.array_equal(first, X)
    assert np.allclose(second, X, atol=1e-6)
    assert np.array_equal(second[:, -1], X[:, -1])


def test_transport_is_monotone_and_does_not_break_ties():
    X = np.array([[0., 0], [0., 1], [10., 0], [10., 1]], dtype=np.float32)
    Q = np.array([[-2., 0], [-1., 1], [1., 0], [2., 1]], dtype=np.float32)
    first, second, _ = align(X, Q, [0])
    assert first[0, 0] == first[1, 0]
    assert first[2, 0] == first[3, 0]
    assert first[0, 0] <= first[2, 0]
    assert np.array_equal(first[:, 1], X[:, 1]) and np.array_equal(second[:, 1], X[:, 1])
    assert first[:, 0].min() >= Q[:, 0].min() and first[:, 0].max() <= Q[:, 0].max()


def test_finite_for_constant_target_and_no_numeric_coordinates():
    X = np.arange(64, dtype=np.float32).reshape(32, 2)
    Q = np.ones((16, 2), dtype=np.float32)
    for numeric in ([], [0, 1]):
        first, second, _ = align(X, Q, numeric)
        assert np.isfinite(first).all() and np.isfinite(second).all()
        if not numeric:
            assert np.array_equal(first, X) and np.array_equal(second, X)


def test_extreme_measurements_and_reproducibility():
    rng = np.random.default_rng(4)
    Q = rng.normal(size=(32, 7)).astype(np.float32)
    X = rng.normal(size=(64, 7)).astype(np.float32)
    X[:5] *= 1000
    a = align(X, Q, np.arange(7))
    b = align(X, Q, np.arange(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.all(a[0].min(axis=0) >= Q.min(axis=0)-1e-6)
    assert np.all(a[0].max(axis=0) <= Q.max(axis=0)+1e-6)
