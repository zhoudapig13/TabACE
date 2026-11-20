import inspect

import numpy as np

from label_aware_context_v124 import smooth


def test_source_only_supervision_interface():
    assert list(inspect.signature(smooth).parameters) == ['X_source', 'y_source', 'numeric_columns', 'task']


def test_class_donors_do_not_cross_classes_and_labels_unchanged():
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=(24, 3)), rng.integers(0, 3, size=24)].astype(np.float32)
    y = np.r_[np.zeros(12), np.ones(12)]
    original = X.copy()
    a, b, info = smooth(X, y, [0, 1, 2], 'classification')
    assert np.array_equal(b, y) and np.array_equal(X, original)
    for i, donors in info['donor_sets']:
        assert all(y[j] == y[i] for j in donors)
    assert set(a[:, 3]).issubset(set(X[:, 3]))
    assert np.array_equal(a, smooth(X, y, [0, 1, 2], 'classification')[0])


def test_conditional_smoothing_reduces_an_isolated_measurement_spike():
    X = np.zeros((10, 2), dtype=np.float32)
    X[0, 0] = 100.
    y = np.zeros(10, dtype=int)
    result, _, _ = smooth(X, y, [0, 1], 'classification')
    assert result[0, 0] == 50.
    assert np.array_equal(result[1:], X[1:])


def test_regression_labels_intact_and_degenerate_input_supported():
    X = np.zeros((8, 2), dtype=np.float32)
    y = np.arange(8, dtype=float)
    result, labels, _ = smooth(X, y, [], 'regression')
    assert np.array_equal(result, X) and np.array_equal(labels, y)
