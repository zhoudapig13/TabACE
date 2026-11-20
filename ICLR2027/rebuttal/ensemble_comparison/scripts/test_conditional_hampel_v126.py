import inspect

import numpy as np

from conditional_hampel_context_v126 import repair


def test_no_query_labels_or_clean_source_interface():
    assert list(inspect.signature(repair).parameters) == ['X_source', 'y_source', 'X_target', 'numeric_columns', 'task']


def test_large_spike_removed_with_labels_and_input_preserved():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3)).astype(np.float32)
    Q = rng.normal(size=(128, 3)).astype(np.float32)
    X[0, 0] = 1000
    y = np.zeros(64, dtype=int)
    before = X.copy()
    result, labels, info = repair(X, y, Q, [0, 1, 2], 'classification')
    assert abs(result[0, 0]) < 5
    assert np.array_equal(y, labels) and np.array_equal(before, X)
    assert 0 < info['changed_cells'] < X.size//5


def test_rare_legitimate_binary_numeric_values_not_collapsed():
    X = np.zeros((32, 2), dtype=np.float32)
    X[:3, 0] = 1
    Q = X.copy()
    y = np.zeros(32, dtype=int)
    result, _, _ = repair(X, y, Q, [0, 1], 'classification')
    assert np.array_equal(result, X)


def test_categorical_changes_remain_observed_codes():
    X = np.zeros((32, 2), dtype=np.float32)
    X[-1, 1] = 2
    y = np.zeros(32, dtype=int)
    result, labels, _ = repair(X, y, X, [], 'classification')
    assert set(result[:, 1]).issubset(set(X[:, 1]))
    assert np.array_equal(labels, y)
