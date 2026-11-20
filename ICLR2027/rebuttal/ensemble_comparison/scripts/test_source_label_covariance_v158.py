import numpy as np
import source_label_covariance_v158 as rule


def test_source_labels_help_reconstruct_linear_measurement_error():
    rng = np.random.default_rng(11)
    clean = rng.normal(size=256)
    X = (clean+rng.normal(size=256)*1.8)[:, None].astype(np.float32)
    Q = rng.normal(size=(512, 1))
    before = X.copy()
    repaired, detail = rule.repair(X, clean, Q, [0], 'regression')
    assert np.mean((repaired[:, 0]-clean)**2) < np.mean((X[:, 0]-clean)**2)
    np.testing.assert_array_equal(X, before)
    assert detail['source_task_labels_used'] and not detail['target_task_labels_used']


def test_classification_and_categorical_values_are_preserved():
    rng = np.random.default_rng(12)
    y = np.arange(60)%3
    X = np.column_stack([y+rng.normal(size=60), rng.integers(0, 4, 60)]).astype(np.float32)
    Q = np.column_stack([rng.normal(size=100), rng.integers(0, 4, 100)])
    repaired, detail = rule.repair(X, y, Q, [0], 'classification')
    np.testing.assert_array_equal(repaired[:, 1], X[:, 1])
    assert np.isfinite(repaired).all() and detail['categorical_columns_unchanged']


def test_no_numeric_case_is_identity():
    X = np.array([[0, 1], [1, 0]], dtype=np.float32)
    repaired, _ = rule.repair(X, np.array([0, 1]), X, [], 'classification')
    np.testing.assert_array_equal(repaired, X)
