import numpy as np
import source_crossfit_manifold_v137 as rule


def test_no_source_crossfit_signal_leaves_context_unchanged():
    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.repeat([0, 1], 6)
    Q = X+100
    P = np.tile([.5, .5], (12, 1))
    outputs, labels, details = rule.repair(X, y, Q, P, [0], np.zeros(12), 'classification')
    for output in outputs:
        np.testing.assert_array_equal(output, X)
    np.testing.assert_array_equal(labels, y)
    assert not details['uses_true_target_labels_or_clean_source_features']


def test_classification_repair_preserves_inputs_and_legal_categorical_codes():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.normal(size=24), rng.integers(0, 3, 24)]).astype(np.float32)
    y = np.repeat([0, 1], 12)
    Q = np.column_stack([rng.normal(size=16), rng.integers(0, 3, 16)]).astype(np.float32)
    P = np.tile([.9, .1], (16, 1)); P[8:] = [.1, .9]
    loss = np.tile(np.arange(12), 2).astype(float)
    before = X.copy()
    outputs, labels, _ = rule.repair(X, y, Q, P, [0], loss, 'classification')
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(y, labels)
    for output in outputs:
        assert np.isfinite(output).all() and output.shape == X.shape
        assert set(output[:, 1]).issubset({0, 1, 2})
        np.testing.assert_array_equal(output[loss == 0], X[loss == 0])


def test_regression_does_not_require_target_labels():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(32, 3)).astype(np.float32)
    Q = rng.normal(size=(20, 3)).astype(np.float32)
    y = np.arange(32, dtype=float)
    pred = np.linspace(0, 31, len(Q))
    outputs, labels, details = rule.repair(X, y, Q, pred, [0, 1, 2], np.tile([0., 1., 2., 3.], 8), 'regression')
    assert all(np.isfinite(output).all() for output in outputs)
    assert all(0 <= d['maximum_actual_row_blend'] <= .9 for d in details['branch_details'])
    np.testing.assert_array_equal(labels, y)
