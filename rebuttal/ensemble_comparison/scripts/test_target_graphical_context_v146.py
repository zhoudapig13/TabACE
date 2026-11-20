import numpy as np
import target_graphical_context_v146 as rule


def test_conditional_prediction_excludes_own_feature_value():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(90, 5))
    model = rule.fit_model(Q)
    A = rng.normal(size=(8, 5))
    B = A.copy(); B[:, 2] += 100
    np.testing.assert_allclose(rule.conditional(A, model)[:, 2],
        rule.conditional(B, model)[:, 2], atol=1e-10)


def test_repair_reduces_obvious_error_and_preserves_categorical_codes():
    rng = np.random.default_rng(2)
    z = rng.normal(size=120)
    Q = np.column_stack([z, 2*z+rng.normal(size=120)*.02, rng.integers(0, 3, 120)])
    X = np.array([[100., 0., 2.], [0., 0., 1.]], dtype=np.float32)
    before = X.copy()
    repaired, details = rule.repair(X, Q, [0, 1])
    assert np.isfinite(repaired).all() and abs(repaired[0, 0]) < 10
    np.testing.assert_array_equal(repaired[:, 2], X[:, 2])
    np.testing.assert_array_equal(X, before)
    assert not details['task_labels_used'] and not details['outcome_labels_used_in_cross_validation']


def test_no_numeric_features_is_identity():
    X = np.array([[0, 1], [1, 2]], dtype=np.float32)
    out, _ = rule.repair(X, X, [])
    np.testing.assert_array_equal(out, X)
