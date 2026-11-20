import inspect
import numpy as np
import domain_evidence_v186 as rule


def test_outcome_free_interface_and_reproducibility():
    assert list(inspect.signature(rule.descriptors).parameters) == ['X_source', 'X_target', 'numeric_columns']
    rng = np.random.default_rng(86)
    X, Q = rng.normal(size=(256, 6)), rng.normal(size=(128, 6))
    a = rule.descriptors(X, Q, np.arange(6))
    b = rule.descriptors(X, Q, np.arange(6))
    np.testing.assert_array_equal(a, b)
    assert len(a) == len(rule.NAMES) == 10


def test_quantization_reduces_matched_unique_count_descriptor():
    rng = np.random.default_rng(186)
    X, Q = rng.normal(size=(256, 5)), rng.normal(size=(256, 5))
    clean = rule.descriptors(X, Q, np.arange(5))
    rounded = rule.descriptors(np.round(X), Q, np.arange(5))
    assert rounded[2] < clean[2]-1


def test_categorical_only_and_constant_features_are_finite():
    X, Q = np.ones((40, 3)), np.ones((30, 3))
    assert np.isfinite(rule.descriptors(X, Q, [])).all()
    assert np.isfinite(rule.descriptors(X, Q, [0, 1, 2])).all()
