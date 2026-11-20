import inspect
import numpy as np
from sklearn.metrics import roc_auc_score
import domain_overlap_context_v166 as rule


def test_no_outcome_interface_and_disjoint_complete_crossfit():
    assert list(inspect.signature(rule.quality).parameters) == ['X_source', 'X_target', 'numeric_columns']
    rng = np.random.default_rng(66)
    X, Q = rng.normal(size=(90, 3)), rng.normal(size=(75, 3))
    a, b, folds = rule.quality(X, Q, np.arange(3))
    seen = []
    for f in folds:
        assert not set(f['training_rows']) & set(f['validation_rows'])
        assert sorted(f['training_rows']+f['validation_rows']) == list(range(165))
        seen += f['validation_rows']
    assert sorted(seen) == list(range(165))
    aa, bb, _ = rule.quality(X, Q, np.arange(3))
    np.testing.assert_array_equal(a, aa)
    np.testing.assert_array_equal(b, bb)


def test_detects_broken_dependence_without_task_outcomes():
    rng = np.random.default_rng(166)
    z = rng.normal(size=450)
    Q = np.stack([z, z+rng.normal(scale=.025, size=len(z))], axis=1)
    X = rng.normal(size=(450, 2))
    a, b, _ = rule.quality(X, Q, [0, 1])
    assert roc_auc_score(np.r_[np.ones(450), np.zeros(450)], np.r_[a, b]) > .7


def test_categorical_values_not_interpolated_and_inputs_unchanged():
    rng = np.random.default_rng(6)
    X = np.c_[rng.normal(size=40), rng.integers(0, 3, size=40)]
    Q = np.c_[rng.normal(size=35), rng.integers(0, 4, size=35)]
    before_x, before_q = X.copy(), Q.copy()
    features = rule.encode(X, Q, [0])
    assert np.all((features[:, 1:] == 0) | (features[:, 1:] == 1))
    np.testing.assert_array_equal(X, before_x)
    np.testing.assert_array_equal(Q, before_q)
