import inspect
import numpy as np
import joint_transport_context_v176 as rule


def test_balanced_coupling_is_finite_and_preserves_marginals():
    cost = np.random.default_rng(0).uniform(0, 2, size=(31, 45))
    rows, info = rule.coupling(cost)
    np.testing.assert_allclose(rows.sum(axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(rows.sum(axis=0)/len(rows), 1/45, atol=1e-6)
    assert info['row_marginal_max_error'] < 1e-6


def test_no_query_outcome_interface_and_no_input_mutation():
    assert list(inspect.signature(rule.construct).parameters) == [
        'X_source', 'y_source', 'X_target', 'numeric_columns', 'task', 'base_prediction']
    rng = np.random.default_rng(17)
    X, Q = rng.normal(size=(40, 3)), rng.normal(size=(50, 3))
    y, p = rng.normal(size=40), rng.normal(size=50)
    before = [v.copy() for v in (X, y, Q, p)]
    a, info = rule.construct(X, y, Q, [0, 1, 2], 'regression', p)
    b, _ = rule.construct(X, y, Q, [0, 1, 2], 'regression', p)
    for old, new in zip(before, (X, y, Q, p)):
        np.testing.assert_array_equal(old, new)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    assert info['query_outcomes_used'] is False


def test_categorical_coordinates_remain_valid_observed_codes():
    rng = np.random.default_rng(176)
    X = np.c_[rng.normal(size=50), rng.integers(0, 3, size=50)]
    Q = np.c_[rng.normal(size=45), rng.integers(0, 3, size=45)]
    y = rng.integers(0, 2, size=50)
    p = rng.uniform(.1, .9, size=45)
    result, _ = rule.construct(X, y, Q, [0], 'classification', np.c_[1-p, p])
    for A in result:
        assert set(A[:, 1]) <= set(X[:, 1]) | set(Q[:, 1])
