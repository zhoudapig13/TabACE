import numpy as np
import source_local_support_v141 as rule


def test_classification_support_without_query_labels_or_scored_rare_labels():
    rng = np.random.default_rng(3)
    X, Q = rng.normal(size=(30, 4)), rng.normal(size=(9, 4))
    y = np.arange(30)%3
    cv = np.full((30, 3), 1/3); cv[-2:] = np.nan
    loss = np.ones(30); loss[-2:] = -1
    before = X.copy()
    values, details = rule.features(X, y, Q, [0, 1, 2, 3], cv, loss, 'classification')
    assert values.shape[:2] == (9, 3) and np.isfinite(values).all()
    np.testing.assert_array_equal(X, before)
    assert not details['query_labels_or_generator_or_strength_used']


def test_regression_source_label_affine_invariance():
    rng = np.random.default_rng(4)
    X, Q = rng.normal(size=(24, 3)), rng.normal(size=(8, 3))
    y = rng.normal(size=24); cv = y+rng.normal(size=24)*.1
    loss = (y-cv)**2/y.var()
    a, _ = rule.features(X, y, Q, [0, 1, 2], cv, loss, 'regression')
    b, _ = rule.features(X, 3*y+7, Q, [0, 1, 2], 3*cv+7, loss, 'regression')
    assert a.shape[0] == 8 and np.isfinite(a).all()
    np.testing.assert_allclose(a, b, atol=1e-12)
