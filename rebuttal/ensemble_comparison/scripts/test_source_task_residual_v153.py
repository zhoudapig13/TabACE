import numpy as np
import source_task_residual_v153 as rule


def test_regression_source_affine_invariance_and_no_input_mutation():
    rng = np.random.default_rng(1)
    X, Q = rng.normal(size=(30, 4)), rng.normal(size=(7, 4))
    y = 2*X[:, 0]+rng.normal(size=30)*.1
    cv = y-.3*X[:, 0]
    before = X.copy()
    a, details = rule.fit_predict(X, y, Q, [0, 1, 2, 3], cv, 'regression')
    b, _ = rule.fit_predict(X, 3*y+5, Q, [0, 1, 2, 3], 3*cv+5, 'regression')
    assert a.shape == (6, 7) and np.isfinite(a).all()
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_array_equal(X, before)
    assert details['no_query_outcome_labels']


def test_unscored_rare_classes_do_not_get_fabricated_OOF_residuals():
    rng = np.random.default_rng(2)
    X, Q = rng.normal(size=(24, 3)), rng.normal(size=(6, 3))
    y = np.r_[np.arange(22)%2, 2, 2]
    cv = np.full((24, 3), 1/3); cv[-2:] = np.nan
    correction, details = rule.fit_predict(X, y, Q, [0, 1, 2], cv, 'classification')
    assert correction.shape == (6, 6, 3) and np.isfinite(correction).all()
    np.testing.assert_array_equal(correction[:, :, 2], 0.)
    assert details['scored_source_rows'] == 22
