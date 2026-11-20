import inspect

import numpy as np

import raw_excess_covariance_contexts_v257 as correction


def test_identity_when_source_and_query_match_and_categorical_values_untouched():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 5))
    X[:, -1] = np.arange(len(X)) % 3
    for output in correction.construct(X, correction.fit_target(X, np.arange(4))):
        np.testing.assert_allclose(output['X'], X.astype(np.float32), atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(output['X'][:, -1], X[:, -1])
    assert set(inspect.signature(correction.construct).parameters) == {'X_source', 'target_model'}


def test_linear_measurement_error_ablation_uses_no_task_labels_in_constructor():
    rng = np.random.default_rng(12)
    latent = rng.normal(size=4000)
    observed = (latent+rng.normal(size=len(latent)))[:, None]
    Q = rng.normal(size=(5000, 1))
    transformed = correction.construct(observed, correction.fit_target(Q, [0]))[1]['X'][:, 0]
    raw_beta = np.cov(observed[:, 0], latent, ddof=1)[0, 1]/observed[:, 0].var(ddof=1)
    repaired_beta = np.cov(transformed, latent, ddof=1)[0, 1]/transformed.var(ddof=1)
    assert abs(repaired_beta-1.) < .15 and abs(repaired_beta-1.) < abs(raw_beta-1.)


def test_no_numeric_and_constant_query_features_are_finite():
    X = np.arange(24).reshape(8, 3).astype(np.float32)
    Q = np.ones((6, 3), dtype=np.float32)
    for result in correction.construct(X, correction.fit_target(Q, [])):
        np.testing.assert_array_equal(result['X'], X)
    for result in correction.construct(X, correction.fit_target(Q, [0, 2])):
        assert np.isfinite(result['X']).all()
        np.testing.assert_array_equal(result['X'][:, 1], X[:, 1])
