import inspect

import numpy as np

import query_cluster_contexts_v247 as local


def test_query_partition_and_class_support_and_rare_rows_preserved():
    rng = np.random.default_rng(3)
    X, Q = rng.normal(size=(100, 5)), rng.normal(size=(35, 5))
    y = np.r_[np.zeros(60), np.ones(38), 2, 2]
    before = X.copy()
    plans = local.construct(X, y, np.zeros(len(y)), 'classification', local.fit_target(Q, np.arange(5)))
    assert len(plans) == 2
    for branch in plans:
        covered = np.concatenate([p['query_rows'] for p in branch])
        np.testing.assert_array_equal(np.sort(covered), np.arange(len(Q)))
        for p in branch:
            assert {98, 99}.issubset(p['source_rows'])
            assert set(y[p['source_rows']]) == {0, 1, 2}
            assert len(p['source_rows']) <= 256
    np.testing.assert_array_equal(X, before)
    assert 'y_target' not in inspect.signature(local.fit_target).parameters


def test_regression_contexts_retain_each_response_stratum_and_are_local():
    y = np.arange(180.)
    X = np.c_[y, np.sin(y)]
    Q = X[np.arange(0, 180, 5)]
    plans = local.construct(X, y, np.zeros(len(y)), 'regression', local.fit_target(Q, [0, 1]))
    strata = np.array_split(np.arange(len(y)), 8)
    for branch in plans:
        for p in branch:
            assert all(set(group) & set(p['source_rows']) for group in strata)
    assert len({tuple(p['source_rows']) for p in plans[0]}) > 1


def test_query_only_encoding_handles_unseen_source_codes_and_high_dimensional_geometry():
    rng = np.random.default_rng(8)
    Q = rng.normal(size=(35, 40))
    X = rng.normal(size=(60, 40))
    Q[:, -1] = np.arange(len(Q)) % 2
    X[:, -1] = 7
    model = local.fit_target(Q, np.arange(39))
    assert model['pca'] is not None
    plans = local.construct(X, np.arange(len(X)), np.zeros(len(X)), 'regression', model)
    assert all(len(p['source_rows']) == len(X) for branch in plans for p in branch)
    constant = local.fit_target(np.ones((5, 1)), [0])
    assert len(constant['cluster_centers']) == 1
