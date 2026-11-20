import numpy as np

import shared_residual_head_v129 as head


def cells(task):
    rng = np.random.default_rng(123)
    result = []
    for did in range(4):
        if task == 'regression':
            P = rng.normal(size=(6, 80))
            y = P[0]+.2*np.tanh(P[1]-P[0])
        else:
            P = rng.uniform(.01, 1, size=(6, 80, 3))
            P /= P.sum(axis=2, keepdims=True)
            y = rng.integers(0, 3, 80)
        result.append({'P': P, 'y': y, 'mu': 0., 'sigma': 1., 'task': task,
            'data_id': did, 'family': 'cell_heavy_tail', 'fold': did})
    return result


def test_regression_prediction_requires_no_query_labels():
    source = cells('regression')
    model = head.fit(source, {'kind': 'residual_hgb', 'strength': .5, 'max_leaf_nodes': 7})
    query = {k: v for k, v in source[0].items() if k != 'y'}
    before = query['P'].copy()
    q = head.predict(query, model)
    assert q.shape == (80,) and np.isfinite(q).all()
    np.testing.assert_array_equal(query['P'], before)
    assert np.max(np.abs(q-before[0])) <= .5+1e-12


def test_classification_probabilities_and_label_independence():
    source = cells('classification')
    model = head.fit(source, {'kind': 'residual_hgb', 'strength': 1., 'max_leaf_nodes': 7})
    query = {k: v for k, v in source[0].items() if k != 'y'}
    q = head.predict(query, model)
    assert q.shape == (80, 3) and np.isfinite(q).all() and (q > 0).all()
    np.testing.assert_allclose(q.sum(axis=1), 1.)
    np.testing.assert_array_equal(q, head.predict(query | {'y': np.full(80, 99)}, model))
