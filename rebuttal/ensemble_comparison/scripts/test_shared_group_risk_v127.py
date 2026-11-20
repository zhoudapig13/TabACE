import numpy as np
import pytest

import adapt_shared_heads_v119 as prediction
import shared_group_risk_v127 as group_risk


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_group_risk_fit_is_finite_and_predictor_does_not_need_truth(task):
    rng = np.random.default_rng(2)
    cells = []
    for did in range(2):
        for family in ('clean', 'cell_heavy_tail', 'feature_quantization'):
            if task == 'classification':
                y = np.tile([0, 1], 12)
                p = np.full((6, 24, 2), .5)
                for i in range(6):
                    p[i, np.arange(24), y] = .65+.03*i
                    p[i, np.arange(24), 1-y] = .35-.03*i
            else:
                y = rng.normal(size=24)
                p = np.stack([y+rng.normal(scale=.2+.03*i, size=24) for i in range(6)])
            cells.append({'task': task, 'P': p, 'mu': 0., 'sigma': 1., 'y': y,
                'family': family, 'data_id': did})
    model = group_risk.fit(cells, {'kind': 'adaptive', 'regularization': .01})
    c = cells[0].copy()
    expected = prediction.predict(c, model)
    del c['y']
    actual = prediction.predict(c, model)
    assert np.isfinite(actual).all() and np.array_equal(actual, expected)
    assert np.isfinite(model['objective'])
