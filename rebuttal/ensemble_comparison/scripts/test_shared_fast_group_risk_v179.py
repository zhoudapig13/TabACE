import numpy as np
import shared_fast_group_risk_v179 as fast
from test_shared_residual_head_v129 import cells


def test_fast_scores_match_original_auc_accuracy_r2_and_nrmse():
    for task in ('classification', 'regression'):
        data = cells(task)
        for kind in ('base', 'mean', 'median'):
            model = {'kind': kind}
            a, b = fast.score(data, model), fast.original.score(data, model)
            assert a['datasets'] == b['datasets']
            np.testing.assert_allclose([a['primary'], a['secondary']], [b['primary'], b['secondary']], atol=1e-12)


def test_prediction_requires_no_query_labels():
    for task in ('classification', 'regression'):
        c = cells(task)[0]
        np.testing.assert_array_equal(fast.predict({k: v for k, v in c.items() if k != 'y'}, {'kind': 'base'}), c['P'][0])
