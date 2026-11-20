import numpy as np
import shared_local_support_head_v143 as head
from test_shared_residual_head_v129 import cells


def source(task):
    rng = np.random.default_rng(43)
    return [c | {'LocalSupport': rng.normal(size=(80, 3, 41) if task == 'classification' else (80, 52))}
        for c in cells(task)]


def test_source_support_is_wired_and_query_labels_not_required():
    for task in ('classification', 'regression'):
        rows = source(task)
        model = head.fit(rows, {'kind': 'residual_hgb', 'strength': .5, 'max_leaf_nodes': 7})
        basic = head.prediction_features(rows[0])
        assert model['training_features'] == basic.shape[1]+(41 if task == 'classification' else 52)
        query = {k: v for k, v in rows[0].items() if k != 'y'}
        before = query['P'].copy()
        prediction = head.predict(query, model)
        assert np.isfinite(prediction).all()
        np.testing.assert_array_equal(prediction, head.predict(query | {'y': np.zeros(80)}, model))
        np.testing.assert_array_equal(query['P'], before)
        if task == 'classification':
            np.testing.assert_allclose(prediction.sum(axis=1), 1.)
        else:
            assert np.max(np.abs(prediction-before[0])) <= .5+1e-12
