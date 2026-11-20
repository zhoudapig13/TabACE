import numpy as np
import original_shared_gate_adaptation_v203 as head
from test_shared_residual_head_v129 import cells


def prepared(task):
    values = cells(task)
    for i, c in enumerate(values):
        c['task_id'] = i
        c['native8_base'] = c['P'][0].copy()
        c['native8_bank'] = c['P'][1:].copy()
    return values


def test_backend_switch_preserves_original_five_alternatives_and_base_reference():
    c = prepared('regression')[0]
    c['native8_base'] += .1
    c['native8_bank'] += .2
    a = head.view(c, 'native8_all')
    np.testing.assert_array_equal(a['base'], c['native8_base'])
    np.testing.assert_array_equal(a['bank'], c['native8_bank'])
    np.testing.assert_array_equal(head.view(c, 'native1')['base'], c['P'][0])


def test_recalibrated_original_formula_predicts_without_query_labels():
    for task in ('classification', 'regression'):
        values = prepared(task)
        model = head.fit(values, {'kind': 'original_gated_fusion', 'backend': 'native1',
            'gain': 1., 'regularization': .01})
        q = head.predict({k: v for k, v in values[0].items() if k != 'y'}, model)
        assert q.shape == values[0]['P'][0].shape and np.isfinite(q).all()
        if task == 'classification':
            np.testing.assert_allclose(q.sum(axis=1), 1., atol=1e-12)
        assert model['fusion_policy']['m'] == 5
