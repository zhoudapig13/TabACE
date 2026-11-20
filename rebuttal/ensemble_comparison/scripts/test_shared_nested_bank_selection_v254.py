import numpy as np
import torch

import shared_nested_bank_selection_v254 as head
from test_eight_output_heads_v220 import expanded


def test_parent_view_is_exact_and_does_not_change_full_bank():
    c = expanded('regression')[0]
    original = c['P'].copy()
    np.testing.assert_array_equal(head.view(c, 'parent6')['P'], original[:6])
    np.testing.assert_array_equal(head.view(c, 'query_local8')['P'], original)
    np.testing.assert_array_equal(c['P'], original)
    for task in ('classification', 'regression'):
        specifications = head.candidates(task)
        assert len(specifications) == 24
        assert {s['predictor_view'] for s in specifications} == {'parent6', 'query_local8'}


def test_both_model_arities_fit_and_predict_without_query_labels():
    torch.set_num_threads(1)
    for task in ('classification', 'regression'):
        values = expanded(task)
        for name in head.VIEWS:
            spec = {'kind': 'adaptive', 'regularization': .01, 'predictor_view': name}
            model = head.fit(values, spec)
            without = {k: v for k, v in values[0].items() if k != 'y'}
            a, b = head.predict(values[0], model), head.predict(without, model)
            np.testing.assert_array_equal(a, b)
            assert np.isfinite(a).all() and a.shape == values[0]['P'][0].shape
            backend = head.six if name == 'parent6' else head.eight
            np.testing.assert_array_equal(a, backend.predict(head.view(without, name), model['head_model']))
