import numpy as np
import shared_native8_generic_heads_v172 as head
from test_shared_residual_head_v129 import cells


def test_canonical_switch_changes_only_first_predictor():
    c = cells('regression')[0]
    c['canonical_native8'] = c['P'][0]+.25
    before = c['P'].copy()
    view = head.view(c, 'native8')
    np.testing.assert_array_equal(view['P'][1:], c['P'][1:])
    np.testing.assert_array_equal(c['P'], before)
    np.testing.assert_array_equal(head.predict({k: v for k, v in c.items() if k != 'y'},
        {'kind': 'base', 'canonical_mode': 'native8'}), c['canonical_native8'])


def test_all_specifications_are_fixed_before_outcome_evaluation():
    for task in ('classification', 'regression'):
        specifications = head.candidates(task)
        assert len(specifications) == 24
        assert {s['canonical_mode'] for s in specifications} == {'original', 'native8'}
