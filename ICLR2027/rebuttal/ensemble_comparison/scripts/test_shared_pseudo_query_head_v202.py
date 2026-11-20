import numpy as np
import shared_pseudo_query_head_v202 as head
from test_shared_residual_head_v129 import cells


def test_teacher_control_is_fixed_and_does_not_require_query_labels():
    for task in ('classification', 'regression'):
        c = cells(task)[0]
        c['teacher'] = c['P'].mean(axis=0)
        q = head.predict({k: v for k, v in c.items() if k != 'y'}, {'kind': 'teacher'})
        np.testing.assert_array_equal(q, c['teacher'])
    assert len(head.candidates('classification')) == 16
    assert len(head.candidates('regression')) == 19
