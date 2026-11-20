import numpy as np
import shared_task_residual_heads_v155 as head
from test_shared_residual_head_v129 import cells
from run_tabicl import metrics


def test_fast_scores_match_original_including_probability_ties():
    for task in ('classification', 'regression'):
        for cell in cells(task):
            prediction = cell['P'][0].copy()
            if task == 'classification':
                prediction = np.round(prediction, 1)+.01
            expected = metrics(cell['y'], prediction, task, max(float(cell['y'].std()), 1e-12))
            actual = head.metric(cell, prediction)
            target = [expected['auc'], expected['accuracy']] if task == 'classification' else [expected['r2'], -expected['nrmse']]
            np.testing.assert_allclose(actual, target, atol=1e-12)


def test_prediction_requires_no_query_outcome_and_zero_gain_is_exact_base():
    for task in ('classification', 'regression'):
        cell = cells(task)[0]
        shape = (6,)+cell['P'][0].shape
        cell['task_corrections'] = np.full(shape, .1)
        query = {k: v for k, v in cell.items() if k != 'y'}
        model = {'global_head': {'kind': 'base'}, 'correction_index': 0, 'gain': 0.}
        np.testing.assert_array_equal(head.predict(query, model), cell['P'][0])
        q = head.predict(query, model | {'gain': .5})
        assert np.isfinite(q).all()
        if task == 'classification': np.testing.assert_allclose(q.sum(axis=1), 1.)


def test_candidate_grid_is_fixed():
    assert len(head.candidates()) == 38
