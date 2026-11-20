import numpy as np
import torch

import adapt_shared_heads_v119 as six
import adapt_shared_heads_eight_v220 as eight
import shared_group_risk_eight_v221 as grouped
from test_shared_residual_head_v129 import cells


def expanded(task):
    result = []
    for c in cells(task):
        center = c['P'].mean(0) if task == 'classification' else np.median(c['P'], axis=0)
        result.append(c | {'P': np.concatenate([c['P'], center[None], center[None]])})
    return result


def test_eight_adaptive_recovers_six_model_when_extra_branches_have_zero_weight():
    rng = np.random.default_rng(1)
    old = rng.normal(size=18)
    new = np.concatenate([np.r_[old[:6], -10000., -10000.], np.r_[old[6:12], 0., 0.], np.r_[old[12:], 0., 0.]])
    for task in ('classification', 'regression'):
        for a, b in zip(cells(task), expanded(task)):
            np.testing.assert_allclose(six.predict(a, {'kind': 'adaptive', 'parameters': old}),
                eight.predict(b, {'kind': 'adaptive', 'parameters': new}), atol=1e-12, rtol=1e-12)


def test_eight_source_normalized_ridge_respects_the_intercept_solution():
    values = expanded('regression')
    model = eight.fit(values, {'kind': 'ridge', 'regularization': .01})
    A, Y, W = [], [], []
    for c, weight in zip(values, eight.weights(values)):
        A.append(np.c_[((c['P']-c['mu'])/c['sigma']).T, np.ones(len(c['y']))])
        Y.append((c['y']-c['mu'])/c['sigma'])
        W.append(weight*c['sigma']**2/max(float(c['y'].var()), 1e-12))
    A, Y, W = np.concatenate(A), np.concatenate(Y), np.concatenate(W)
    p = np.r_[model['beta'], model['intercept']]
    prior = np.r_[1., np.zeros(8)]
    np.testing.assert_allclose((A*W[:, None]).T@(A@p-Y)+.01*(p-prior), 0., atol=1e-10)


def test_group_risk_fits_eight_outputs_and_predicts_without_query_truth():
    torch.set_num_threads(1)
    for task in ('classification', 'regression'):
        values = expanded(task)
        kinds = ('adaptive', 'convex', 'loglinear') if task == 'classification' else ('adaptive', 'ridge')
        for kind in kinds:
            model = grouped.fit(values, {'kind': kind, 'regularization': .01})
            pred = eight.predict({k: v for k, v in values[0].items() if k != 'y'}, model)
            assert pred.shape == values[0]['P'][0].shape and np.isfinite(pred).all()
            if task == 'classification':
                np.testing.assert_allclose(pred.sum(1), 1., atol=1e-12)
