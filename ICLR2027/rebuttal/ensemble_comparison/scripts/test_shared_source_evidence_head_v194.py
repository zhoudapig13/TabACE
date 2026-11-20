import numpy as np
import shared_source_evidence_head_v194 as head
from test_shared_residual_head_v129 import cells


def test_prediction_uses_evidence_but_does_not_require_outcomes():
    c = cells('classification')[0]
    c['domain_evidence'] = np.r_[1., np.zeros(6)]
    p = np.zeros(60)
    p[19] = 3.
    model = {'kind': 'domain_evidence', 'parameters': p.tolist()}
    q = head.predict({k: v for k, v in c.items() if k != 'y'}, model)
    np.testing.assert_allclose(q.sum(axis=1), 1., atol=1e-12)
    c['domain_evidence'][0] = -1.
    other = head.predict(c, model)
    assert np.max(np.abs(q-other)) > 1e-5


def test_conditional_routing_transfers_to_held_out_synthetic_datasets():
    rng = np.random.default_rng(188)
    training, validation = [], []
    for did in range(10):
        for sign in (-1., 1.):
            y = rng.normal(size=64)
            P = np.stack([y+(.35 if sign > 0 else 0.),
                y+(-.35 if sign < 0 else 0.)]+[y+.55]*4)
            c = {'task': 'regression', 'data_id': did, 'fold': did % 5,
                'family': 'cell_heavy_tail' if sign > 0 else 'column_desynchronization',
                'rho': 50, 'P': P, 'y': y, 'mu': 0., 'sigma': 1.,
                'domain_evidence': np.r_[sign, np.zeros(6)]}
            (training if did < 8 else validation).append(c)
    model = head.fit(training, {'kind': 'domain_evidence', 'regularization': .001})
    assert set(c['data_id'] for c in training).isdisjoint(c['data_id'] for c in validation)
    assert head.score(validation, model)['primary'] > head.score(validation, {'kind': 'base'})['primary']+.02


def test_classification_training_has_finite_normalized_outcome_free_predictions():
    data = cells('classification')
    for i, c in enumerate(data):
        c['domain_evidence'] = np.r_[i/len(data), np.zeros(6)]
    model = head.fit(data, {'kind': 'domain_evidence', 'regularization': .01, 'cap': None})
    c = {k: v for k, v in data[0].items() if k != 'y'}
    q = head.predict(c, model)
    assert np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(axis=1), 1., atol=1e-12)
