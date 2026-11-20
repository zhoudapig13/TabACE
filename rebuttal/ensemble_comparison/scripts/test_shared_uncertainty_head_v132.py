import numpy as np
import shared_uncertainty_head_v132 as head
from test_shared_residual_head_v129 import cells


def sample():
    source = cells('regression')
    rng = np.random.default_rng(42)
    return [c | {'IQR': rng.uniform(.1, 2, c['P'].shape)} for c in source]


def test_uncertainty_gate_does_not_require_query_labels():
    source = sample()
    model = head.fit(source, {'kind': 'iqr_adaptive', 'regularization': .01})
    query = {k: v for k, v in source[0].items() if k != 'y'}
    prediction = head.predict(query, model)
    assert prediction.shape == (80,) and np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction, head.predict(query | {'y': np.zeros(80)}, model))


def test_uncertainty_residual_head_is_finite_and_bounded():
    source = sample()
    model = head.fit(source, {'kind': 'residual_hgb', 'max_leaf_nodes': 7, 'strength': .5})
    query = {k: v for k, v in source[0].items() if k != 'y'}
    prediction = head.predict(query, model)
    assert prediction.shape == (80,) and np.isfinite(prediction).all()
    assert np.max(np.abs(prediction-query['P'][0])) <= .5+1e-12
