import numpy as np
from official_native_none_predictor_v208 import create
from tabicl._sklearn.preprocessing import EnsembleGenerator


def test_only_documented_normalizer_and_estimator_count_are_configured():
    for task in ('classification', 'regression'):
        params = create(task, 'cpu').models[8].get_params()
        assert params['norm_methods'] == 'none' and params['n_estimators'] == 8
        assert params['random_state'] == 0 and params['allow_auto_download'] is False


def test_official_generator_uses_none_only_and_records_fewer_available_views():
    rng = np.random.default_rng(208)
    X, y = rng.normal(size=(50, 12)), rng.normal(size=50)
    default = EnsembleGenerator(False, 8, random_state=0).fit(X, y)
    adapted = EnsembleGenerator(False, 8, norm_methods='none', random_state=0).fit(X, y)
    assert {k: len(v) for k, v in default.ensemble_configs_.items()} == {'none': 4, 'power': 4}
    assert set(adapted.ensemble_configs_) == {'none'}
    assert sum(len(v) for v in adapted.ensemble_configs_.values()) == 8
    small = EnsembleGenerator(False, 8, norm_methods='none', random_state=0).fit(X[:, :2], y)
    assert 0 < sum(len(v) for v in small.ensemble_configs_.values()) <= 8
