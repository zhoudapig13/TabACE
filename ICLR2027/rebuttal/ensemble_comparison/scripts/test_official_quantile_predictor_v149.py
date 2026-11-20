from official_quantile_predictor_v149 import create


def test_only_official_normalization_configuration_is_adapted():
    for task in ('classification', 'regression'):
        model = create(task, 'cpu').models[1]
        params = model.get_params()
        assert params['norm_methods'] == 'quantile'
        assert params['n_estimators'] == 1 and params['random_state'] == 0
        assert params['allow_auto_download'] is False
        assert params['device'] == 'cpu' and params['batch_size'] == 1
