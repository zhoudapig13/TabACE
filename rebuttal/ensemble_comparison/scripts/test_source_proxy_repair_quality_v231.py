import inspect

import numpy as np

import source_proxy_repair_quality_v231 as proxy


def test_source_validation_labels_excluded_and_input_proxy_shared():
    X = np.c_[np.arange(18.)/18, np.arange(18.)]
    y = X[:, 0]**2
    mild, full = X.copy(), X.copy()
    mild[:, 0] += .1
    full[:, 0] += .2
    calls = []
    def predict(A, b, T):
        assert not set(A[:, 1]) & set(T[:, 1])
        np.testing.assert_array_equal(T[:, 0], full[T[:, 1].astype(int), 0])
        calls.append((A.copy(), b.copy(), T.copy()))
        return np.full(len(T), b.mean())
    result = proxy.quality(X, y, [mild, full], 'regression', predict)
    assert len(calls) == 9
    np.testing.assert_array_equal(result['loss'][0], result['loss'][1])
    np.testing.assert_array_equal(result['loss'][0], result['loss'][2])
    assert 'y_target' not in inspect.signature(proxy.quality).parameters


def test_rare_source_classes_stay_in_training_and_are_never_scored():
    X = np.c_[np.arange(13.), np.arange(13.)]
    y = np.r_[np.zeros(6), np.ones(6), 2].astype(int)
    def predict(A, b, T):
        assert 2 in b and 12 in A[:, 1] and 12 not in T[:, 1]
        return np.full((len(T), 3), 1/3)
    r = proxy.quality(X, y, [X.copy(), X.copy()], 'classification', predict)
    assert not r['scored'][12] and r['scored'].sum() == 12
    assert np.isnan(r['loss'][:, 12]).all()
