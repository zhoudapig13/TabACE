import inspect

import numpy as np

import source_label_row_repair_guard_v241 as guard


def test_only_label_consistent_source_edits_survive_with_no_validation_label_callback():
    X = np.c_[np.arange(18.)/18, np.arange(18.)]
    y = X[:, 0]+1.
    mild, full = X.copy(), X.copy()
    mild[:, 0] += 1.
    full[:, 0] -= 1.
    calls = []
    def predict(A, b, T):
        assert not set(A[:, 1]) & set(T[:, 1])
        assert len(T) % 3 == 0
        ids = np.array_split(T[:, 1], 3)
        np.testing.assert_array_equal(ids[0], ids[1])
        np.testing.assert_array_equal(ids[0], ids[2])
        calls.append(1)
        return T[:, 0]
    result = guard.construct(X, y, [mild, full], 'regression', predict)
    assert len(calls) == 3 and result['accepted'][0].all() and not result['accepted'][1].any()
    np.testing.assert_array_equal(result['contexts'][0], mild)
    np.testing.assert_array_equal(result['contexts'][1], X)
    assert set(inspect.signature(guard.construct).parameters) == {
        'X_source', 'y_source', 'repaired_source_views', 'task', 'predict'}


def test_ties_and_rare_source_classes_keep_original_rows():
    X = np.c_[np.arange(13.), np.arange(13.)]
    y = np.r_[np.zeros(6), np.ones(6), 2].astype(int)
    def predict(A, b, T):
        assert 2 in b and 12 in A[:, 1] and 12 not in T[:, 1]
        return np.full((len(T), 3), 1/3)
    mild, full = X.copy(), X.copy()
    mild[:, 0] += 1.
    full[:, 0] += 2.
    r = guard.construct(X, y, [mild, full], 'classification', predict)
    assert not r['accepted'].any() and not r['scorable'][12]
    assert np.isnan(r['loss'][:, 12]).all()
    for out in r['contexts']:
        np.testing.assert_array_equal(out, X)


def test_all_rare_classes_need_no_source_probe_and_preserve_inputs():
    X = np.arange(12.).reshape(6, 2)
    def forbidden(*args):
        raise AssertionError('There are no eligible held-out source labels')
    r = guard.construct(X, np.arange(6), [X+1., X+2.], 'classification', forbidden)
    assert not r['scorable'].any() and not r['accepted'].any()
    np.testing.assert_array_equal(r['contexts'][0], X)
