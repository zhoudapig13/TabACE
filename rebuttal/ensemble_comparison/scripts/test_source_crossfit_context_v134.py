import numpy as np
import source_crossfit_context_v134 as rule


def test_rare_classes_never_leak_into_held_out_score():
    y = np.array([0]*10+[1]*6+[2]*2+[3])
    partitions, mandatory = rule.folds(y, 'classification')
    assert set(mandatory) == {16, 17, 18}
    held_out = []
    for training, validation in partitions:
        assert set(training).isdisjoint(validation)
        assert set(mandatory).issubset(training)
        assert set(y[validation]).issubset(y[training])
        held_out.extend(validation)
    assert sorted(held_out) == list(range(16))


def test_callback_cannot_observe_held_out_source_labels():
    X = np.arange(30, dtype=float)[:, None]
    y = np.sin(X[:, 0])
    def predict(A, b, Q):
        assert set(A[:, 0]).isdisjoint(Q[:, 0])
        assert len(A) == len(b)
        return np.full(len(Q), b.mean())
    loss, prediction, mandatory, provenance = rule.quality(X, y, 'regression', predict)
    assert np.isfinite(loss).all() and np.isfinite(prediction).all()
    assert len(mandatory) == 0 and len(provenance) == 3


def test_selection_preserves_classes_and_prefers_reliable_rows():
    y = np.repeat(np.arange(3), 8)
    loss = np.tile(np.arange(8), 3).astype(float)
    ids = rule.select(y, loss, 'classification', .75)
    assert len(ids) == 18 and set(y[ids]) == {0, 1, 2}
    assert np.all(loss[ids] <= 5)
    assert len(np.unique(ids)) == len(ids)
