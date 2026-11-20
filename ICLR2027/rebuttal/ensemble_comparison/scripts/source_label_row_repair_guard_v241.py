"""Source-label consistency guard for individual repaired context rows.

A source-fold model is fitted to OBSERVED training features and labels. It
predicts observed/mild/full features of the SAME source-held-out rows in one
call. A repaired feature row is retained only if it lowers that row's held-out
task loss. True query outcomes and query predictions are absent from the API.
"""
import numpy as np

from source_crossfit_context_v134 import folds


def construct(X_source, y_source, repaired_source_views, task, predict):
    X, y = np.asarray(X_source), np.asarray(y_source)
    assert task in ('classification', 'regression')
    assert len(repaired_source_views) == 2
    views = [X]+[np.asarray(v) for v in repaired_source_views]
    assert all(v.shape == X.shape and np.isfinite(v).all() for v in views)
    partitions, mandatory = folds(y, task)
    shape = (3, len(y), len(np.unique(y))) if task == 'classification' else (3, len(y))
    predictions, loss = np.full(shape, np.nan), np.full((3, len(y)), np.nan)
    records = []
    for train, valid in partitions:
        probe = np.concatenate([view[valid] for view in views], axis=0)
        raw = np.asarray(predict(X[train], y[train], probe))
        assert len(raw) == 3*len(valid) and np.isfinite(raw).all()
        scored = raw.reshape((3, len(valid))+raw.shape[1:])
        predictions[:, valid] = scored
        if task == 'classification':
            probability = scored[:, np.arange(len(valid)), y[valid].astype(int)]
            loss[:, valid] = -np.log(np.clip(probability, 1e-12, 1.))
        else:
            loss[:, valid] = ((scored-y[valid])/max(float(y.std()), 1e-12))**2
        records.append({'training_rows': train.tolist(), 'validation_rows': valid.tolist(),
            'probe_order': ['observed', 'mild_repair', 'full_repair']})
    scorable = np.ones(len(y), dtype=bool)
    scorable[mandatory] = False
    accepted = (loss[1:] < loss[0][None]) & scorable[None]
    output = []
    for index, active in enumerate(accepted):
        guarded = X.copy()
        guarded[active] = views[1+index][active]
        output.append(guarded)
    assert np.isfinite(loss[:, scorable]).all()
    assert not accepted[:, mandatory].any()
    return {'contexts': output, 'accepted': accepted, 'predictions': predictions, 'loss': loss,
        'scorable': scorable, 'mandatory_unscored_rows': mandatory, 'partitions': records}
