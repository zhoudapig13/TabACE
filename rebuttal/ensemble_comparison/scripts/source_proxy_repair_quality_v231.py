"""Compare source contexts on a shared, unlabeled-manifold-repaired CV query.

Existing source labels only. The callback never receives validation labels.
Every candidate predicts the SAME repaired validation features, so the
comparison varies training-context repair rather than validation inputs.
"""
import numpy as np

from source_crossfit_context_v134 import folds


def quality(X_source, y_source, repaired_source_views, task, predict):
    X, y = np.asarray(X_source), np.asarray(y_source)
    assert len(repaired_source_views) == 2
    views = [X]+[np.asarray(v) for v in repaired_source_views]
    assert all(v.shape == X.shape for v in views)
    partitions, mandatory = folds(y, task)
    shape = (3, len(y), len(np.unique(y))) if task == 'classification' else (3, len(y))
    predictions = np.full(shape, np.nan)
    loss = np.full((3, len(y)), np.nan)
    records = []
    for training, validation in partitions:
        target_like_validation = views[2][validation]
        for i, view in enumerate(views):
            q = predict(view[training], y[training], target_like_validation)
            assert np.isfinite(q).all()
            predictions[i, validation] = q
            if task == 'classification':
                loss[i, validation] = -np.log(np.clip(q[np.arange(len(validation)), y[validation].astype(int)], 1e-12, 1.))
            else:
                loss[i, validation] = ((q-y[validation])/max(float(y.std()), 1e-12))**2
        records.append({'training_rows': training.tolist(), 'validation_rows': validation.tolist()})
    scored = np.ones(len(y), dtype=bool)
    scored[mandatory] = False
    assert np.isfinite(loss[:, scored]).all()
    assert np.isnan(loss[:, ~scored]).all()
    return {'predictions': predictions, 'loss': loss, 'scored': scored,
        'mandatory_unscored_rows': mandatory, 'partitions': records,
        'same_full_repair_validation_features_for_all_candidates': True}
