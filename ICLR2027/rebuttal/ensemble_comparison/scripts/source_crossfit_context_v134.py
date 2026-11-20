"""Source-only out-of-fold reliability filtering; explicitly adapted Shared."""
import numpy as np


def folds(y, task):
    y = np.asarray(y)
    rng = np.random.default_rng(0)
    assignment = np.full(len(y), -1, dtype=int)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)] if task == 'classification' else [np.arange(len(y))]
    for group in groups:
        if task == 'classification' and len(group) < 3:
            # Rare labels remain in all fitting folds and are never scored
            # as held-out examples. Their rows are mandatory in the context.
            continue
        for fold, ids in enumerate(np.array_split(rng.permutation(group), 3)):
            assignment[ids] = fold
    result = []
    for fold in range(3):
        validation = np.flatnonzero(assignment == fold)
        training = np.flatnonzero(assignment != fold)
        if len(validation):
            assert set(training).isdisjoint(validation)
            if task == 'classification':
                assert set(np.unique(y[validation])).issubset(np.unique(y[training]))
            result.append((training, validation))
    return result, np.flatnonzero(assignment == -1)


def quality(X_source, y_source, task, predict):
    """Callback receives only source-training rows and source-held-out X."""
    X, y = np.asarray(X_source), np.asarray(y_source)
    partitions, mandatory = folds(y, task)
    loss = np.full(len(y), np.nan)
    prediction = np.full((len(y), len(np.unique(y))), np.nan) if task == 'classification' else np.full(len(y), np.nan)
    provenance = []
    for training, validation in partitions:
        q = predict(X[training], y[training], X[validation])
        prediction[validation] = q
        if task == 'classification':
            loss[validation] = -np.log(np.clip(q[np.arange(len(validation)), y[validation].astype(int)], 1e-12, 1.))
        else:
            loss[validation] = ((q-y[validation])/max(float(y.std()), 1e-12))**2
        provenance.append({'training_rows': training.tolist(), 'validation_rows': validation.tolist()})
    loss[mandatory] = -1.
    assert np.isfinite(loss).all()
    return loss, prediction, mandatory, provenance


def select(y_source, loss, task, fraction):
    y, loss = np.asarray(y_source), np.asarray(loss)
    assert len(y) == len(loss) and np.isfinite(loss).all()
    assert 0 < fraction <= 1
    groups = [np.flatnonzero(y == label) for label in np.unique(y)] if task == 'classification' else [g for g in np.array_split(np.argsort(y, kind='stable'), 8) if len(g)]
    chosen = []
    for group in groups:
        count = max(1, int(np.ceil(fraction*len(group))))
        order = np.argsort(loss[group], kind='stable')
        chosen.extend(group[order[:count]].tolist())
    return np.sort(np.asarray(chosen, dtype=int))
