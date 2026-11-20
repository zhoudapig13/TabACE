"""Other-dataset selection of source-trained task residual adapters."""
import numpy as np
from scipy.stats import rankdata

import adapt_shared_heads_v119 as original
import shared_group_risk_v127 as grouped
from run_tabicl import normalize

FAMILIES = grouped.FAMILIES


def predict(cell, model):
    prediction = original.predict(cell, model['global_head'])
    if model['gain'] == 0:
        return prediction
    correction = cell['task_corrections'][model['correction_index']]
    if cell['task'] == 'regression':
        return prediction+model['gain']*np.clip(correction, -1., 1.)*cell['sigma']
    return normalize(np.clip(prediction+model['gain']*np.clip(correction, -.5, .5), 1e-9, 1.))


def metric(cell, prediction):
    y = cell['y']
    if cell['task'] == 'classification':
        p = normalize(prediction)
        positive = np.arange(p.shape[1])[None] == y[:, None]
        count = positive.sum(axis=0)
        valid = (count > 0) & (count < len(y))
        ranks = rankdata(p, axis=0, method='average')
        numerator = (ranks*positive).sum(axis=0)-count*(count+1)/2
        auc = np.mean(numerator[valid]/(count[valid]*(len(y)-count[valid]))) if valid.any() else np.nan
        accuracy = float(np.mean(p.argmax(axis=1) == y))
        return [accuracy if not np.isfinite(auc) else float(auc), accuracy]
    mse = float(np.mean((prediction-y)**2))
    variance = max(float(np.var(y)), 1e-12/len(y))
    return [1-mse/variance, -np.sqrt(mse)/max(float(np.std(y)), 1e-12)]


def score(cells, model):
    grouped_values = {}
    for cell in cells:
        value = metric(cell, predict(cell, model))
        grouped_values.setdefault(cell['data_id'], {}).setdefault(cell['family'], []).append(value)
    values = np.array([np.mean([np.mean(v, axis=0) for v in families.values()], axis=0)
        for families in grouped_values.values()])
    return {'primary': float(values[:, 0].mean()), 'secondary': float(values[:, 1].mean()),
        'datasets': len(values)}


def candidates():
    values = []
    for kind in ('base', 'adaptive'):
        values.append({'global_kind': kind, 'gain': 0., 'correction_index': None})
        for index in range(6):
            for gain in (.25, .5, 1.):
                values.append({'global_kind': kind, 'gain': gain, 'correction_index': index})
    return values


def fit_global(source, kind):
    spec = {'kind': kind} if kind == 'base' else {'kind': 'adaptive', 'regularization': .01}
    return grouped.fit(source, spec)


def fit(source, specification):
    return specification | {'kind': 'task_residual',
        'global_head': fit_global(source, specification['global_kind']),
        'own_source_labels_used_for_local_kernel_adapter': True,
        'own_query_labels_required': False}


def choose(source):
    folds = sorted({c['fold'] for c in source})
    families = sorted({c['family'] for c in source if c['family'] != 'clean'})
    assert len(folds) == 4
    cache, baseline, trials = {}, {}, []
    for fold in folds:
        validation = [c for c in source if c['fold'] == fold]
        for family in families:
            baseline[fold, family] = score([c for c in validation if c['family'] == family],
                {'global_head': {'kind': 'base'}, 'gain': 0.})
    for spec in candidates():
        inner, per_family = [], {f: [] for f in families}
        for fold in folds:
            training = [c for c in source if c['fold'] != fold]
            validation = [c for c in source if c['fold'] == fold]
            train_ids, val_ids = {c['data_id'] for c in training}, {c['data_id'] for c in validation}
            assert train_ids.isdisjoint(val_ids)
            key = (fold, spec['global_kind'])
            if key not in cache:
                cache[key] = fit_global(training, spec['global_kind'])
            model = spec | {'global_head': cache[key]}
            scores = {}
            for family in families:
                values = [c for c in validation if c['family'] == family]
                a, b = score(values, model), baseline[fold, family]
                gain = a['primary']-b['primary']
                per_family[family].append((a['datasets'], gain))
                scores[family] = {'candidate': a, 'Base': b, 'gain': gain}
            inner.append({'fold': fold, 'model': model, 'scores_by_family': scores,
                'training_data_ids': sorted(train_ids), 'validation_data_ids': sorted(val_ids)})
        gains = {family: sum(n*v for n, v in values)/sum(n for n, v in values)
            for family, values in per_family.items()}
        trials.append({'specification': spec, 'inner': inner, 'family_gains': gains,
            'score': {'worst_family_gain': min(gains.values()),
                'macro_family_gain': float(np.mean(list(gains.values())))}})
    best = max(trials, key=lambda t: (t['score']['worst_family_gain'], t['score']['macro_family_gain']))
    return fit(source, best['specification']), best, trials
