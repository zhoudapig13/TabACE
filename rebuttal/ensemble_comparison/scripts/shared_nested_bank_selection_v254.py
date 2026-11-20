"""Nested dataset-OOF selection between the preserved six-output parent and
the eight-output local extension. No view is chosen using own query labels.

Keeping a separate parent candidate matters: adding outputs also changes the
support features of old outputs, so zero new-output weights do not generally
recover the old confidence/support fusion model.
"""
import numpy as np

import shared_fast_group_risk_v179 as six
import shared_fast_group_risk_eight_v222 as eight

FAMILIES = six.FAMILIES
VIEWS = ('parent6', 'query_local8')
original, grouped, fast = eight.original, eight.grouped, eight.fast


def view(cell, name):
    assert name in VIEWS and cell['P'].shape[0] == 8
    return cell | {'P': cell['P'][:6] if name == 'parent6' else cell['P']}


def candidates(task):
    return [spec | {'predictor_view': name} for name in VIEWS for spec in six.original.candidates(task)]


def fit(cells, specification):
    name = specification['predictor_view']
    backend = six if name == 'parent6' else eight
    spec = {k: v for k, v in specification.items() if k != 'predictor_view'}
    model = backend.fit([view(c, name) for c in cells], spec)
    return specification | {'head_model': model, 'predictors': 6 if name == 'parent6' else 8}


def predict(cell, model):
    name = model['predictor_view']
    backend = six if name == 'parent6' else eight
    return backend.predict(view(cell, name), model['head_model'])


def score(cells, model):
    groups = {}
    for c in cells:
        groups.setdefault(c['data_id'], {}).setdefault(c['family'], []).append(fast.metric(c, predict(c, model)))
    values = np.array([np.mean([np.mean(v, axis=0) for v in families.values()], axis=0)
        for families in groups.values()])
    return {'primary': float(values[:, 0].mean()), 'secondary': float(values[:, 1].mean()), 'datasets': len(values)}


def reference(cells):
    return score(cells, {'predictor_view': 'parent6', 'head_model': {'kind': 'base'}})


def choose(source):
    folds = sorted({c['fold'] for c in source})
    families = sorted({c['family'] for c in source if c['family'] != 'clean'})
    assert len(folds) == 4
    trials = []
    for specification in candidates(source[0]['task']):
        inner, per_family = [], {f: [] for f in families}
        for fold in folds:
            training = [c for c in source if c['fold'] != fold]
            validation = [c for c in source if c['fold'] == fold]
            train_ids, valid_ids = {c['data_id'] for c in training}, {c['data_id'] for c in validation}
            assert train_ids.isdisjoint(valid_ids)
            model = fit(training, specification)
            scores = {}
            for family in families:
                values = [c for c in validation if c['family'] == family]
                a, b = score(values, model), reference(values)
                gain = a['primary']-b['primary']
                per_family[family].append((a['datasets'], gain))
                scores[family] = {'candidate': a, 'Base': b, 'gain': gain}
            inner.append({'fold': fold, 'model': model, 'scores_by_family': scores,
                'training_data_ids': sorted(train_ids), 'validation_data_ids': sorted(valid_ids)})
        gains = {f: sum(n*v for n, v in values)/sum(n for n, v in values) for f, values in per_family.items()}
        trials.append({'specification': specification, 'inner': inner, 'family_gains': gains,
            'score': {'worst_family_gain': min(gains.values()), 'macro_family_gain': float(np.mean(list(gains.values())))}})
    best = max(trials, key=lambda t: (t['score']['worst_family_gain'], t['score']['macro_family_gain']))
    return fit(source, best['specification']), best, trials
