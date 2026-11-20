"""Nested selection of Native-1/Native-8 canonical gate and generic context fusion.

Both canonical gates keep six contexts but have different internal estimator budgets (6 versus up to13). The reference for selection
is always the unchanged original Native-1 Base, not whichever gate is chosen.
"""
import numpy as np

import adapt_shared_heads_v119 as original
import shared_group_risk_v127 as grouped

FAMILIES = grouped.FAMILIES


def view(cell, mode):
    if mode == 'original':
        return cell
    assert mode == 'native8'
    return cell | {'P': np.concatenate([cell['canonical_native8'][None], cell['P'][1:]], axis=0)}


def predict(cell, model):
    return original.predict(view(cell, model['canonical_mode']), model)


def fit(cells, specification):
    converted = [view(c, specification['canonical_mode']) for c in cells]
    return grouped.fit(converted, specification)


def score(cells, model):
    converted = [view(c, model['canonical_mode']) for c in cells]
    return original.score(converted, model)


def candidates(task):
    return [spec | {'canonical_mode': mode} for mode in ('original', 'native8')
        for spec in original.candidates(task)]


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
            train_ids, val_ids = {c['data_id'] for c in training}, {c['data_id'] for c in validation}
            assert train_ids.isdisjoint(val_ids)
            model = fit(training, specification)
            scores = {}
            for family in families:
                values = [c for c in validation if c['family'] == family]
                a = score(values, model)
                b = original.score(values, {'kind': 'base'})
                gain = a['primary']-b['primary']
                per_family[family].append((a['datasets'], gain))
                scores[family] = {'candidate': a, 'Base': b, 'gain': gain}
            inner.append({'fold': fold, 'model': model, 'scores_by_family': scores,
                'training_data_ids': sorted(train_ids), 'validation_data_ids': sorted(val_ids)})
        gains = {family: sum(n*v for n, v in values)/sum(n for n, v in values)
            for family, values in per_family.items()}
        trials.append({'specification': specification, 'inner': inner, 'family_gains': gains,
            'score': {'worst_family_gain': min(gains.values()),
                'macro_family_gain': float(np.mean(list(gains.values())))}})
    best = max(trials, key=lambda t: (t['score']['worst_family_gain'], t['score']['macro_family_gain']))
    return fit(source, best['specification']), best, trials
