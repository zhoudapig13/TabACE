"""The same grouped-risk candidates, with direct rank-sum AUC scoring.

Selection remains dataset-OOF. Faster scoring does not add model parameters.
"""
import numpy as np
import adapt_shared_heads_v119 as original
import shared_group_risk_v127 as grouped
import shared_task_residual_heads_v155 as fast

FAMILIES = grouped.FAMILIES
fit = grouped.fit
predict = original.predict


def score(cells, model):
    groups = {}
    for c in cells:
        groups.setdefault(c['data_id'], {}).setdefault(c['family'], []).append(fast.metric(c, predict(c, model)))
    values = np.array([np.mean([np.mean(v, axis=0) for v in families.values()], axis=0)
        for families in groups.values()])
    return {'primary': float(values[:, 0].mean()), 'secondary': float(values[:, 1].mean()), 'datasets': len(values)}


def choose(source):
    folds = sorted({c['fold'] for c in source})
    families = sorted({c['family'] for c in source if c['family'] != 'clean'})
    assert len(folds) == 4
    trials = []
    for specification in original.candidates(source[0]['task']):
        inner = []
        per_family = {f: [] for f in families}
        for fold in folds:
            training = [c for c in source if c['fold'] != fold]
            validation = [c for c in source if c['fold'] == fold]
            train_ids, val_ids = {c['data_id'] for c in training}, {c['data_id'] for c in validation}
            assert train_ids.isdisjoint(val_ids)
            model = fit(training, specification)
            family_scores = {}
            for family in families:
                values = [c for c in validation if c['family'] == family]
                a = score(values, model)
                b = score(values, {'kind': 'base'})
                gain = a['primary']-b['primary']
                per_family[family].append((a['datasets'], gain))
                family_scores[family] = {'candidate': a, 'Base': b, 'gain': gain}
            inner.append({'fold': fold, 'model': model, 'scores_by_family': family_scores,
                'training_data_ids': sorted(train_ids), 'validation_data_ids': sorted(val_ids)})
        gains = {f: sum(n*v for n, v in scores)/sum(n for n, v in scores) for f, scores in per_family.items()}
        trials.append({'specification': specification, 'inner': inner, 'family_gains': gains,
            'score': {'worst_family_gain': min(gains.values()), 'macro_family_gain': float(np.mean(list(gains.values())))}})
    best = max(trials, key=lambda t: (t['score']['worst_family_gain'], t['score']['macro_family_gain']))
    return fit(source, best['specification']), best, trials
