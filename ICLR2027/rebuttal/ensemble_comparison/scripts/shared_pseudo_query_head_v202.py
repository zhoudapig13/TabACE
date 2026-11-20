"""Fusion of pseudo contexts with an explicit raw-teacher fallback control."""
import numpy as np
import shared_source_evidence_head_v194 as previous

original = previous.original
grouped = previous.grouped
fast = previous.fast
FAMILIES = previous.FAMILIES


def fit(cells, specification):
    return specification.copy() if specification['kind'] == 'teacher' else previous.fit(cells, specification)


def predict(cell, model):
    return cell['teacher'] if model['kind'] == 'teacher' else previous.predict(cell, model)


def score(cells, model):
    groups = {}
    for cell in cells:
        groups.setdefault(cell['data_id'], {}).setdefault(cell['family'], []).append(fast.metric(cell, predict(cell, model)))
    values = np.array([np.mean([np.mean(v, axis=0) for v in families.values()], axis=0) for families in groups.values()])
    return {'primary': float(values[:, 0].mean()), 'secondary': float(values[:, 1].mean()), 'datasets': len(values)}


def candidates(task):
    return original.candidates(task)+[s for s in previous.candidates(task) if s['kind'] == 'domain_evidence']+[{'kind': 'teacher'}]


def choose(source):
    folds = sorted({c['fold'] for c in source})
    families = sorted({c['family'] for c in source if c['family'] != 'clean'})
    assert len(folds) == 4
    trials = []
    for specification in candidates(source[0]['task']):
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
