"""Recalibrate the reconstructed original five-context gated fusion formula.

No new context rule, domain feature, pseudo label, or residual model is added.
The optional Native8 backend changes internal estimator budget explicitly.
"""
import numpy as np
import tabace_fusion_v28 as fusion
import shared_task_residual_heads_v155 as fast

FAMILIES = ('cell_heavy_tail', 'column_desynchronization', 'feature_quantization')
BACKENDS = ('native1', 'native8_bank_native1_gate', 'native8_all')


def view(cell, backend, training=False):
    assert backend in BACKENDS
    base = cell['native8_base'] if backend == 'native8_all' else cell['P'][0]
    bank = cell['P'][1:] if backend == 'native1' else cell['native8_bank']
    value = {'task': cell['task'], 'base': base, 'bank': bank,
        'mu': cell['mu'], 'sigma': cell['sigma']}
    if training:
        value.update(y=cell['y'], role='validation', meta={'task_id': cell['task_id'],
            'data_id': cell['data_id'], 'scale': max(float(np.std(cell['y'])), 1e-12)})
    return value


def predict(cell, model):
    return fusion.predict(view(cell, model['backend']), model['fusion_policy'])


def fit(cells, specification):
    if specification['kind'] == 'base':
        policy = {'kind': 'base'}
    else:
        values = [view(c, specification['backend'], training=True) for c in cells]
        policy = fusion.fit(values, cells[0]['task'], 5, specification['gain'], specification['regularization'])
    return specification | {'fusion_policy': policy,
        'original_reconstructed_context_bank_and_fusion_formula_retained': True}


def score(cells, model):
    grouped = {}
    for cell in cells:
        grouped.setdefault(cell['data_id'], {}).setdefault(cell['family'], []).append(
            fast.metric(cell, predict(cell, model)))
    values = np.array([np.mean([np.mean(v, axis=0) for v in families.values()], axis=0)
        for families in grouped.values()])
    return {'primary': float(values[:, 0].mean()), 'secondary': float(values[:, 1].mean()), 'datasets': len(values)}


def reference(cells):
    return score(cells, {'backend': 'native1', 'fusion_policy': {'kind': 'base'}})


def candidates(task):
    result = []
    for backend in BACKENDS:
        result.append({'kind': 'base', 'backend': backend})
        for gain in (1., 2., 4.):
            for regularization in (.0001, .01):
                result.append({'kind': 'original_gated_fusion', 'backend': backend,
                    'gain': gain, 'regularization': regularization})
    assert len(result) == 21
    return result


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
        gains = {family: sum(n*v for n, v in values)/sum(n for n, v in values)
            for family, values in per_family.items()}
        trials.append({'specification': specification, 'inner': inner, 'family_gains': gains,
            'score': {'worst_family_gain': min(gains.values()), 'macro_family_gain': float(np.mean(list(gains.values())))}})
    best = max(trials, key=lambda t: (t['score']['worst_family_gain'], t['score']['macro_family_gain']))
    return fit(source, best['specification']), best, trials
