"""Domain-evidence-augmented fusion: observable source/query statistics, no family at deployment.

Family labels are used in calibration only, never deployment. Evaluated
dataset labels must be excluded by the caller. This is a new adaptation.
"""
import numpy as np
import torch
from scipy.optimize import minimize

import adapt_shared_heads_v119 as original
import shared_group_risk_v127 as grouped
import shared_task_residual_heads_v155 as fast
from scipy.special import softmax
from run_tabicl import normalize

TEMPERATURE = .02
FAMILIES = ('cell_heavy_tail', 'column_desynchronization', 'feature_quantization')


def tensors(cells, cap):
    task = cells[0]['task']
    groups = {}
    rng = np.random.default_rng(0)
    for family in sorted({c['family'] for c in cells}):
        values = [c for c in cells if c['family'] == family]
        fields = {k: [] for k in ('Q', 'S', 'P', 'Y', 'W', 'B', 'E')}
        pairs = {k: [] for k in ('posQ', 'posS', 'posP', 'posE', 'negQ', 'negS', 'negP', 'negE', 'W')}
        for c, w in zip(values, original.weights(values)):
            P, Q, S = original.components(c, cap)
            fields['Q'].append(Q)
            fields['S'].append(S)
            fields['E'].append(np.repeat(np.asarray(c['domain_evidence'])[None], len(c['y']), axis=0))
            if task == 'classification':
                truth = c['y'].astype(int)
                fields['P'].append(P[:, np.arange(len(truth)), truth].T)
                fields['B'].append(c['P'][0, np.arange(len(truth)), truth])
                fields['Y'].append(np.zeros(len(truth)))
                fields['W'].append(w)
                labels = np.unique(truth)
                for label in labels:
                    if len(labels) < 2:
                        continue
                    positive = rng.choice(np.flatnonzero(truth == label), 16, replace=True)
                    negative = rng.choice(np.flatnonzero(truth != label), 16, replace=True)
                    for side, ids in (('pos', positive), ('neg', negative)):
                        pairs[side+'Q'].append(Q[ids])
                        pairs[side+'S'].append(S[ids])
                        pairs[side+'P'].append(P[:, ids, int(label)].T)
                        pairs[side+'E'].append(np.repeat(np.asarray(c['domain_evidence'])[None], len(ids), axis=0))
                    pairs['W'].append(np.full(16, w.sum()/len(labels)/16))
            else:
                fields['P'].append(P.T)
                fields['B'].append((c['P'][0]-c['mu'])/c['sigma'])
                fields['Y'].append((c['y']-c['mu'])/c['sigma'])
                fields['W'].append(w*c['sigma']**2/max(float(c['y'].var()), 1e-12))
        fields = {k: torch.as_tensor(np.concatenate(v), dtype=torch.float64) for k, v in fields.items()}
        pairs = {k: torch.as_tensor(np.concatenate(v), dtype=torch.float64) for k, v in pairs.items()} if task == 'classification' else {}
        groups[family] = (fields, pairs)
    return groups


def fit(cells, specification):
    kind, task = specification['kind'], cells[0]['task']
    if kind != 'domain_evidence':
        return grouped.fit(cells, specification)
    if kind in ('base', 'mean', 'median', 'loglinear'):
        # Fixed candidates need no fit; loglinear remains a CE-trained candidate
        # selected by worst-family held-out calibration gain rather than mean gain.
        return original.fit(cells, specification)
    groups = tensors(cells, specification.get('cap'))
    regularization = specification['regularization']
    initial = np.r_[3., np.zeros(5), np.ones(12), np.zeros(60)]
    bounds = [(-10., 10.)]*18+[(-5., 5.)]*60
    prior = torch.as_tensor(initial, dtype=torch.float64)

    def mixed(p, Q, S, P, E):
        return (torch.softmax(p[:6]+Q*p[6:12]+S*p[12:18]+E@p[18:].reshape(10, 6), dim=1)*P).sum(1)

    def loss(p, values, base=False):
        t, pairs = values
        if base:
            prediction = t['B']
        elif kind == 'ridge':
            prediction = t['P']@p[:6]+p[-1]
        else:
            prediction = mixed(p, t['Q'], t['S'], t['P'], t['E'])
        if task == 'regression':
            return ((prediction-t['Y']).square()*t['W']).sum()
        value = -.5*(prediction.clamp(1e-12, 1.).log()*t['W']).sum()
        if base:
            pos, neg = pairs['posP'][:, 0], pairs['negP'][:, 0]
        else:
            pos = mixed(p, pairs['posQ'], pairs['posS'], pairs['posP'], pairs['posE'])
            neg = mixed(p, pairs['negQ'], pairs['negS'], pairs['negP'], pairs['negE'])
        delta = torch.logit(pos.clamp(1e-7, 1-1e-7))-torch.logit(neg.clamp(1e-7, 1-1e-7))
        return value+.5*(torch.nn.functional.softplus(-delta)*pairs['W']).sum()

    with torch.no_grad():
        baseline = {family: loss(prior, group, base=True) for family, group in groups.items()}
    nonclean = [f for f in groups if f != 'clean']
    assert nonclean

    def objective(values):
        p = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        regrets = {family: loss(p, group)-baseline[family] for family, group in groups.items()}
        vector = torch.stack([regrets[f] for f in nonclean])
        value = TEMPERATURE*(torch.logsumexp(vector/TEMPERATURE, dim=0)-np.log(len(nonclean)))
        if 'clean' in regrets:
            value += .1*regrets['clean']
        value += regularization*((p[:18]-prior[:18]).square().mean()+p[18:].square().mean())
        value.backward()
        return float(value.detach()), p.grad.detach().numpy()
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 240, 'ftol': 1e-9, 'gtol': 1e-6})
    assert np.isfinite(result.x).all()
    model = specification | {'risk_objective': 'softmax_family_excess_over_Base',
        'risk_temperature': TEMPERATURE, 'clean_relative_loss_weight': .1,
        'objective': float(result.fun), 'optimizer_success': bool(result.success)}
    if kind == 'ridge':
        model.update(beta=result.x[:6].tolist(), intercept=float(result.x[-1]))
    else:
        model['parameters'] = result.x.tolist()
    return model




def predict(cell, model):
    if model['kind'] != 'domain_evidence':
        return original.predict(cell, model)
    P, Q, S = original.components(cell, model.get('cap'))
    p = np.asarray(model['parameters'])
    evidence = np.asarray(cell['domain_evidence'])
    logits = p[:6]+Q*p[6:12]+S*p[12:18]+evidence@p[18:].reshape(10, 6)
    w = softmax(logits, axis=1)
    if cell['task'] == 'classification':
        return normalize(np.einsum('nr,rnc->nc', w, P))
    return (w*P.T).sum(axis=1)*cell['sigma']+cell['mu']


def score(cells, model):
    grouped_values = {}
    for cell in cells:
        value = fast.metric(cell, predict(cell, model))
        grouped_values.setdefault(cell['data_id'], {}).setdefault(cell['family'], []).append(value)
    values = np.array([np.mean([np.mean(v, axis=0) for v in families.values()], axis=0)
        for families in grouped_values.values()])
    return {'primary': float(values[:, 0].mean()), 'secondary': float(values[:, 1].mean()), 'datasets': len(values)}


def candidates(task):
    values = [{'kind': kind} for kind in ('base', 'mean', 'median')]
    values.append({'kind': 'adaptive', 'regularization': .01, 'cap': None})
    for regularization in (.001, .01, .1):
        for cap in ([None, 2.] if task == 'regression' else [None]):
            values.append({'kind': 'domain_evidence', 'regularization': regularization, 'cap': cap})
    return values


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
