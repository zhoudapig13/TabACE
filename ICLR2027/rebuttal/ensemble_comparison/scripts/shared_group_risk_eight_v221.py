"""Conservative calibration: minimize worst-family excess risk over Base.

Family labels are used in calibration only, never deployment. Evaluated
dataset labels must be excluded by the caller. This is a new adaptation.
"""
import numpy as np
import torch
from scipy.optimize import minimize

import adapt_shared_heads_eight_v220 as original

TEMPERATURE = .02
FAMILIES = ('cell_heavy_tail', 'column_desynchronization', 'feature_quantization')


def tensors(cells, cap):
    task = cells[0]['task']
    groups = {}
    rng = np.random.default_rng(0)
    for family in sorted({c['family'] for c in cells}):
        values = [c for c in cells if c['family'] == family]
        fields = {k: [] for k in ('Q', 'S', 'P', 'Y', 'W', 'B')}
        pairs = {k: [] for k in ('posQ', 'posS', 'posP', 'negQ', 'negS', 'negP', 'W')}
        for c, w in zip(values, original.weights(values)):
            P, Q, S = original.components(c, cap)
            fields['Q'].append(Q)
            fields['S'].append(S)
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
    if kind in ('base', 'mean', 'median', 'loglinear'):
        # Fixed candidates need no fit; loglinear remains a CE-trained candidate
        # selected by worst-family held-out calibration gain rather than mean gain.
        return original.fit(cells, specification)
    groups = tensors(cells, specification.get('cap'))
    regularization = specification['regularization']
    if kind == 'ridge':
        initial = np.r_[1., np.zeros(8)]
        bounds = [(-5., 5.)]*9
    else:
        initial = np.r_[3., np.zeros(7), np.ones(16)]
        bounds = [(-10., 10.)]*24
        if kind == 'convex':
            initial[8:] = 0
            bounds[8:] = [(0., 0.)]*16
    prior = torch.as_tensor(initial, dtype=torch.float64)

    def mixed(p, Q, S, P):
        return (torch.softmax(p[:8]+Q*p[8:16]+S*p[16:24], dim=1)*P).sum(1)

    def loss(p, values, base=False):
        t, pairs = values
        if base:
            prediction = t['B']
        elif kind == 'ridge':
            prediction = t['P']@p[:8]+p[-1]
        else:
            prediction = mixed(p, t['Q'], t['S'], t['P'])
        if task == 'regression':
            return ((prediction-t['Y']).square()*t['W']).sum()
        value = -.5*(prediction.clamp(1e-12, 1.).log()*t['W']).sum()
        if base:
            pos, neg = pairs['posP'][:, 0], pairs['negP'][:, 0]
        else:
            pos = mixed(p, pairs['posQ'], pairs['posS'], pairs['posP'])
            neg = mixed(p, pairs['negQ'], pairs['negS'], pairs['negP'])
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
        value += regularization*(p-prior).square().mean()
        value.backward()
        return float(value.detach()), p.grad.detach().numpy()
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6})
    assert np.isfinite(result.x).all()
    model = specification | {'risk_objective': 'softmax_family_excess_over_Base',
        'risk_temperature': TEMPERATURE, 'clean_relative_loss_weight': .1,
        'objective': float(result.fun), 'optimizer_success': bool(result.success)}
    if kind == 'ridge':
        model.update(beta=result.x[:8].tolist(), intercept=float(result.x[-1]))
    else:
        model['parameters'] = result.x.tolist()
    return model


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
                a = original.score(values, model)
                b = original.score(values, {'kind': 'base'})
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
