"""One declared source-only calibration diagnostic; preserve first-round results."""
import argparse
from datetime import datetime, timezone
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import run_public_shift as exp

DATA = exp.ROOT/'data/processed/public_casecontrol_v1'
OUT = exp.ROOT/'results/public_casecontrol_crossfit_v1'


def select(base, bank, y, weights, m, fallback, fallback_oof, folds):
    best = float(np.average((fallback_oof >= fallback['threshold']) == y, weights=weights))
    selected, selected_oof = None, fallback_oof
    trials = [dict(candidate='fallback', source_oof_accuracy=best)]
    for gain in [1., 2.]:
        for reg in [.01, .1]:
            oof = np.zeros(len(y))
            for fit, val in folds:
                p = exp.fit_gate(base[fit], bank[:, fit], y[fit], weights[fit], m, gain, reg)
                oof[val] = exp.probability(base[val], bank[:, val], p)
            threshold = exp.threshold(oof, y, weights)
            score = float(np.average((oof >= threshold) == y, weights=weights))
            trials.append(dict(gain=gain, regularization=reg, threshold=threshold, source_oof_accuracy=score))
            if score > best+1e-12:
                best = score
                selected = dict(gain=gain, regularization=reg, threshold=threshold)
                selected_oof = oof
    if selected is None:
        return fallback, fallback_oof, trials
    final = exp.fit_gate(base, bank, y, weights, m, selected['gain'], selected['regularization'])
    final['threshold'] = selected['threshold']
    return final, selected_oof, trials


def run(task):
    torch.set_num_threads(1)
    source, dest = DATA/task, OUT/task
    with np.load(source/'inputs.npz') as z:
        pack = {k: z[k] for k in z.files}
    with np.load(source/'member_predictions.npz') as z:
        bank = {k: z[k] for k in z.files}
    # Domain-density estimator sees covariates, never target clinical outcomes.
    _, _, density = exp.construct(pack['X_train'], pack['y_train'], pack['X_ood'], pack['align_columns'])
    y, base, members = pack['y_cal'], bank['base_cal'], bank['bank_cal']
    weights = np.sqrt(density(pack['X_cal']))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(base, y))
    baseline = dict(kind='base', threshold=exp.threshold(base, y, np.ones(len(y))))
    shared, shared_oof, st = select(base, members, y, np.ones(len(y)), 5, baseline, base, folds)
    specific, _, pt = select(base, members, y, weights, 8, shared, shared_oof, folds)
    policies = dict(Base=baseline, Shared=shared, Specific=specific)
    exp.write_json(dest/'frozen_policies.json', dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), test_labels_opened=False,
        policies=policies, source_calibration_trials=dict(Shared=st, Specific=pt),
        target_labels_used_for_fitting=False, protocol=json.loads((exp.ROOT/'configs/public_casecontrol_crossfit_protocol.json').read_text()),
        source_oof_is_development_selection_not_unbiased_performance=True))
    with np.load(source/'test_truth.npz') as z:
        truth = {k: z[k] for k in z.files}
    result = dict(task=task, seed=0, domains={}, exploratory_development=True,
                  previously_observed_holdout_cohort=True, target_labels_used_for_fitting=False)
    for domain in ['id', 'ood']:
        target_y = truth['y_'+domain]
        ps = {m: exp.probability(bank['base_'+domain], bank['bank_'+domain], p) for m, p in policies.items()}
        methods = {m: exp.metrics(target_y, p, policies[m]['threshold']) for m, p in ps.items()}
        tests = {b+'_minus_'+a: exp.paired(target_y, ps[a] >= policies[a]['threshold'], ps[b] >= policies[b]['threshold'])
                 for a, b in [('Base', 'Shared'), ('Shared', 'Specific')]}
        result['domains'][domain] = dict(n=len(target_y), positives=int(target_y.sum()),
                                         methods=methods, adjacent_comparisons=tests)
        print('CROSSFIT_RESULT', task, domain, {m: s['accuracy'] for m, s in methods.items()}, flush=True)
    exp.write_json(dest/'metrics.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    run(parser.parse_args().task)
