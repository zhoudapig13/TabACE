"""Select source-prior TFM readouts with both disjoint validation domains.

This is supervised method development, not original zero-shot Shared. Evaluation
labels are deliberately absent from this API. Cached OOF predictions may be
reused only from the identical training folds and class-weighted v357 fits.
"""
import numpy as np
from run_tabicl import normalize
from source_prior_probe_adapter_v357 import PENALTIES, fit, predict


def score(target_y, target_p, source_y, source_p, source_prior):
    target_y = np.asarray(target_y, dtype=int)
    source_y = np.asarray(source_y, dtype=int)
    target_p, source_p = normalize(target_p), normalize(source_p)
    prior = np.asarray(source_prior, dtype=float)
    observed = (np.bincount(source_y, minlength=len(prior)) + 8. * prior) / (len(source_y) + 8.)
    weights = (prior / np.maximum(observed, 1e-12))[source_y]
    weights /= weights.sum()
    target_acc = float(np.mean(target_p.argmax(axis=1) == target_y))
    source_acc = float(np.dot(weights, source_p.argmax(axis=1) == source_y))
    target_nll = -float(np.log(np.maximum(target_p[np.arange(len(target_y)), target_y], 1e-12)).mean())
    source_nll = -float(np.dot(weights, np.log(np.maximum(source_p[np.arange(len(source_y)), source_y], 1e-12))))
    return (target_acc, -.5 * (target_nll + source_nll), source_acc)


def training_weights(reference_y, target_y, prior_probabilities, K, source_prior=None):
    labels = np.r_[reference_y, target_y].astype(int)
    per_target = max(4., len(reference_y) / max(1, len(target_y)))
    base = np.r_[np.ones(len(reference_y)), np.full(len(target_y), per_target)]
    desired = np.asarray(source_prior, dtype=float)
    assert len(desired)==K and np.all(desired>=0) and abs(desired.sum()-1.)<1e-10
    observed = np.bincount(labels, weights=base, minlength=K) / base.sum()
    ratios = np.clip(desired / np.maximum(observed, 1e-12), .05, 20.)
    weights = base * ratios[labels]
    # Hold total loss mass fixed to the v413 baseline so the changed factor is
    # relative domain influence, rather than effective regularization strength.
    weights *= (len(reference_y) + 4.*len(target_y)) / weights.sum()
    return weights, {'desired_prior':desired.tolist(), 'observed_weighted_prior':observed.tolist(),
                     'class_factors':ratios.tolist(), 'per_target_weight_before_class_correction':per_target,
                     'normalized_total_weight':float(weights.sum())}


def adapt(F, raw, reference_y, source_validation_y, adaptation_y, apos, epos,
          source_prior=None, cached_oof=None):
    assert source_prior is not None
    F, raw = np.asarray(F, dtype=float), normalize(raw)
    K = raw.shape[1]
    m, s = len(reference_y), len(source_validation_y)
    offset = m + s
    folds = np.arange(len(apos)) % 4
    reference, source_val = np.arange(m), np.arange(m, offset)
    target, evaluation = offset + apos, offset + epos
    measure = lambda pt, ps: score(adaptation_y, pt, source_validation_y, ps, source_prior)
    best = {'kind': 'raw', 'C': None, 'alpha': 0.,
            'score': list(measure(raw[target], raw[source_val]))}
    trials, oof = [], {}
    for C in PENALTIES:
        if cached_oof is not None:
            cv = np.asarray(cached_oof[str(C)]['target']).copy()
            cal = np.asarray(cached_oof[str(C)]['source_validation']).copy()
            assert cv.shape == (len(apos), K) and cal.shape == (s, K)
        else:
            cv, cal = np.full((len(apos), K), np.nan), np.zeros((s, K))
            for fold in range(4):
                tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                rows = np.r_[reference, target[tr]]
                labels = np.r_[reference_y, adaptation_y[tr]]
                weights, _ = training_weights(reference_y, adaptation_y[tr], raw[target[tr]], K, source_prior)
                model = fit(F[rows], labels, C, weights)
                cv[va] = predict(model, F[target[va]], K)
                cal += predict(model, F[source_val], K) / 4.
        assert np.isfinite(cv).all() and np.isfinite(cal).all()
        oof[str(C)] = {'target': cv, 'source_validation': cal}
        for alpha in (.25, .5, .75, 1.):
            pt = normalize((1-alpha) * raw[target] + alpha * cv)
            ps = normalize((1-alpha) * raw[source_val] + alpha * cal)
            value = measure(pt, ps)
            trials.append({'C': C, 'alpha': alpha, 'score': list(value)})
            if value > tuple(best['score']):
                best = {'kind': 'linear_probe', 'C': C, 'alpha': alpha, 'score': list(value)}
    model, weight_info = None, None
    if best['kind'] == 'linear_probe':
        rows, labels = np.r_[reference, target], np.r_[reference_y, adaptation_y]
        weights, weight_info = training_weights(reference_y, adaptation_y, raw[target], K, source_prior)
        model = fit(F[rows], labels, best['C'], weights)
        final = normalize((1-best['alpha']) * raw[evaluation] + best['alpha'] * predict(model, F[evaluation], K))
    else:
        final = raw[evaluation].copy()
    best.update(trials=trials, model=model, reference_training_rows=m,
                source_validation_rows=s, target_adaptation_rows=len(target),
                target_training_weight=max(4., m/len(target)), class_prior_weighting=weight_info,
                total_loss_weight_normalization='source_rows+4*target_rows',
                domain_balance_before_class_prior_weighting=True,
                prior_origin='observed_source_labels', source_prior=np.asarray(source_prior).tolist(),
                target_prior_smoothing_mass=0., class_weight_ratio_clip=[.05, 20.],
                source_validation_prior_smoothing_mass=8.,
                selection='target OOF accuracy; equal-domain source-prior-weighted NLL tie-break; source validation accuracy third')
    return final, best, oof
