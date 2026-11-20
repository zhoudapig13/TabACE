"""Pinned official PHE/RF implementations and qualified PFN adaptations.

Outer seed is zero. Member streams/configurations are deterministic children.
Benchmark query labels enter only the metrics function after predictions.
"""
import argparse
from copy import deepcopy
from contextlib import contextmanager
import functools
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from prepare_data import RUN, atomic_json, protocol
from tabpfn_official_benchmark_v2 import OUT, PLAN, tasks, conditions, frozen_input, canonical
sys.path.insert(0, str(RUN / 'external/pfn_official_dependencies_v2'))
from run_tabpfn import model, fit_predict, aligned, normalize, micp_no_capfn, CLASS_CKPT, REG_CKPT
from run_tabpfn import AutoPostHocEnsemblePredictor, TabPFNBaseModelSource, TaskType
from tabpfn_extensions.post_hoc_ensembles.pfn_phe import _get_base_models_from_random_search
from tabpfn_extensions.rf_pfn.SklearnBasedRandomForestTabPFN import RandomForestTabPFNClassifier, RandomForestTabPFNRegressor
from run_tabicl import metrics

METHODS = ['Native6', 'PHE6', 'RF_PFN6', 'MixturePFN_MICP6_no_CaPFN']
_PORTFOLIO = {}


def portfolio(task, n_features):
    if task not in _PORTFOLIO:
        typ = TaskType.MULTICLASS if task == 'classification' else TaskType.REGRESSION
        candidates = _get_base_models_from_random_search(task_type=typ, device='cuda', random_state=0,
            categorical_indices=[], random_portfolio_size=32, ignore_pretraining_limits=True)
        chosen = []
        for name, estimator in candidates:
            if type(estimator).__name__ not in ['TabPFNClassifier', 'TabPFNRegressor']:
                continue
            estimator.set_params(model_path=str(CLASS_CKPT if task == 'classification' else REG_CKPT),
                n_estimators=1, device='cuda:0', n_preprocessing_jobs=1, ignore_pretraining_limits=True)
            chosen.append((name, estimator))
            if len(chosen) == 6:
                break
        assert len(chosen) == 6
        _PORTFOLIO[task] = chosen
    # PHE clones supplied models before fitting; return fresh wrappers to avoid
    # sharing fitted preprocessors across benchmark conditions.
    from sklearn.base import clone
    result = []
    for name, estimator in _PORTFOLIO[task]:
        copied = clone(estimator)
        config = deepcopy(copied.inference_config)
        if config and 'PREPROCESS_TRANSFORMS' in config:
            transforms = []
            for original in config['PREPROCESS_TRANSFORMS']:
                transform = dict(original)
                fraction = transform.pop('subsample_features', -1)
                cap = int(np.ceil(n_features * fraction)) if 0 < fraction <= 1 else int(fraction) if fraction > 1 else 10**9
                transform['max_features_per_estimator'] = max(1, cap)
                transforms.append(transform)
            config['PREPROCESS_TRANSFORMS'] = tuple(transforms)
            copied.set_params(inference_config=config)
        result.append((name, copied))
    return result


@contextmanager
def count_calls():
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    counter = {'predict_calls': 0, 'sum_n_estimators': 0}
    originals = []
    for cls, name in [(TabPFNRegressor, 'predict'), (TabPFNClassifier, 'predict_proba')]:
        original = getattr(cls, name)
        def wrap(function):
            @functools.wraps(function)
            def counted(self, *args, **kwargs):
                counter['predict_calls'] += 1
                counter['sum_n_estimators'] += int(self.n_estimators)
                return function(self, *args, **kwargs)
            return counted
        originals.append((cls, name, original))
        setattr(cls, name, wrap(original))
    try:
        yield counter
    finally:
        for cls, name, original in originals:
            setattr(cls, name, original)


def raw_prediction(task, method, X, y, Q, classes):
    # Analytical cases follow the same input-only policy as the Base wrapper.
    degenerate = (np.std(y) < 1e-12 or np.all(np.ptp(X.astype(float), axis=0) == 0))
    if degenerate:
        return fit_predict(model(task, 'cuda:0'), X, y, Q, task, classes), {'analytical_case': True}
    if method == 'Native6':
        est = model(task, 'cuda:0')
        est.n_estimators = 6
        return fit_predict(est, X, y, Q, task, classes), {'native_estimators': 6}
    if method == 'PHE6':
        candidates = portfolio(task, X.shape[1])
        typ = TaskType.REGRESSION if task == 'regression' else TaskType.BINARY if classes == 2 else TaskType.MULTICLASS
        est = AutoPostHocEnsemblePredictor(preset='default', max_time=30, task_type=typ,
            ges_scoring_string='log_loss' if task == 'classification' else 'mse', device='cuda',
            bm_random_state=0, ges_random_state=0, tabpfn_base_model_source=TabPFNBaseModelSource.CUSTOM,
            custom_tabpfn_models=candidates, max_models=6, n_repeats=1, holdout_fraction=.25,
            ges_n_iterations=25, ignore_pretraining_limits=True)
        est.fit(X, y)
        if task == 'regression':
            pred = np.asarray(est.predict(Q), dtype=float).reshape(-1)
        else:
            values = est.predict_proba(Q)
            pred = np.zeros((len(Q), classes))
            pred[:, np.asarray(est._label_encoder.classes_, dtype=int)] = values
            pred = normalize(pred)
        return pred, {'implementation': 'tabpfn-extensions 0.0.4 PHE',
            'portfolio': [(name, estimator.get_params(deep=False)) for name, estimator in candidates],
            'portfolio_sampling': 'official random search seed0, first six non-RF PFNs; pinned v3 checkpoint and one estimator each',
            'v2_to_v3_API_adaptation': 'subsample_features mapped to integer max_features_per_estimator using input dimension; -1 maps to no cap',
            'source_holdout_fraction': .25, 'GES_iterations': 25, 'max_time_seconds': 30,
            'active_members': int(np.count_nonzero(est._ens_model.weights_)),
            'unrestricted_default_PHE': False}
    if method == 'RF_PFN6':
        cls = RandomForestTabPFNClassifier if task == 'classification' else RandomForestTabPFNRegressor
        est = cls(tabpfn=model(task, 'cuda:0'), n_jobs=1, n_estimators=6, random_state=0)
        pred = fit_predict(est, X, y, Q, task, classes)
        return pred, {'implementation': 'tabpfn-extensions 0.0.4 RF-PFN',
            'parameters': est.get_params(deep=False), 'tree_count': 6,
            'all_other_RF_parameters_default': True, 'six_trees_are_not_six_TFM_calls': True}
    assert task == 'classification' and method == 'MixturePFN_MICP6_no_CaPFN'
    pred, details = micp_no_capfn(model(task, 'cuda:0'), X, y, Q, classes, budget=128, count_override=min(6, len(y)))
    return pred, details | {'implementation': 'paper-derived MICP component, budget-adapted',
        'trained_CaPFN': False, 'full_MixturePFN_reproduction': False,
        'six_routed_contexts_not_six_predictions_per_query': True}


def source_clean(task, track, meta):
    if track == 'five':
        setting = json.loads((OUT / f'frozen_{task}_clean.json').read_text())['settings']
        data, _ = canonical(task, meta, 'original', 0, setting, 'clean')
    else:
        setting = json.loads((OUT / f'frozen_{task}_seven.json').read_text())['settings']
        data, _ = canonical(task, meta, 'covariate', 0, setting, 'seven')
    return np.asarray(data['X_source'], dtype=np.float32), data['y_source'].copy()


def safe_split(y, task):
    rng = np.random.default_rng(0)
    if task == 'regression':
        order = rng.permutation(len(y)); n = max(2, int(np.ceil(.2 * len(y))))
        return order[n:], order[:n]
    tr, val = [], []
    for label in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == label))
        n = min(len(ids) - 1, max(1, int(np.ceil(.2 * len(ids)))))
        val.extend(ids[:n]); tr.extend(ids[n:])
    assert len(val) >= 1 and len(tr) >= 2
    return np.asarray(tr), np.asarray(val)


def source_validation_condition(task, track, meta, X, y, Q, truth, family, rho):
    X, y, Q, truth = X.copy(), y.copy(), Q.copy(), truth.copy()
    if not rho:
        return X, y, Q, truth
    frozen = json.loads((OUT / f'frozen_{task}_{track}.json').read_text())['settings']
    if track == 'five' and task == 'regression':
        changed = protocol.manipulate_source_context(task, X, y, family=family, rho=rho / 100., seed=1729 + rho,
            shift_scale=frozen['shift_scale'], label_scale=frozen['label_scale'])
        return changed.X_context, changed.y_context, Q, truth
    if track == 'five':
        from scaled_historical_feature_protocol import apply_feature_scaled_shift
        teacher = ExtraTreesClassifier(n_estimators=128, random_state=0, n_jobs=1).fit(X, y)
        kind = 'aucband' if family == 'band' else family
        Q, _ = apply_feature_scaled_shift(f"publicshift_d{meta['data_id']}_{kind}_rho{rho}_beta500", X, y, Q,
            teacher.predict(Q), 0, feature_multiplier=frozen['feature_multiplier'])
        return X, y, Q, truth
    from seven_shifts import materialize as seven, FAMILIES
    if task == 'classification':
        cut = max(1, len(y) - min(64, max(1, len(y) // 4)))
        clean = (X[:cut], y[:cut], X[cut:], y[cut:], Q, truth)
        data, _ = seven(meta, family, int(round(rho * frozen['amplitude'])), clean)
    else:
        from calibrated_seven_protocol import materialize
        previous = json.loads((OUT.parents[0] / 'internal_tabpfn_regression_v1/frozen_seven.json').read_text())['generator_parameters']
        amplitude = previous['family_amplitude_multipliers'][family]
        cut = min(64, max(1, len(y) // 4))
        clean = (X[cut:2 * cut], y[cut:2 * cut], X[:cut], y[:cut], X, y, Q, truth)
        data, _ = materialize(meta, family, rho, clean, amplitude)
    return tuple(data[k] for k in ['X_source', 'y_source', 'X_target', 'y_target'])


def adapter(task, track, meta):
    folder = OUT / 'source_adapters' / task / track / str(meta['task_id'])
    path = folder / 'policy.json'
    if path.exists():
        return joblib.load(folder / 'extra_trees.joblib'), json.loads(path.read_text())
    X, y = source_clean(task, track, meta)
    tr, val = safe_split(y, task)
    assert not set(tr).intersection(val)
    cls = ExtraTreesClassifier if task == 'classification' else ExtraTreesRegressor
    args = {'n_estimators': 128, 'max_depth': 10, 'min_samples_leaf': 2, 'random_state': 0, 'n_jobs': 1}
    aux = cls(**args).fit(X[tr], y[tr])
    from seven_shifts import FAMILIES
    families = protocol.FAMILIES if track == 'five' else FAMILIES
    cells = [('original' if track == 'five' else 'covariate', 0)] + [(f, r) for f in families for r in [50, 100]]
    est = model(task, 'cuda:0')
    classes = len(np.unique(y)) if task == 'classification' else None
    losses = {str(a): [] for a in [0., .1, .25]}
    records = []
    for family, rho in cells:
        A, b, V, t = source_validation_condition(task, track, meta, X[tr], y[tr], X[val], y[val], family, rho)
        pred = fit_predict(est, A, b, V, task, classes)
        auxpred = aligned(aux, V, task, classes)
        for alpha in [0., .1, .25]:
            combined = (1 - alpha) * pred + alpha * auxpred
            loss = -float(np.log(np.clip(combined[np.arange(len(t)), t.astype(int)], 1e-12, 1)).mean()) if task == 'classification' else float(np.mean((combined - t) ** 2) / max(float(np.var(y[val])), 1e-12))
            losses[str(alpha)].append(loss)
        records.append({'family': family, 'rho': rho, 'queries': len(t)})
    scores = {a: .25 * values[0] + .75 * float(np.mean(values[1:])) for a, values in losses.items()}
    selected = min(scores, key=scores.get)
    aux.fit(X, y)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(aux, folder / 'extra_trees.joblib')
    np.savez_compressed(folder / 'source_split.npz', train_relative_ids=tr, validation_relative_ids=val,
                        pristine_source_X_sha256=protocol.array_sha256(X), pristine_source_y_sha256=protocol.array_sha256(y))
    policy = {'seed': 0, 'alpha': float(selected), 'validation_loss_by_alpha': scores, 'all_validation_losses': losses,
        'max_alpha': .25, 'extra_clean_source_TRAIN_supervision': True,
        'backbone_gradient_finetuning': False, 'adaptation': 'supervised auxiliary predictor and source-validation selected common convex blend',
        'selection_proxy': 'single PFN; one common alpha applied to all ensemble methods, no per-method test ranking selection',
        'query_labels_used_for_fit_selection_or_stopping': False, 'source_rows': len(y), 'validation_rows': len(val),
        'source_split_indices_are_relative_to_canonical_pristine_context': True, 'validation_conditions': records,
        'ExtraTrees_parameters': args}
    atomic_json(path, policy)
    print('PFN_SOURCE_ADAPTER_FROZEN', task, track, meta['dataset'], selected, flush=True)
    return aux, policy


def save_prediction(path, task, data, info, base, raw, adjusted, policy, details, elapsed, calls):
    assert np.isfinite(raw).all() and np.isfinite(adjusted).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, raw_prediction=raw, adapted_prediction=adjusted, original_Base=base, y_target=data['y_target'])
    temp.replace(path)
    def serializable(value):
        return json.loads(json.dumps(value, default=str))
    atomic_json(path.with_suffix('.json'), serializable(info | {'status': 'COMPLETE', 'seed': 0,
        'raw_metrics': metrics(data['y_target'], raw, task, info['scale']),
        'adapted_metrics': metrics(data['y_target'], adjusted, task, info['scale']),
        'input_hashes': {k: protocol.array_sha256(v) for k, v in data.items()},
        'adapter_alpha': policy['alpha'], 'extra_clean_source_TRAIN_supervision': policy['alpha'] > 0,
        'backbone_gradient_finetuning': False, 'benchmark_query_labels_used_for_training_or_selection': False,
        'method_details': details, 'elapsed_seconds': elapsed, 'measured_PFN_calls': calls}))


def run(shard, shards, limit=0, smoke=False, selected_task='both'):
    torch.set_num_threads(1); logging.disable(logging.INFO); warnings.filterwarnings('ignore')
    work = [(task, track, meta) for task in ['classification', 'regression'] if selected_task in ['both', task] for track in ['five', 'seven'] for meta in tasks(task, track)]
    work = work[shard::shards]
    if limit:
        work = work[:limit]
    for task, track, meta in work:
        aux, policy = adapter(task, track, meta)
        grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
        if smoke:
            grid = grid[:1]
        for family, rho in grid:
            data, info, base = frozen_input(task, track, meta, family, rho)
            X, y, Q = (data[k] for k in ['X_source', 'y_source', 'X_target'])
            classes = info.get('n_classes')
            auxiliary = aligned(aux, Q, task, classes)
            for method in METHODS:
                if task == 'regression' and method.startswith('Mixture'):
                    continue
                path = OUT / 'official_predictions' / task / track / str(meta['task_id']) / f'{family}_{rho}__{method}.npz'
                if path.exists() and path.with_suffix('.json').exists():
                    continue
                tick = time.perf_counter()
                with count_calls() as counter:
                    raw, details = raw_prediction(task, method, X, y, Q, classes)
                adjusted = (1 - policy['alpha']) * raw + policy['alpha'] * auxiliary
                save_prediction(path, task, data, info | {'method': method}, base, raw, adjusted, policy, details,
                                time.perf_counter() - tick, counter)
                print('PFN_OFFICIAL_METHOD_DONE', task, track, meta['dataset'], family, rho, method, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--task', choices=['both', 'classification', 'regression'], default='both')
    args = parser.parse_args()
    run(args.shard, args.shards, args.limit, args.smoke, args.task)
