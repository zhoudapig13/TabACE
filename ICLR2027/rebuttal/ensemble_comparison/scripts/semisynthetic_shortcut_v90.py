"""Explicitly semisynthetic, label-conditional Shortcut development tasks.

Real OpenML labels and row cohorts are retained. A source-fitted projection of
the real covariates supplies a stable feature block; source and target obtain
a synthetic noisy ordinal class feature. Target labels are used to GENERATE
that synthetic benchmark feature, never passed to predictors/constructors.
This is a controlled P(X|Y) construction, not original-data reproduction.
Only the synthetic source coordinate is corrupted at positive severities.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr, ndtri
from sklearn.decomposition import TruncatedSVD

import abcd_mechanism_v84 as origin
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor, metrics

OUT = RUN / 'results/semisynthetic_shortcut_development_v90'
CONFIGS = [dict(name=f'realnoise{beta:g}_signal{signal:g}_amplitude{amplitude:g}',
                real_noise=beta, signal=signal, amplitude=amplitude)
           for beta, signal, amplitude in itertools.product((.5, 1., 2.), (.65, .85), (1., 3.))]


def cohort():
    return [json.loads(p.read_text()) for p in sorted((RUN / 'cache/five/classification').glob('*.json'))]


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text())['code_sha256'] == origin.sha(__file__)
        return
    atomic_json(p, {'status': 'FROZEN_BEFORE_BASE_GRID', 'seed': 0, 'datasets': 72,
        'source_rows': 256, 'query_rows': 128, 'n_estimators': 1,
        'nominal_severities': [.2, .5, .8, 1.], 'configuration_grid': CONFIGS,
        'real_feature_block': 'source-standardized clipped real features, source-only SVD at most16 dimensions, additive source-scaled Gaussian nuisance',
        'synthetic_feature': 'ordinal class center plus independent unit Gaussian noise; class-count-adjusted adjacent center separation',
        'synthetic_feature_generation_uses_source_and_target_labels': True,
        'target_label_use_explanation': 'target labels define the label-conditional synthetic feature distribution; never supplied to evaluated methods; this is not a label-free original-data perturbation',
        'source_labels_changed': False, 'target_labels_changed': False,
        'source_corruption': 'only appended synthetic coordinate; fixed nested source rows ranked by distance of label from its median; overwrite opposite-side values',
        'corrupted_coordinate_jitter_sd_fraction': .05,
        'jitter_reason': 'preserve a continuous observed coordinate rather than collapse all source rows to two categorical values',
        'stable_features_corrupted_by_the_shortcut_operator': False,
        'matched_constructor_does_not_receive_synthetic_column_identity': True,
        'same_noise_streams_for_all_configurations': True,
        'selection': 'minimize maximum absolute discrepancy of four aggregate Base metrics (Clean Acc/AUC and Shortcut Acc/AUC) to paper reference; retain every candidate',
        'reference': {'clean_accuracy': .8037109375, 'clean_auc': .9007194650323471,
                      'shortcut_accuracy': .4726833767361111, 'shortcut_auc': .6521315630491129},
        'development_metric_tolerance': .05, 'calibration_uses_this_development_cohort_metrics': True,
        'independent_confirmation': False, 'original_paper_replication': False,
        'statistical_significance_or_ABCD_ranking_used_for_Base_selection': False,
        'Markdown_written': False, 'code_sha256': origin.sha(__file__)})


def latent(meta):
    task_id = str(meta['task_id'])
    source = RUN / 'cache/five/classification' / f'{task_id}.npz'
    with np.load(source) as z:
        X, y, Q, t = (z[k] for k in ('X_source', 'y_source', 'X_target', 'y_target'))
    center = X.mean(axis=0); scale = np.maximum(X.std(axis=0), 1e-6)
    S, T = np.clip((X - center) / scale, -5, 5), np.clip((Q - center) / scale, -5, 5)
    if S.shape[1] > 16:
        projection = TruncatedSVD(n_components=16, random_state=0, n_iter=5).fit(S)
        S, T = projection.transform(S), projection.transform(T)
    center2 = S.mean(axis=0); scale2 = np.maximum(S.std(axis=0), 1e-6)
    S, T = (S - center2) / scale2, (T - center2) / scale2
    rng = np.random.default_rng(100000 + int(meta['task_id']))
    return {'source_real': S, 'target_real': T, 'source_noise': rng.normal(size=S.shape),
        'target_noise': rng.normal(size=T.shape), 'source_z_noise': rng.normal(size=len(y)),
        'target_z_noise': rng.normal(size=len(t)), 'source_corruption_noise': rng.normal(size=len(y)),
        'y_source': y, 'y_target': t}


def materialize(meta, config, rho, data):
    y, t = data['y_source'], data['y_target']; classes = len(np.unique(y))
    desired = ndtr(config['signal'])
    # Mean nearest-center accuracy of K equally spaced Gaussian classes is
    # (2-2/K)Phi(gap/2) - (1-2/K). This is a generator setting, not a claim
    # about achieved TabICL accuracy, which is measured independently.
    probability = (desired + 1 - 2 / classes) / (2 - 2 / classes)
    gap = 2 * ndtri(np.clip(probability, 1e-6, 1 - 1e-6))
    source_z = gap * (y - (classes - 1) / 2.) + data['source_z_noise']
    target_z = gap * (t - (classes - 1) / 2.) + data['target_z_noise']
    X = np.column_stack([data['source_real'] + config['real_noise'] * data['source_noise'], source_z])
    Q = np.column_stack([data['target_real'] + config['real_noise'] * data['target_noise'], target_z])
    count = int(np.ceil(rho / 100. * len(y)))
    affected = np.sort(np.argsort(np.abs(y - np.median(y)), kind='stable')[-count:]) if count else np.empty(0, dtype=int)
    if count:
        means = np.array([source_z[y == label].mean() for label in np.unique(y)])
        distances = np.abs(means[:, None] - means[None]); np.fill_diagonal(distances, -np.inf)
        farthest = np.argmax(distances, axis=1)
        direction = np.sign(means[farthest] - means)
        X[affected, -1] = source_z.mean() + direction[y[affected].astype(int)] * config['amplitude'] * source_z.std()
        X[affected, -1] += .05 * source_z.std() * data['source_corruption_noise'][affected]
    return {'X_source': X.astype(np.float32), 'y_source': y.copy(),
        'X_target': Q.astype(np.float32), 'y_target': t.copy(), 'affected_rows': affected}


def worker(shard, shards):
    torch.set_num_threads(1); torch.manual_seed(0)
    predictor = Predictor('classification', 'cuda:0')
    rows = []
    for i, meta in enumerate(cohort()):
        if i % shards != shard:
            continue
        data = latent(meta)
        for config in CONFIGS:
            for rho in (0, 20, 50, 80, 100):
                out = OUT / 'predictions' / str(meta['task_id']) / config['name'] / f'{rho}.npz'
                if not out.with_suffix('.json').exists():
                    case = materialize(meta, config, rho, data)
                    for model in predictor.models.values():
                        root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    prediction, _ = predictor(case['X_source'], case['y_source'], case['X_target'], n_classes=len(np.unique(data['y_source'])))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    temp = out.with_suffix('.tmp.npz'); np.savez_compressed(temp, prediction=prediction, y=case['y_target']); temp.replace(out)
                    atomic_json(out.with_suffix('.json'), {'status': 'COMPLETE', 'task_id': meta['task_id'],
                        'configuration': config, 'rho': rho, 'features': case['X_source'].shape[1],
                        'source_X_sha256': protocol.array_sha256(case['X_source']),
                        'source_y_sha256': protocol.array_sha256(case['y_source']),
                        'target_X_sha256': protocol.array_sha256(case['X_target']),
                        'target_y_sha256': protocol.array_sha256(case['y_target']),
                        'target_labels_used_for_synthetic_feature_generation': True,
                        'target_labels_received_by_predictor': False})
                with np.load(out) as z:
                    values = metrics(z['y'], z['prediction'], 'classification', 1.)
                rows.append({'task_id': meta['task_id'], 'data_id': meta['data_id'], 'dataset': meta['dataset'],
                    'configuration': config['name'], 'rho': rho, **values})
        print('SEMISYNTH_BASE_TASK', shard, meta['task_id'], flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'detail_{shard}.csv', index=False)


def report():
    paths = sorted(OUT.glob('detail_*.csv'))
    if not paths:
        return
    data = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    data.to_csv(OUT / 'detail.csv', index=False)
    rows = []
    ref = json.loads((OUT / 'plan.json').read_text())['reference']
    for name, group in data.groupby('configuration'):
        c = group[group.rho == 0][['accuracy', 'auc']].mean()
        s = group[group.rho > 0].groupby('data_id')[['accuracy', 'auc']].mean().mean()
        values = {'clean_accuracy': float(c.accuracy), 'clean_auc': float(c.auc),
                  'shortcut_accuracy': float(s.accuracy), 'shortcut_auc': float(s.auc)}
        errors = {key: abs(value - ref[key]) for key, value in values.items()}
        rows.append({'configuration': name, 'datasets': group.data_id.nunique(), **values,
                     'max_abs_error': max(errors.values()), 'sum_squared_error': sum(v*v for v in errors.values())})
    frame = pd.DataFrame(rows).sort_values(['max_abs_error', 'sum_squared_error'])
    frame.to_csv(OUT / 'configuration_summary.csv', index=False)
    if data.data_id.nunique() == 72 and len(data) == 4320:
        selected = frame.iloc[0].to_dict()
        config = next(c for c in CONFIGS if c['name'] == selected['configuration'])
        atomic_json(OUT / 'Base_selection.json', {'status': 'FROZEN_BEFORE_ABCD' if selected['max_abs_error'] <= .05 else 'ALIGNMENT_NOT_YET_MET',
            'configuration': config, 'development_result': selected, 'tolerance': .05,
            'selection_uses_ABCD_outcomes': False, 'original_paper_replication': False,
            'independent_confirmation': False, 'all_candidates_retained': True})
    print(frame.to_string(index=False), flush=True)


def launch(shards):
    freeze()
    def execute(i):
        log = OUT / f'worker_{i}.log'; env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(i)
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i), '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'job': i, 'returncode': p.returncode, 'log': str(log)}
    records = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for f in as_completed([pool.submit(execute, i) for i in range(shards)]):
            r = f.result(); records.append(r); print(r, flush=True)
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE' if all(r['returncode'] == 0 for r in records) else 'FAILED', 'workers': records})
    report()
    if any(r['returncode'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['launch', 'worker', 'report'])
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=8)
    args = p.parse_args()
    if args.phase == 'worker':
        worker(args.shard, args.shards)
    elif args.phase == 'report':
        report()
    else:
        launch(args.shards)
