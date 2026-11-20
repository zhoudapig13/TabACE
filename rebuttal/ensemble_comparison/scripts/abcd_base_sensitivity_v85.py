"""Source-operator sensitivity on a declared development cohort.

This grid does not alter target examples or labels, context row identities,
nominal severities, or the frozen backbone. It diagnoses unknown amplitude
constants in the missing historical generator. Any score-based alignment on
this cohort is benchmark development, never independent confirmation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, '1')

import numpy as np
import pandas as pd
import torch

import abcd_mechanism_v84 as previous
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor, metrics
import corrected_source_protocol as source
from seven_shifts import registry, regression_clean
from calibrated_seven_protocol import materialize

OUT = RUN / 'results/abcd_base_source_sensitivity_v85'
CLASS_AMPLITUDES = (.25, .5, 1., 2., 3., 6.)
REG_MULTIPLIERS = (.375, .5, .625, .75)


def tasks():
    classes = [json.loads(p.read_text()) for p in sorted((RUN / 'cache/five/classification').glob('*.json'))]
    regressions = [m for m in registry() if m['task'] == 'regression']
    return classes + regressions


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'plan.json'
    if path.exists():
        assert json.loads(path.read_text())['code_sha256'] == previous.sha(__file__)
        return
    atomic_json(path, {'seed': 0, 'context_rows': 256, 'n_estimators': 1,
        'classification_datasets': 72, 'regression_datasets': 33,
        'nominal_severities': [.2, .5, .8, 1.],
        'classification_feature_amplitudes_sd': CLASS_AMPLITUDES,
        'regression_response_amplitude_multipliers': REG_MULTIPLIERS,
        'original_regression_response_drift_sd': 2.,
        'fixed': ['source and target cohorts', 'selected affected rows', 'classification source labels',
                  'regression source features', 'all target features and labels', 'frozen checkpoint'],
        'query_labels_may_be_used_for': 'development metric computation and transparent benchmark alignment only',
        'original_generator_recovered': False, 'independent_test_claim': False,
        'target_ranking_selection': False, 'adapter_training': False, 'Markdown_written': False,
        'code_sha256': previous.sha(__file__)})


def worker(shard, shards):
    torch.set_num_threads(1); torch.manual_seed(0)
    predictors = {}
    rows = []
    for i, meta in enumerate(tasks()):
        if i % shards != shard:
            continue
        task = meta['task']; identifier = str(meta['task_id'])
        if task not in predictors:
            predictors[task] = Predictor(task, 'cuda:0')
        predictor = predictors[task]
        if task == 'classification':
            with np.load(RUN / 'cache/five/classification' / f'{identifier}.npz') as z:
                original = {k: z[k] for k in z.files}
            grid = CLASS_AMPLITUDES
            n_classes = len(np.unique(original['y_source']))
        else:
            clean = regression_clean(meta); grid = REG_MULTIPLIERS; n_classes = None
        for config in grid:
            for rho in (0, 20, 50, 80, 100):
                # Every configuration shares one newly computed clean control.
                label = 'clean' if rho == 0 else f'{config:g}_{rho}'
                out = OUT / 'predictions' / task / identifier / f'{label}.npz'
                if not out.exists():
                    if task == 'classification':
                        changed = source.manipulate_source_context(task, original['X_source'], original['y_source'],
                            family='shortcut', rho=rho / 100., seed=1729 + rho, shift_scale=config)
                        data = original | {'X_source': changed.X_context, 'y_source': changed.y_context}
                        assert np.array_equal(data['y_source'], original['y_source'])
                    else:
                        data, _ = materialize(meta, 'concept', rho, clean, config)
                        assert np.array_equal(data['X_source'], clean[4].astype(np.float32))
                    for model in predictor.models.values():
                        root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    prediction, _ = predictor(data['X_source'], data['y_source'], data['X_target'], n_classes=n_classes)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp, prediction=prediction, y=data['y_target']); tmp.replace(out)
                    atomic_json(out.with_suffix('.json'), {'task': task, 'task_id': meta['task_id'],
                        'config': config if rho else None, 'rho': rho,
                        'X_source_sha256': protocol.array_sha256(data['X_source']),
                        'y_source_sha256': protocol.array_sha256(data['y_source']),
                        'X_target_sha256': protocol.array_sha256(data['X_target']),
                        'y_target_sha256': protocol.array_sha256(data['y_target']),
                        'inference_receives_target_labels': False})
                with np.load(out) as z:
                    truth, prediction = z['y'], z['prediction']
                scale = max(float(truth.std()), 1e-12) if task == 'regression' else 1.
                rows.append({'task': task, 'task_id': meta['task_id'], 'data_id': meta['data_id'],
                    'dataset': meta['dataset'], 'config': config, 'rho': rho, **metrics(truth, prediction, task, scale)})
        print('BASE_GRID_TASK', shard, task, identifier, flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'detail_{shard}.csv', index=False)


def report():
    files = list(OUT.glob('detail_*.csv'))
    if not files:
        return
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    frame.to_csv(OUT / 'detail.csv', index=False)
    keys = ['accuracy', 'auc', 'r2', 'nrmse']
    curve = frame.groupby(['task', 'config', 'rho'])[keys].mean().reset_index()
    curve.to_csv(OUT / 'strength_curves.csv', index=False)
    means = frame[frame.rho > 0].groupby(['task', 'config', 'data_id'])[keys].mean().groupby(['task', 'config']).mean()
    means.to_csv(OUT / 'means.csv')
    print(means.to_string(), flush=True)


def launch(shards):
    freeze()
    def execute(i):
        log = OUT / f'worker_{i}.log'
        with log.open('a') as handle:
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(i)
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i), '--shards', str(shards)],
                               env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(log)}
    results = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            r = future.result(); results.append(r); print(r, flush=True)
            report()
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE' if all(r['returncode'] == 0 for r in results) else 'FAILED', 'workers': results})
    if any(r['returncode'] for r in results):
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
