"""Source-label-only diagnostic; no new task-query hypothesis is tested.

Includes one explicitly outcome-selected failure case (DID44987) and fixed
controls. Not an independent confirmation and not a passing-result search.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
import torch

import shared_feature_forest_banks_v227 as bank
import source_proxy_repair_quality_v231 as proxy
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

source = bank.source
OUT = RUN/'results/source_proxy_repair_pilot_v233'


def cases():
    manifest = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    classification = sorted([c for c in manifest if c['task'] == 'classification'], key=lambda c: (c['data_id'], c['task_id']))[:3]
    regression = sorted([c for c in manifest if c['task'] == 'regression' and c['data_id'] != 44987], key=lambda c: (c['data_id'], c['task_id']))[:2]
    failure = [c for c in manifest if c['task'] == 'regression' and c['data_id'] == 44987]
    assert len(failure) == 1
    return classification+regression+failure


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = cases()
    plan = {'status': 'SOURCE_ONLY_DIAGNOSTIC_CORRECTED_RHO_UNITS_NOT_NEW_QUERY_HYPOTHESIS', 'seed': 0,
        'task_ids': [c['task_id'] for c in selected], 'data_ids': [c['data_id'] for c in selected],
        'outcome_selected_failure_case': {'task': 'regression', 'data_id': 44987},
        'control_selection': 'first3 classification and first2 other regression IDs in ascending order',
        'conditions_each_case': 'clean plus all pre-eligible families at severity0.5 and1.0', 'generator_internal_rho_units': [50, 100],
        'expected_conditions': sum(1+2*len(c['families']) for c in selected),
        'source_validation_input': 'same full-repair source-held-out features for all three contexts',
        'source_training_contexts': ['observed', 'mild_repair', 'full_repair'],
        'source_labels_used': 'same existing source labels; held-out rows excluded from every callback fit',
        'task_query_labels_opened': False, 'query_task_predictions_produced': False,
        'TFM_calls_each_condition': 'up to9 source-CV calls; no final-query inference',
        'rare_source_classes': 'counts<3 always kept in training, never source-scored',
        'loss': 'source negative log likelihood or source-variance-normalized MSE',
        'not_independent_confirmation': True, 'no_new_primary_hypothesis_tests': True,
        'script_sha256': source.sha(__file__), 'proxy_sha256': source.sha(proxy.__file__),
        'constructor_sha256': source.sha(bank.constructor.__file__),
        'original_plan_sha256': source.sha(source.OUT/'plan.json'), 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def worker(index):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    c = cases()[index]
    task, tid = c['task'], c['task_id']
    assert source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as data:
        assert 'y_target' not in data.files
        X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
    model = bank.constructor.fit_target(Q0, numeric)
    predictor = Predictor(task, 'cuda:0')
    K = len(np.unique(y0)) if task == 'classification' else None
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in (50, 100)]:
        dest = OUT/'features'/task/str(tid)/f'{family}_{rho}.npz'
        if dest.with_suffix('.json').exists():
            assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
            continue
        if rho:
            X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
        else:
            X, y, Q = X0, y0, Q0
        np.testing.assert_array_equal(Q, Q0)
        repaired = [v['X'] for v in bank.constructor.repair_source(X, model)]
        calls = 0
        def infer(A, b, T):
            nonlocal calls
            for estimator in predictor.models.values():
                root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                if root is not None and hasattr(root, 'root'):
                    root.root = None
            torch.cuda.empty_cache()
            calls += 1
            return predictor(A, b, T, members=1, random_state=0, n_classes=K)[0]
        result = proxy.quality(X, y, repaired, task, infer)
        source.npwrite(dest, predictions=result['predictions'], loss=result['loss'], scored=result['scored'])
        average = result['loss'][:, result['scored']].mean(1) if result['scored'].any() else np.zeros(3)
        atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
            {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'original_input_sha256': c['input_sha256'], 'TFM_source_CV_calls': calls,
                'source_partitions': result['partitions'],
                'mandatory_unscored_rows': result['mandatory_unscored_rows'].tolist(),
                'same_proxy_validation_features': True, 'source_labels_sha256': protocol.array_sha256(y),
                'source_context_X_sha256': [protocol.array_sha256(v) for v in [X]+repaired],
                'source_proxy_loss_mean_observed_mild_full': average.tolist(),
                'source_proxy_loss_gain_mild_full': (average[0]-average[1:]).tolist(),
                'task_query_labels_opened': False, 'no_task_query_predictions': True})
        print('SOURCE_PROXY_CASE_CONDITION', task, tid, family, rho, average.tolist(), flush=True)


def launch():
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        log = OUT/f'worker_{i}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--index', str(i)],
                env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'index': i, 'returncode': p.returncode, 'log': str(log)}
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(6)]):
            results.append(future.result())
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': results})
    metas = [json.loads(p.read_text()) for p in (OUT/'features').glob('*/*/*.json')]
    expected = json.loads((OUT/'plan.json').read_text())['expected_conditions']
    success = all(r['returncode'] == 0 for r in results) and len(metas) == expected
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': results, 'conditions': len(metas)})
    assert success
    rows = []
    for m in metas:
        for i, method in enumerate(('observed', 'mild_repair', 'full_repair')):
            losses = m['source_proxy_loss_mean_observed_mild_full']
            rows.append({k: m[k] for k in ('task', 'task_id', 'data_id', 'family', 'rho')} |
                {'method': method, 'source_proxy_loss': losses[i], 'source_proxy_gain_over_observed': losses[0]-losses[i]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'source_proxy_losses.csv', index=False)
    atomic_json(OUT/'experiment_log.json', {
        'Purpose': 'Diagnose whether source-only task validation on a target-like feature proxy can reject harmful repairs',
        'Setting': json.loads((OUT/'plan.json').read_text()),
        'Results': rows, 'Analysis': {'one_failure_case_selected_from_previous_query_diagnostics': True,
            'no_task_query_labels_opened_here': True, 'no_new_primary_hypothesis_tests': True},
        'Next Steps': 'Check transfer on the failure case and controls before committing to all-dataset source-CV extraction.',
    })
    assert (source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--index', type=int, default=0)
    args = p.parse_args()
    launch() if args.phase == 'launch' else worker(args.index)
