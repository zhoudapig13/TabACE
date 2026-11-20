"""Explicitly higher-budget Shared inference on the frozen new perturbations.

Base-1 is preserved. Every Shared forward context uses official Native-8.
Report Native-8 and equal-bank Mean6 controls as well as both frozen fusion
and a predeclared .25 Base-anchored update. No backbone-weight finetuning.
Any gain over Base-1 may come from the extra native ensemble budget.
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

import unseen_shared_robustness_v117 as source
from prepare_data import ROOT, RUN, atomic_json, protocol
from run_tabicl import Predictor, normalize, metrics

OUT = RUN / 'results/shared_native8_adaptation_v120'
METHODS = ('Base1', 'Native8', 'Mean6_Native8', 'Shared_Native8', 'Shared_Native8_conservative')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [c for c in json.loads((source.OUT / 'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    plan = {
        'status': 'HIGHER_INFERENCE_BUDGET_ADAPTATION_NOT_ORIGINAL_SHARED',
        'seed': 0, 'context_cap': 256, 'datasets': {'classification': 72, 'regression': 35},
        'backbone_weights_and_context_constructors': 'unchanged v117',
        'estimators_per_context': 8, 'Shared_contexts_including_native_anchor': 6,
        'maximum_native_estimator_passes_per_condition': 48,
        'original_Base_native_estimator_passes': 1,
        'controls': ['original paired Base1', 'Native8 canonical context', 'Mean6 on identical Native8 bank'],
        'variants': ['original frozen fusion on Native8 outputs', 'Native8 + .25*(SharedNative8 - Native8)'],
        'conservative_update_strength_fixed_before_this_inference': .25,
        'new_family_calibration_or_new_weights': False,
        'original_Base_and_operators_not_modified': True,
        'increased_compute_budget_must_be_disclosed': True,
        'extra_target_labels': 0, 'new_context_constructors': False,
        'strengths': source.novel.STRENGTHS,
        'Shared_does_not_receive_family_or_severity': True,
        'all_attempted_variants_and_control_metrics_preserved': True,
        'all_query_cohorts_previously_evaluated_development': True,
        'independent_confirmation': False,
        'primary_tests': 'two Shared variants times three perturbations times two tasks;Holm12 within this run',
        'cumulative_exploratory_tests': 'six frozen perturbation endpoints +12 v119 endpoints +12 current endpoints;Holm30',
        'gain_interpretation': 'Base1 comparison alone cannot attribute improvement to Shared fusion rather than Native8',
        'expected_conditions': sum(1+4*len(c['families']) for c in cases),
        'script_sha256': source.sha(__file__),
        'source_plan_sha256': source.sha(source.OUT / 'plan.json'),
        'source_generator_sha256': source.sha(source.novel.__file__), 'Markdown_written': False,
    }
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == json.loads(json.dumps(plan))
    else:
        atomic_json(p, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT / 'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    order = np.random.default_rng(120).permutation(len(cases))
    models = {}
    for index, pos in enumerate(order):
        if index % shards != shard:
            continue
        c = cases[int(pos)]
        task, tid = c['task'], c['task_id']
        if task not in models:
            models[task] = Predictor(task, 'cuda:0')
        predictor = models[task]
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0, y0, Q0, numeric = z['X_source'], z['y_source'], z['X_target'], z['numeric_columns']
        K = len(np.unique(y0)) if task == 'classification' else None
        policy_file = source.OUT / 'policies' / task / f'{c["fold"]}.json'
        saved = json.loads(policy_file.read_text())
        assert c['data_id'] not in saved['training_data_ids']
        for family, rho in [('clean', 0)] + [(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT / 'predictions' / task / str(tid) / f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                continue
            oldfile = source.OUT / 'predictions/perturbation' / task / str(tid) / dest.name
            oldmeta = json.loads(oldfile.with_suffix('.json').read_text())
            if family == 'clean':
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
            else:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            for value, key in ((X, 'observed_source_X_sha256'), (y, 'observed_source_y_sha256'), (Q, 'observed_target_X_sha256')):
                assert protocol.array_sha256(value) == oldmeta[key]
            cache = {}
            calls = 0
            def infer(A, b, T):
                nonlocal calls
                key = tuple(protocol.array_sha256(np.asarray(v)) for v in (A, b, T))
                if key in cache:
                    return cache[key]
                for model in predictor.models.values():
                    root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
                    if root is not None and hasattr(root, 'root'):
                        root.root = None
                torch.cuda.empty_cache()
                p = predictor(A, b, T, members=8, random_state=0, n_classes=K)[0]
                assert np.isfinite(p).all()
                cache[key] = p
                calls += 1
                return p
            anchor = infer(X, y, Q)
            views = source.constructors.shared_contexts(X, y, Q, len(y), task, anchor)
            assert len(views) == 5 and max(len(v) for _, _, v, _ in views) <= 256
            members = np.stack([infer(A, b, T) for _, A, b, T in views])
            cell = {'task': task, 'base': anchor, 'bank': members,
                'mu': float(y.mean()), 'sigma': max(float(y.std()), 1e-12)}
            shared = source.fusion.predict(cell, saved['Shared'])
            conservative = anchor+.25*(shared-anchor)
            mean = np.concatenate([anchor[None], members], axis=0).mean(0)
            if task == 'classification':
                conservative, mean = normalize(conservative), normalize(mean)
            with np.load(oldfile) as z:
                base1 = z['Base']
            source.npwrite(dest, Base1=base1, Native8=anchor, Mean6_Native8=mean,
                Shared_Native8=shared, Shared_Native8_conservative=conservative,
                members=members, source_mean=cell['mu'], source_scale=cell['sigma'])
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'dataset', 'fold')} |
                {'family': family, 'rho': rho, 'original_bank': str(oldfile),
                'original_bank_sha256': source.sha(oldfile), 'output_sha256': source.sha(dest),
                'source_input_sha256': c['input_sha256'], 'source_truth_file': c['truth_file'],
                'source_truth_sha256': c['truth_sha256'], 'policy_file': str(policy_file),
                'policy_sha256': source.sha(policy_file), 'canonical_context_rows': len(y),
                'branch_rows': [len(v) for _, _, v, _ in views],
                'actual_TFM_context_calls': calls, 'native_estimators_each_call': 8,
                'extra_target_labels': 0, 'target_labels_loaded_for_inference': False})
        print('NATIVE8_SHARED_TASK', task, tid, flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        path = OUT / f'worker_{i}.log'
        with path.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i),
                '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(path)}
    records = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            records.append(future.result())
            print('NATIVE8_SHARED_WORKER', records[-1], flush=True)
            atomic_json(OUT / 'execution.json', {'status': 'RUNNING', 'workers': records})
    success = all(r['returncode'] == 0 for r in records)
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': records})
    assert success


def report():
    plan = json.loads((OUT / 'plan.json').read_text())
    assert source.sha(__file__) == plan['script_sha256']
    assert json.loads((OUT / 'execution.json').read_text())['status'] == 'COMPLETE'
    files = list((OUT / 'predictions').glob('*/*/*.npz'))
    assert len(files) == plan['expected_conditions']
    rows = []
    for path in sorted(files):
        m = json.loads(path.with_suffix('.json').read_text())
        assert source.sha(path) == m['output_sha256']
        assert source.sha(m['original_bank']) == m['original_bank_sha256']
        assert source.sha(m['source_truth_file']) == m['source_truth_sha256']
        assert source.sha(m['policy_file']) == m['policy_sha256']
        assert not m['target_labels_loaded_for_inference']
        with np.load(m['source_truth_file']) as z:
            y = z['y_target']
        with np.load(path) as z:
            predictions = {method: z[method] for method in METHODS}
        with np.load(m['original_bank']) as z:
            assert np.array_equal(predictions['Base1'], z['Base'])
        for method, p in predictions.items():
            rows.append({k: m[k] for k in ('task', 'task_id', 'data_id', 'dataset', 'family', 'rho')}
                | {'method': method, **metrics(y, p, m['task'], max(float(y.std()), 1e-12))})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'detail.csv', index=False)
    tables, tests = [], []
    for (task, family), group in detail[detail.rho > 0].groupby(['task', 'family']):
        cols = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        values = group.groupby(['data_id', 'method'])[cols].mean()
        for method in METHODS:
            v = values.xs(method, level='method')
            tables.append({'task': task, 'family': family, 'method': method, 'datasets': len(v),
                **{k: float(v[k].mean()) for k in cols}})
        primary = 'auc' if task == 'classification' else 'r2'
        for method in ('Shared_Native8', 'Shared_Native8_conservative'):
            delta = values.xs(method, level='method')[primary]-values.xs('Base1', level='method')[primary]
            tests.append({'task': task, 'family': family, 'method': method, 'metric': primary,
                **source.paired_statistics(delta)})
    table, stats = pd.DataFrame(tables), pd.DataFrame(tests)
    for n, frame, column in [(12, stats, 'p_holm_12')]:
        last = 0.
        for rank, i in enumerate(sorted(frame.index, key=lambda j: frame.loc[j, 'p_two_sided'])):
            last = max(last, min(1., (n-rank)*frame.loc[i, 'p_two_sided']))
            frame.loc[i, column] = last
    previous = pd.read_csv(RUN / 'results/unseen_shared_adaptation_v119/paired_statistics.csv')
    previous['version'] = 'head_v119_'+previous['mode']
    frozen = pd.read_csv(source.OUT / 'paired_statistics.csv')
    frozen = frozen[frozen.track == 'perturbation'].copy()
    frozen['version'] = 'frozen_v117'
    current = stats.copy()
    current['version'] = current.method
    combined = pd.concat([frozen, previous, current], ignore_index=True)
    assert len(combined) == 30
    last = 0.
    for rank, i in enumerate(sorted(combined.index, key=lambda j: combined.loc[j, 'p_two_sided'])):
        last = max(last, min(1., (30-rank)*combined.loc[i, 'p_two_sided']))
        combined.loc[i, 'p_holm_cumulative30'] = last
    table.to_csv(OUT / 'family_averages.csv', index=False)
    stats.to_csv(OUT / 'paired_statistics.csv', index=False)
    combined.to_csv(OUT / 'cumulative_exploratory_statistics.csv', index=False)
    success = {}
    for method, g in combined[combined.version.isin(['Shared_Native8', 'Shared_Native8_conservative'])].groupby('version'):
        success[method] = bool(((g.gain > 0) & (g.ci_low > 0) & (g.p_holm_cumulative30 < .05)).all())
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'audit.json', {'status': 'PASS_HIGHER_BUDGET_INFERENCE_ADAPTATION',
        'conditions': len(files), 'Base1_predictions_bitwise_unchanged': True,
        'no_new_mechanism_calibration_or_TFM_weight_changes': True,
        'inference_native_estimator_budget_is_48_instead_of_6': True,
        'Native8_and_equal_bank_Mean6_controls_saved': True,
        'all_six_endpoints_cumulative_significant': success,
        'gain_cannot_be_attributed_to_fusion_from_Base1_comparison_alone': True,
        'previously_evaluated_development_cohort': True, 'Markdown_bytes': 0})
    atomic_json(OUT / 'experiment_log.json', {
        'Purpose': 'Test whether stronger native backbone inference improves the unchanged Shared bank',
        'Setting': plan, 'Results': json.loads(table.to_json(orient='records', double_precision=15)),
        'Analysis': {'all_six_endpoints_significant': success,
            'extra_native_ensemble_compute_is_a_confound_for_fusion_attribution': True},
        'Next Steps': 'Inspect controls and keep every null/negative outcome. Independently confirm any selected adaptation.',
    })
    print(table.to_string(index=False), flush=True)
    print(stats.to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker', 'report'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    if a.phase == 'worker':
        worker(a.shard, a.shards)
    elif a.phase == 'launch':
        launch(a.shards)
    else:
        report()
