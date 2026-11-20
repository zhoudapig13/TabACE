"""Explicit clean-reference adaptation of reconstructed Shared contexts.

The original source-corruption benchmark and Base are immutable. The adapted
context contains 192 observed source rows and 64 additional clean labeled rows.
The original five constructor rules and frozen dataset-OOF gate are retained.
Base+sameRef is reported separately to expose the extra-supervision effect.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

import clean_reference_manifest_v268 as reference
import unseen_shared_robustness_v117 as source
from prepare_data import ROOT, RUN, atomic_json, protocol
from run_tabicl import Predictor, metrics

OUT = RUN/'results/shared_clean_reference_contexts_v269'


def source_rows(y, task):
    rng = np.random.default_rng(0)
    if task == 'regression':
        return rng.choice(len(y), 192, replace=False)
    mandatory = np.asarray([rng.choice(np.flatnonzero(y == label)) for label in np.unique(y)])
    assert len(mandatory) <= 192
    rest = np.setdiff1d(np.arange(len(y)), mandatory)
    rows = np.r_[mandatory, rng.choice(rest, 192-len(mandatory), replace=False)]
    rng.shuffle(rows)
    return rows


def freeze():
    assert json.loads((reference.OUT/'audit.json').read_text())['status'] == 'PASS'
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'EXPLICIT_EXTRA_CLEAN_REFERENCE_SUPERVISION_ADAPTATION',
        'seed': 0, 'context_cap': 256, 'observed_source_rows_retained': 192,
        'extra_clean_labeled_reference_rows': 64,
        'reference_disjoint_from_source_and_evaluation': True,
        'five_shared_constructor_rules': 'unchanged reconstructed v28',
        'fusion': 'unchanged frozen v77 dataset-OOF policy from v117 snapshot',
        'new_family_or_severity_specific_weights': False,
        'TFM_weights_updated': False, 'new_local_head_fitting': False,
        'Base': 'cached original Native1 on corrupted source256',
        'Base_plus_same_reference_control': 'Native1 on the same source192 + reference64 context as Shared adaptation',
        'compute_matched_to_Base': False, 'additional_TFM_calls_per_condition_maximum': 6,
        'original_Shared_unseen_generalization_claimed': False,
        'previously_scored_development_cohort': True,
        'new_reference_labels_are_additional_supervision': True,
        'final_query_labels_opened_for_prediction_or_selection': False,
        'acceptance': {'each_family_accuracy_gain_at_least': .05,
                       'each_family_r2_gain_at_least': .05},
        'expected_conditions': 1327,
        'reference_manifest_sha256': source.sha(reference.OUT/'reference_manifest.json'),
        'script_sha256': source.sha(__file__),
        'constructors_sha256': source.sha(source.constructors.__file__),
        'fusion_sha256': source.sha(source.fusion.__file__), 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    plan = json.loads((OUT/'plan.json').read_text())
    assert source.sha(__file__) == plan['script_sha256']
    refs = {(r['task'], r['task_id']): r for r in json.loads((reference.OUT/'reference_manifest.json').read_text())}
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track']=='perturbation']
    predictors = {}
    for counter, pos in enumerate(np.random.default_rng(269).permutation(len(cases))):
        if counter % shards != shard:
            continue
        c = cases[int(pos)]
        task, tid = c['task'], c['task_id']
        if task not in predictors:
            predictors[task] = Predictor(task, 'cuda:0')
        predictor = predictors[task]
        assert source.sha(c['input_file']) == c['input_sha256']
        ref = refs[(task, tid)]
        assert source.sha(ref['reference_file']) == ref['reference_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0, y0, Q0, numeric, sid, qid = (z[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns', 'source_indices', 'target_indices'))
        with np.load(ref['reference_file']) as z:
            RX, Ry, rid = (z[k] for k in ('X_reference', 'y_reference', 'reference_indices'))
        assert set(map(int, rid)).isdisjoint(map(int, np.r_[sid, qid]))
        rows = source_rows(y0, task)
        K = len(np.unique(y0)) if task=='classification' else None
        policy_file = source.OUT/'policies'/task/f'{c["fold"]}.json'
        saved = json.loads(policy_file.read_text())
        assert c['data_id'] not in saved['training_data_ids']
        for family, rho in [('clean', 0)] + [(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'predictions'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            if rho:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y, Q = X0, y0, Q0
            old = source.OUT/'predictions/perturbation'/task/str(tid)/dest.name
            om = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == om['output_sha256']
            assert protocol.array_sha256(X) == om['observed_source_X_sha256']
            assert protocol.array_sha256(y) == om['observed_source_y_sha256']
            assert protocol.array_sha256(Q) == om['observed_target_X_sha256']
            with np.load(old) as z:
                base, original_shared = z['Base'], z['Shared']
            A, b = np.concatenate([X[rows], RX]), np.concatenate([y[rows], Ry])
            assert len(b) == 256
            cache = {}
            def infer(V, labels, T):
                key = tuple(protocol.array_sha256(v) for v in (V, labels, T))
                if key not in cache:
                    for estimator in predictor.models.values():
                        root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    cache[key] = predictor(V, labels, T, members=1, random_state=0, n_classes=K)[0]
                return cache[key]
            control = infer(A, b, Q)
            contexts = source.constructors.shared_contexts(A, b, Q, 256, task, control)
            assert len(contexts) == 5 and max(len(labels) for _, _, labels, _ in contexts) <= 256
            members = np.stack([infer(V, labels, T) for _, V, labels, T in contexts])
            mu, sigma = float(b.mean()), max(float(b.std()), 1e-12)
            shared = source.fusion.predict({'task': task, 'base': control, 'bank': members,
                'mu': mu, 'sigma': sigma}, saved['Shared'])
            assert np.isfinite(shared).all()
            source.npwrite(dest, Base=base, OriginalShared=original_shared,
                Base_plus_same_reference=control, Shared_extra_reference=shared,
                members=members, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                 'original_input_sha256': c['input_sha256'], 'original_prediction_sha256': om['output_sha256'],
                 'reference_sha256': ref['reference_sha256'], 'extra_labeled_reference_rows': 64,
                 'retained_source_row_positions': rows.tolist(), 'context_rows': 256,
                 'context_X_sha256': protocol.array_sha256(A), 'context_y_sha256': protocol.array_sha256(b),
                 'roles': [name for name, _, _, _ in contexts],
                 'actual_TFM_calls': len(cache), 'policy_sha256': source.sha(policy_file),
                 'query_labels_opened_for_inference': False})
        print('EXTRA_REFERENCE_CASE_COMPLETE', task, tid, flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        log = OUT/f'worker_{i}.log'
        with log.open('a') as f:
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i), '--shards', str(shards)],
                               env=env, stdout=f, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(log)}
    results = []
    atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': [], 'shards': shards})
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            results.append(future.result())
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': results, 'shards': shards})
    count = len(list((OUT/'predictions').glob('*/*/*.json')))
    success = all(r['returncode']==0 for r in results) and count==1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': results, 'conditions': count})
    assert success


def report():
    assert json.loads((OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    cells = []
    for c in json.loads((source.OUT/'case_manifest.json').read_text()):
        if c['track'] != 'perturbation':
            continue
        assert source.sha(c['truth_file']) == c['truth_sha256']
        with np.load(c['truth_file']) as z:
            truth = z['y_target']
        for family in c['families']:
            for rho in source.novel.STRENGTHS:
                path = OUT/'predictions'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                assert source.sha(path) == meta['output_sha256']
                with np.load(path) as z:
                    for method in ('Base', 'OriginalShared', 'Base_plus_same_reference', 'Shared_extra_reference'):
                        values = metrics(c['task'], truth, z[method], max(float(np.std(truth)), 1e-12))
                        cells.append({'task': c['task'], 'data_id': c['data_id'], 'family': family,
                                      'rho': rho, 'method': method, **values})
    df = pd.DataFrame(cells)
    df.to_csv(OUT/'condition_metrics.csv', index=False)
    keys = ['task', 'family', 'method']
    cols = ['accuracy', 'auc', 'r2', 'nrmse']
    means = df.groupby(keys+['data_id'])[cols].mean().reset_index()
    table = means.groupby(keys)[cols].mean().reset_index()
    counts = means.groupby(keys).size().rename('datasets').reset_index()
    table = table.merge(counts, on=keys)
    gains = []
    for (task, family), group in table.groupby(['task', 'family']):
        metric = 'accuracy' if task=='classification' else 'r2'
        b = float(group.loc[group.method=='Base', metric].iloc[0])
        s = float(group.loc[group.method=='Shared_extra_reference', metric].iloc[0])
        gains.append({'task': task, 'family': family, 'metric': metric,
                      'gain': s-b, 'required_gain': .05, 'passed': s-b >= .05})
    table.to_csv(OUT/'family_averages.csv', index=False)
    atomic_json(OUT/'numerical_acceptance.json', {'family_results': gains,
        'all_six_pass': all(g['passed'] for g in gains), 'additional_clean_labeled_rows_per_dataset': 64,
        'not_original_frozen_Shared': True})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    print(table.to_string(index=False), flush=True)
    print(json.dumps(gains, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker', 'report'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    args = p.parse_args()
    if args.phase == 'launch':
        launch(args.shards)
    elif args.phase == 'worker':
        worker(args.shard, args.shards)
    else:
        report()
