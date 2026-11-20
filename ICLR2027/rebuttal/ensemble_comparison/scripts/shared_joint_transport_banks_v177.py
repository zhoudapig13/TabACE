"""Two mild joint-transport contexts, replacing two generic alternatives.

Keep all observed source rows and labels. The Base pseudo outcome is not a
true target label. This is a full constructor extension, not frozen Shared.
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
import torch
import shared_native8_generic_banks_v171 as parent
import joint_transport_context_v176 as constructor
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

source = parent.source
OUT = RUN/'results/shared_joint_transport_v177'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'JOINT_TRANSPORT_CONSTRUCTOR_ADAPTATION_NOT_ORIGINAL_SHARED',
        'seed': 0, 'context_cap': 256, 'alternative_contexts': 5,
        'alternative_estimators_each': 1, 'replacement_slots_both_tasks': [3, 4],
        'first_three_generic_alternatives_and_Base_unchanged': True,
        'source_rows_preserved_by_new_contexts': True, 'source_labels_preserved': True,
        'Base_pseudo_outcomes_used': True, 'true_query_outcomes_used': False,
        'unlabeled_query_features_used_transductively': True,
        'no_clean_source_counterpart_or_new_labeled_rows': True,
        'constructor_receives_corruption_or_strength': False,
        'inference_passes_original_canonical': 6,
        'inference_passes_if_Native8_canonical_selected_later': 14,
        'Native8_case_requires_Native1_pseudo_outcomes_in_addition_to_Native8_gate': True,
        'source_CV_training_TFM_calls_required_by_this_bank': 0,
        'transport_mixing_strengths': [.25, .5], 'entropy_regularization': .15,
        'joint_cost': 'median-scaled feature cost +0.5 median-scaled source-label/Base-pseudo-label cost, clipped20',
        'source_reference': 'https://arxiv.org/abs/1705.08848', 'official_JDOT_reproduction': False,
        'benchmark_and_query_cohort_unchanged': True, 'expected_conditions': 1327,
        'already_evaluated_development_cohort': True, 'independent_confirmation': False,
        'script_sha256': source.sha(__file__), 'constructor_sha256': source.sha(constructor.__file__),
        'feature_encoder_sha256': source.sha(RUN/'scripts/domain_overlap_context_v166.py'),
        'parent_plan_sha256': source.sha(parent.OUT/'plan.json'), 'Markdown_written': False}
    p = OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    predictors = {}
    for i, position in enumerate(np.random.default_rng(177).permutation(len(cases))):
        if i % shards != shard:
            continue
        c = cases[int(position)]
        task, tid = c['task'], c['task_id']
        if task not in predictors:
            predictors[task] = Predictor(task, 'cuda:0')
        predictor = predictors[task]
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as data:
            assert 'y_target' not in data.files
            X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
        K = len(np.unique(y0)) if task == 'classification' else None
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'banks'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            old = parent.OUT/'banks'/task/str(tid)/dest.name
            meta = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == meta['output_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            with np.load(old) as data:
                base, members = data['Base'], data['members'].copy()
                native8 = data['Canonical_native8']
                mu, sigma = float(data['source_mean']), float(data['source_scale'])
            if rho:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
            repaired, info = constructor.construct(X, y, Q, numeric, task, base)
            cache = {protocol.array_sha256(X): base}
            calls = 0
            for slot, A in zip((3, 4), repaired):
                key = protocol.array_sha256(A)
                if key not in cache:
                    for estimator in predictor.models.values():
                        root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                        if root is not None and hasattr(root, 'root'):
                            root.root = None
                    torch.cuda.empty_cache()
                    cache[key] = predictor(A, y, Q, members=1, random_state=0, n_classes=K)[0]
                    calls += 1
                members[slot] = cache[key]
            policy = json.loads((source.OUT/'policies'/task/f'{c["fold"]}.json').read_text())
            frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
                'mu': mu, 'sigma': sigma}, policy['Shared'])
            source.npwrite(dest, Base=base, members=members, FrozenFusion=frozen,
                Canonical_native8=native8, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'original_input_sha256': c['input_sha256'], 'parent_bank': str(old),
                'parent_bank_sha256': source.sha(old), 'replacement_slots': [3, 4],
                'source_y_sha256': protocol.array_sha256(y), 'repair_hashes': [protocol.array_sha256(A) for A in repaired],
                'new_context_rows': [len(y), len(y)], 'TFM_calls_added': calls,
                'transport': info, 'final_query_labels_opened': False,
                'constructor_adaptation': True, 'official_JDOT_reproduction': False})
        print('JOINT_TRANSPORT_CASE_COMPLETE', task, tid, flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[key] = '1'
        log = OUT/f'worker_{i}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i), '--shards', str(shards)],
                env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(log)}
    results = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            results.append(future.result())
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': results})
            print('JOINT_TRANSPORT_WORKER', results[-1], flush=True)
    success = all(r['returncode'] == 0 for r in results)
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': results})
    assert success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['launch', 'worker'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    launch(args.shards) if args.phase == 'launch' else worker(args.shard, args.shards)
