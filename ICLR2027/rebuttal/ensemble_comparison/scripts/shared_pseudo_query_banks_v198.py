"""Two query-cross-fitted pseudo-context experts in a six-output ensemble.

No final query label is loaded. A fixed raw committee supplies pseudo labels;
using a query-calibrated learned teacher here would require nested refitting.
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
import shared_dual_canonical_banks_v182 as parent
import pseudo_query_contexts_v197 as constructor
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

source = parent.source
OUT = RUN/'results/shared_pseudo_query_v198'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'PSEUDO_QUERY_CONTEXT_EXTENSION_NOT_ORIGINAL_SHARED', 'seed': 0,
        'context_row_cap': 256, 'final_predictor_outputs_per_query_including_Base': 6,
        'replaced_slots': [3, 4], 'first_three_alternative_predictions_unchanged': True,
        'teacher': 'classification normalized mean6; regression median6 clipped to observed source-y range',
        'teacher_has_query_label_calibrated_learned_weights': False,
        'teacher_full_raw_context_bank_required': True,
        'pseudo_row_fractions_of_context_cap': [.25, .5], 'query_folds': 4,
        'each_query_excluded_from_its_own_pseudo_context': True,
        'source_class_support_preserved': True, 'new_true_labeled_rows': 0,
        'unlabeled_query_features_used_transductively': True,
        'final_query_outcomes_used_for_pseudo_labels': False,
        'constructor_receives_generator_name_or_severity': False,
        'backbone_weights_updated': False, 'Base_and_benchmark_unchanged': True,
        'maximum_raw_teacher_estimator_passes': 13,
        'maximum_added_estimator_calls_for_full_query_batch': 8,
        'maximum_query_prediction_estimator_calls_for_full_batch': 21,
        'maximum_estimator_predictions_contributing_to_each_query': 15,
        'extra_source_CV_for_teacher_generic_bank': {'classification': 0, 'regression': 3},
        'compute_matched_to_Native1_Base': False,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'expected_conditions': 1327, 'script_sha256': source.sha(__file__),
        'constructor_sha256': source.sha(constructor.__file__),
        'parent_bank_plan_sha256': source.sha(parent.OUT/'plan.json'), 'Markdown_written': False}
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
    for i, position in enumerate(np.random.default_rng(198).permutation(len(cases))):
        if i % shards != shard:
            continue
        c = cases[int(position)]
        task, tid = c['task'], c['task_id']
        if task not in predictors:
            predictors[task] = Predictor(task, 'cuda:0')
        predictor = predictors[task]
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0, y0, Q0, numeric = (z[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
        K = len(np.unique(y0)) if task == 'classification' else None
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'banks'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            old = parent.OUT/'banks'/task/str(tid)/dest.name
            old_meta = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == old_meta['output_sha256']
            assert old_meta['original_input_sha256'] == c['input_sha256']
            with np.load(old) as z:
                base, members = z['Base'], z['members'].copy()
                P = np.concatenate([base[None], members], axis=0)
                native8 = z['Canonical_native8']
                mu, sigma = float(z['source_mean']), float(z['source_scale'])
            if rho:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y, Q = X0, y0, Q0
            plans, teacher = constructor.contexts(X, y, Q, P, task)
            records, calls = [], 0
            for slot, branch in zip((3, 4), plans):
                prediction = np.empty_like(base)
                details = []
                for p in branch:
                    ids = p['query_rows']
                    if p['no_pseudo_fallback']:
                        prediction[ids] = base[ids]
                    else:
                        for estimator in predictor.models.values():
                            root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                            if root is not None and hasattr(root, 'root'):
                                root.root = None
                        torch.cuda.empty_cache()
                        prediction[ids] = predictor(p['X'], p['y'], Q[ids], members=1,
                            random_state=0, n_classes=K)[0]
                        calls += 1
                    details.append({name: p[name].tolist() for name in ('query_rows', 'source_rows', 'pseudo_query_rows')} |
                        {'context_X_sha256': protocol.array_sha256(p['X']),
                        'context_y_sha256': protocol.array_sha256(p['y']),
                        'pseudo_label_sha256': protocol.array_sha256(p['pseudo_labels']),
                        'context_rows': len(p['y']), 'no_pseudo_fallback': p['no_pseudo_fallback']})
                assert np.isfinite(prediction).all()
                members[slot] = prediction
                records.append(details)
            policy = json.loads((source.OUT/'policies'/task/f'{c["fold"]}.json').read_text())
            frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
                'mu': mu, 'sigma': sigma}, policy['Shared'])
            source.npwrite(dest, Base=base, members=members, Teacher=teacher, Canonical_native8=native8,
                FrozenFusion=frozen, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'original_input_sha256': c['input_sha256'], 'parent_bank': str(old),
                'parent_bank_sha256': source.sha(old), 'query_crossfit_contexts': records,
                'new_TFM_calls': calls, 'true_source_labels_sha256': protocol.array_sha256(y),
                'final_query_outcome_files_opened': False, 'learned_query_label_teacher_used': False})
        print('PSEUDO_QUERY_CONTEXT_CASE', task, tid, flush=True)


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
    values = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            values.append(future.result())
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': values})
            print('PSEUDO_QUERY_CONTEXT_WORKER', values[-1], flush=True)
    count = len(list((OUT/'banks').glob('*/*/*.json')))
    success = all(p['returncode'] == 0 for p in values) and count == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': values, 'conditions': count})
    assert success


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    args = p.parse_args()
    launch(args.shards) if args.phase == 'launch' else worker(args.shard, args.shards)
