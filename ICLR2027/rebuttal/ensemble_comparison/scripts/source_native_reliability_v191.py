"""Observed-source cross-fitting evidence for Native1 versus Native8.

Only existing source labels are used. No final query feature or outcome enters
the source validation callback. This adds calibration compute, not new labels.
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
import unseen_shared_robustness_v117 as source
import source_crossfit_context_v134 as crossfit
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor
from shared_task_residual_heads_v155 import metric

OUT = RUN/'results/source_native_reliability_v191'
BASE_CV = RUN/'results/shared_source_crossfit_v135'
NAMES = ('source_CV_primary_gain_Native8', 'source_CV_log_loss_ratio',
    'source_CV_paired_loss_gain_over_standard_error', 'source_CV_prediction_disagreement')


def reliability(y_source, prediction1, prediction8, task):
    y, a, b = np.asarray(y_source), np.asarray(prediction1), np.asarray(prediction8)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) if task == 'classification' else np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return np.zeros(4), {'scored_source_rows': 0, 'unscored_rare_source_rows': int(len(y)),
            'source_primary_Native1': None, 'source_primary_Native8': None,
            'no_valid_source_holdout_neutral_evidence': True,
            'paired_loss_gain_over_SE_is_a_feature_not_a_reported_significance_test': True}
    y, a, b = y[valid], a[valid], b[valid]
    primary1, _ = metric({'task': task, 'y': y}, a)
    primary8, _ = metric({'task': task, 'y': y}, b)
    if task == 'classification':
        labels = y.astype(int)
        loss1 = -np.log(np.clip(a[np.arange(len(y)), labels], 1e-12, 1.))
        loss8 = -np.log(np.clip(b[np.arange(len(y)), labels], 1e-12, 1.))
        disagreement = .5*np.abs(a-b).sum(axis=1).mean()
    else:
        sigma = max(float(np.std(y_source)), 1e-12)
        loss1, loss8 = ((a-y)/sigma)**2, ((b-y)/sigma)**2
        disagreement = np.sqrt(np.mean(((a-b)/sigma)**2))
    difference = loss1-loss8
    se = float(difference.std()/np.sqrt(len(difference)))
    values = np.array([primary8-primary1, np.log((loss1.mean()+1e-8)/(loss8.mean()+1e-8)),
        difference.mean()/max(se, 1e-3), disagreement], dtype=np.float64)
    values = np.clip(values, -3., 3.)
    assert np.isfinite(values).all()
    return values, {'scored_source_rows': int(valid.sum()), 'unscored_rare_source_rows': int((~valid).sum()),
        'source_primary_Native1': float(primary1), 'source_primary_Native8': float(primary8),
        'paired_loss_gain_over_SE_is_a_feature_not_a_reported_significance_test': True}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((BASE_CV/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'SOURCE_LABEL_ONLY_NATIVE_RELIABILITY_CALIBRATION_EXTENSION', 'seed': 0,
        'source_context_cap': 256, 'source_crossfit_folds': 3,
        'features': list(NAMES), 'same_source_partitions_for_Native1_and_Native8': True,
        'Native1_source_CV_predictions_reused_from': str(BASE_CV),
        'new_source_CV_Native8_inference_calls': 'up to3 per condition',
        'new_source_CV_estimator_forward_passes': 'up to24 per condition',
        'total_source_CV_estimator_forward_passes_including_Native1': 'up to27',
        'existing_observed_source_labels_used_for_local_calibration': True,
        'extra_labeled_rows': 0, 'clean_source_counterfactual_used': False,
        'final_query_X_or_y_used_by_source_validation': False,
        'Base_and_benchmark_and_final_query_cohort_unchanged': True,
        'no_corruption_name_or_severity_received_by_reliability_function': True,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'script_sha256': source.sha(__file__), 'source_folding_sha256': source.sha(crossfit.__file__),
        'Base_CV_plan_sha256': source.sha(BASE_CV/'plan.json'), 'expected_conditions': 1327,
        'frozen_checkpoints_sha256': {name: source.sha(source.CHECKPOINTS/f'tabicl-{name}-v2-20260212.ckpt')
            for name in ('classifier', 'regressor')},
        'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    predictors = {}
    for i, position in enumerate(np.random.default_rng(191).permutation(len(cases))):
        if i % shards != shard:
            continue
        c = cases[int(position)]
        task, tid = c['task'], c['task_id']
        if task not in predictors:
            predictors[task] = Predictor(task, 'cuda:0')
        predictor = predictors[task]
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as inputs:
            assert 'y_target' not in inputs.files
            X0, y0, Q0, numeric = (inputs[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
        K = len(np.unique(y0)) if task == 'classification' else None
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest = OUT/'features'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest) == json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            old = BASE_CV/'banks'/task/str(tid)/dest.name
            previous = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == previous['output_sha256']
            assert previous['original_input_sha256'] == c['input_sha256']
            with np.load(old) as data:
                a = data['SourceCV_prediction']
            if rho:
                X, y, _, _ = source.novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            else:
                X, y = X0, y0
            calls = 0
            def infer(A, b, T):
                nonlocal calls
                for estimator in predictor.models.values():
                    root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                    if root is not None and hasattr(root, 'root'):
                        root.root = None
                torch.cuda.empty_cache()
                calls += 1
                return predictor(A, b, T, members=8, random_state=0, n_classes=K)[0]
            loss8, b, mandatory, partitions = crossfit.quality(X, y, task, infer)
            assert partitions == previous['source_crossfit_partitions']
            values, info = reliability(y, a, b, task)
            source.npwrite(dest, source_reliability=values, SourceCV_Native1=a, SourceCV_Native8=b)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
                'original_input_sha256': c['input_sha256'], 'Base_CV_bank': str(old),
                'Base_CV_bank_sha256': source.sha(old), 'source_y_sha256': protocol.array_sha256(y),
                'source_crossfit_partitions': partitions, 'mandatory_rare_rows': mandatory.tolist(),
                'Native8_source_CV_calls': calls, 'features': info,
                'final_query_outcome_files_opened': False,
                'validation_callback_receives_source_train_Xy_and_source_heldout_X_only': True})
        print('SOURCE_NATIVE_RELIABILITY_CASE', task, tid, flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[key] = '1'
        log = OUT/f'worker_{i}.log'
        with log.open('a') as handle:
            result = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i), '--shards', str(shards)],
                env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': result.returncode, 'log': str(log)}
    values = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            values.append(future.result())
            atomic_json(OUT/'execution.json', {'status': 'RUNNING', 'workers': values})
            print('SOURCE_NATIVE_RELIABILITY_WORKER', values[-1], flush=True)
    count = len(list((OUT/'features').glob('*/*/*.json')))
    success = all(v['returncode'] == 0 for v in values) and count == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': values, 'conditions': count})
    assert success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['launch', 'worker'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    launch(args.shards) if args.phase == 'launch' else worker(args.shard, args.shards)
