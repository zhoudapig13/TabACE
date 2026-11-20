"""One stronger member-randomization sensitivity, five perturbations only.

No tuning toward a decimal pattern or a score interval. All outcomes retained.
Clean/seven benchmark artifacts and original FT checkpoints remain read-only.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from prepare_data import RUN, atomic_json, protocol
from internal_regression_benchmark import manifest

OUT = RUN / 'results/tabicl_five_randomization_seed0_v21'
PLAN = OUT / 'plan.json'
PRIOR = RUN / 'results/tabicl_randomized_members_sensitivity_v4'
TRAINED = RUN / 'results/tabicl_regression_weight_adaptation_v3'
TAG = 'five_only_seed0'


@lru_cache(None)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declare():
    assert not PLAN.exists()
    assert json.loads((PRIOR / 'reports/single_seed0/audit.json').read_text())['status'] == 'PASS'
    frozen = []
    for path in (PRIOR / 'benchmark_predictions/single_seed0').glob('*/*/*'):
        if path.suffix not in ['.json', '.npz'] or '.tmp.' in path.name:
            continue
        if path.parent.parent.name == 'seven' or path.stem == 'original_0':
            stat = path.stat()
            frozen.append([str(path.relative_to(RUN)), stat.st_size, stat.st_mtime_ns])
    reports = {str(p.relative_to(RUN)): sha(p) for p in
               [PRIOR / 'reports/single_seed0/latest_summary.json',
                PRIOR / 'reports/single_seed0/averages.csv']}
    atomic_json(PLAN, {
        'status': 'DECLARED_BEFORE_NEW_PREDICTIONS', 'outer_seed': 0,
        'outer_repetitions': 1, 'new_configurations': 1, 'candidate_count': 6,
        'feature_bank': 'all canonical rows; column fraction Uniform(0.50,1.00)',
        'bootstrap_bank': 'row fraction Uniform(0.50,1.00), replacement=True; column fraction Uniform(0.50,1.00)',
        'streams': 'SeedSequence([0, task_id, 21, bank_id, member_id]).spawn(4)',
        'same_random_banks_for_applicable_fusion_methods': True,
        'fractions_fixed_per_dataset_member_across_conditions': True,
        'benchmark_nonzero_five_only': True, 'datasets': 35, 'conditions': 700,
        'strengths': [.2, .5, .8, 1.], 'Base_and_Shared_inference': False,
        'Base_Shared_role': 'historical reference only, not parameter selection',
        'reference_intervals': {'r2': [-1.077, .363], 'nrmse': [.783, 1.270]},
        'intervals_are_descriptive_checks_only': True,
        'decimal_prefix_and_score_dispersion_selection': False,
        'all_results_reported_even_if_similar_or_outside_reference_interval': True,
        'checkpoints_retrained': False, 'extra_clean_TRAIN_supervision_same_as_v4': True,
        'combiner_refit': 'existing algorithm and candidate grid, official TRAIN validation only',
        'benchmark_query_labels_used_for_fit_selection': False,
        'Clean_seven_frozen_file_index': frozen, 'previous_reports_sha256': reports,
        'execution_script_sha256': sha(Path(__file__)), 'Markdown_written': False,
    })
    print('DECLARED', len(frozen), 'frozen Clean/seven artifacts', flush=True)


def member_spec(task_id, bank, member, n, p):
    streams = np.random.SeedSequence([0, int(task_id), 21, bank, member]).spawn(4)
    rng = np.random.default_rng(streams[0])
    rf = 1. if bank == 0 else float(rng.uniform(.5, 1.))
    cf = float(rng.uniform(.5, 1.))
    count = min(n, max(2, int(np.ceil(rf * n))))
    columns = min(p, max(1, int(np.ceil(cf * p))))
    rows = np.arange(n) if bank == 0 else np.random.default_rng(streams[1]).choice(n, count, replace=True)
    cols = np.sort(np.random.default_rng(streams[2]).choice(p, columns, replace=False))
    seed = int(streams[3].generate_state(1, dtype=np.uint32)[0])
    return rows, cols, seed, rf, cf


def bank(predictor, X, y, Q, controls=None):
    if controls is None:
        single, _ = predictor(X, y, Q, quantiles=True, random_state=0)
        native, _ = predictor(X, y, Q, members=6, random_state=0)
    else:
        single, native = controls
    result = {'single': single, 'native': native}
    for bank_id, name in [(0, 'feature'), (1, 'bootstrap')]:
        predictions, uncertainty = [], []
        indices = np.full((6, len(y)), -1, dtype=np.int64)
        masks = np.zeros((6, X.shape[1]), dtype=bool)
        seeds, fractions = [], []
        for member in range(6):
            rows, cols, seed, rf, cf = member_spec(predictor.task_id, bank_id, member, len(y), X.shape[1])
            pred, iqr = predictor(X[rows][:, cols], y[rows], Q[:, cols], quantiles=True, random_state=seed)
            predictions.append(pred); uncertainty.append(iqr)
            indices[member, :len(rows)] = rows; masks[member, cols] = True
            seeds.append(seed); fractions.append([rf, cf])
        result[name] = np.stack(predictions)
        result[name + '_row_indices'] = indices
        result[name + '_column_masks'] = masks
        result[name + '_member_seeds'] = np.asarray(seeds, dtype=np.uint32)
        result[name + '_fractions'] = np.asarray(fractions)
        if name == 'bootstrap': result['iqr'] = np.stack(uncertainty)
    return result


def worker(shard, shards):
    import torch
    import warnings
    import evaluate_tabicl_finetuned_ensembles_v3 as engine
    from tabicl_regression_comparison_v2 import load_condition
    from run_tabicl import metrics
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    warnings.filterwarnings('ignore')
    engine.OUT = OUT
    engine.bank = bank
    for meta in manifest()[shard::shards]:
        task_id = meta['task_id']
        checkpoint = TRAINED / 'checkpoints/initial128' / str(task_id) / 'best.ckpt'
        checkpoint_sha = sha(checkpoint)
        predictor = engine.UpdatedPredictor(checkpoint)
        predictor.task_id = task_id
        policy_path = OUT / 'combiner_policies' / f'{task_id}.json'
        if not policy_path.exists():
            cells = engine.calibration(meta, predictor, TAG, checkpoint_sha)
            models, trials = engine.calibrate(cells)
            atomic_json(policy_path, {'models': models, 'trials': trials,
                        'checkpoint_sha256': checkpoint_sha, 'benchmark_query_labels_used': False,
                        'role': 'TRAIN-validation-only calibration; benchmark Clean/seven unchanged'})
        else:
            policy = json.loads(policy_path.read_text())
            assert policy['checkpoint_sha256'] == checkpoint_sha
            models = policy['models']
        print('TRAIN_CALIBRATION_COMPLETE', task_id, flush=True)
        for family in protocol.FAMILIES:
            for rho in [20, 50, 80, 100]:
                path = OUT / 'predictions/five' / str(task_id) / f'{family}_{rho}.npz'
                if path.exists() and path.with_suffix('.json').exists(): continue
                start = time.time()
                data, info, _ = load_condition('five', meta, family, rho)
                old = PRIOR / 'benchmark_predictions/single_seed0/five' / str(task_id) / path.name
                old_meta = json.loads(old.with_suffix('.json').read_text())
                hashes = {k: protocol.array_sha256(data[k]) for k in ['X_source', 'y_source', 'X_target', 'y_target']}
                assert old_meta['input_hashes'] == hashes
                with np.load(old) as saved:
                    controls = saved['single'].copy(), saved['native'].copy()
                    reference = saved['original_Base'].copy()
                    assert np.array_equal(saved['y_target'], data['y_target'])
                b = bank(predictor, data['X_source'], data['y_source'], data['X_target'], controls)
                preds = {method: engine.prediction(b, method, models[method]) for method in engine.METHODS}
                assert all(np.isfinite(p).all() for p in preds.values())
                observed = {method: metrics(data['y_target'], p, 'regression', info['scale']) for method, p in preds.items()}
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, **b, y_target=data['y_target'], original_Base=reference,
                                    **{'prediction__' + k: p for k, p in preds.items()})
                tmp.replace(path)
                atomic_json(path.with_suffix('.json'), {
                    'status': 'COMPLETE', 'task_id': task_id, 'dataset': meta['dataset'],
                    'family': family, 'rho': rho, 'seed': 0, 'candidate_count': 6,
                    'source_rows': len(data['y_source']), 'source_features': data['X_source'].shape[1],
                    'scale': info['scale'], 'methods': observed, 'input_hashes': hashes,
                    'checkpoint_sha256': checkpoint_sha, 'checkpoint_retrained': False,
                    'benchmark_query_labels_used_for_training_or_selection': False,
                    'control_predictions_reused_from': str(old.relative_to(RUN)),
                    'Base_and_Shared_not_rerun': True, 'seconds': time.time() - start,
                })
        print('FIVE_DATASET_COMPLETE', task_id, meta['dataset'], flush=True)
    atomic_json(OUT / f'worker_{shard}.json', {'status': 'COMPLETE'})


def launch():
    def one(gpu):
        env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        with (RUN / 'logs' / f'tabicl_five_randomization_v21_gpu{gpu}.log').open('a') as f:
            p = subprocess.run([str(RUN / '.venv/bin/python'), '-u', str(Path(__file__)), 'worker',
                                '--shard', str(gpu), '--shards', '8'], env=env, stdout=f, stderr=subprocess.STDOUT)
        row = {'gpu': gpu, 'exit_code': p.returncode}; print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=8) as pool: rows = list(pool.map(one, range(8)))
    atomic_json(OUT / 'execution.json', rows)
    if any(r['exit_code'] for r in rows): raise SystemExit(1)


def report(complete):
    import pandas as pd
    files = [p for p in (OUT / 'predictions/five').glob('*/*.json') if not p.name.endswith('.tmp.json')]
    rows = []
    for path in files:
        m = json.loads(path.read_text())
        for method, result in m['methods'].items():
            rows.append({'method': method, 'task_id': m['task_id'], 'family': m['family'], 'rho': m['rho'], **result})
    frame = pd.DataFrame(rows)
    summary = []
    for method, g in frame.groupby('method', sort=False):
        row = {'method': method, 'conditions': len(g), 'complete': len(g) == 700,
               'r2': float(g.r2.mean()), 'nrmse': float(g.nrmse.mean())}
        row['within_historical_reference_intervals'] = -1.077 <= row['r2'] <= .363 and .783 <= row['nrmse'] <= 1.270
        summary.append(row)
    if complete:
        assert len(files) == 700 and len(summary) == 10 and all(r['complete'] for r in summary)
        plan = json.loads(PLAN.read_text())
        for relative, size, mtime in plan['Clean_seven_frozen_file_index']:
            stat = (RUN / relative).stat(); assert (stat.st_size, stat.st_mtime_ns) == (size, mtime)
        for relative, expected in plan['previous_reports_sha256'].items(): assert sha(RUN / relative) == expected
    frame.to_csv(OUT / 'detail.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'averages.csv', index=False)
    atomic_json(OUT / 'summary.json', {'complete': len(files) == 700, 'records': summary,
                'single_seed': 0, 'one_fixed_randomization_config': True,
                'Clean_seven_unchanged': True, 'Base_Shared_not_rerun': True,
                'interval_checks_do_not_filter_results': True})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['declare', 'launch', 'worker', 'report', 'live'])
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    if args.phase == 'declare': declare()
    elif args.phase == 'launch': launch()
    elif args.phase == 'worker': worker(args.shard, args.shards)
    else: report(args.phase == 'report')
