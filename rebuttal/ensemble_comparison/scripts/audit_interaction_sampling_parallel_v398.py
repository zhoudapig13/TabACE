"""Run the full independent v368 audit on disjoint shards of one endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import pandas as pd
import audit_interaction_sampling_probe_v397 as audit
from prepare_data import atomic_json, ROOT

run = audit.run


def worker(shard):
    original = run.features.schedule.endpoint.endpoint_cases
    selected = [c for i, c in enumerate(original()) if i % 6 == shard]
    run.features.schedule.endpoint.endpoint_cases = lambda: selected
    dest = run.OUT / 'audit_shards' / str(shard)
    dest.mkdir(parents=True, exist_ok=True)
    def write(path, value):
        path = Path(path)
        return atomic_json(dest / path.name if path.parent == run.OUT else path, value)
    audit.atomic_json = write
    original_csv = pd.DataFrame.to_csv
    def csv(frame, path, *args, **kwargs):
        path = Path(path)
        return original_csv(frame, dest / path.name if path.parent == run.OUT else path, *args, **kwargs)
    pd.DataFrame.to_csv = csv
    audit.main(pilot=False, replay=False)
    atomic_json(dest / 'case_partition.json', {'shard': shard, 'task_ids': [c['task_id'] for c in selected]})


def launch():
    assert json.loads((run.OUT / 'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'active_endpoint': ['shift', 'classification', 'interaction_sampling'],
            'CPU_workers': 6, 'GPU_workers': 0, 'audit_script_sha256': run.source.sha(audit.__file__),
            'driver_sha256': run.source.sha(__file__), 'independent_refits': True,
            'case_partition': 'original35 case order modulo6', 'other_endpoints_not_started': True}
    p = run.OUT / 'parallel_audit_plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)
    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''
        log = run.OUT / f'audit_CPU_worker_{i}.log'
        with log.open('a') as handle:
            process = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i)],
                                     env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': process.returncode, 'log': str(log)}
    atomic_json(run.OUT / 'parallel_audit_execution.json', {'status': 'RUNNING', **plan})
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(execute, range(6)))
    assert all(r['returncode'] == 0 for r in results), results
    frames, partitions, conditions = [], [], 0
    for i in range(6):
        dest = run.OUT / 'audit_shards' / str(i)
        report = json.loads((dest / 'independent_audit.json').read_text())
        assert report['status'] == 'PASS' and not report['pilot_only']
        conditions += report['conditions']
        partition = json.loads((dest / 'case_partition.json').read_text())['task_ids']
        partitions.extend(partition)
        frame = pd.read_csv(dest / 'condition_metrics.csv')
        assert set(frame.task_id) == set(partition)
        frames.append(frame)
    expected = [c['task_id'] for c in run.features.schedule.endpoint.endpoint_cases()]
    assert len(partitions) == len(set(partitions)) == 35 and set(partitions) == set(expected)
    assert conditions == 140
    df = pd.concat(frames, ignore_index=True)
    assert len(df) == 140 * 5 and not df.duplicated(['task_id', 'rho', 'method']).any()
    df.to_csv(run.OUT / 'condition_metrics.csv', index=False)
    ds = df.groupby(['data_id', 'method'])[['accuracy', 'auc']].mean().reset_index()
    table = ds.groupby('method')[['accuracy', 'auc']].mean().reset_index()
    table['datasets'] = 35
    table.to_csv(run.OUT / 'family_averages.csv', index=False)
    get = lambda method: float(table.loc[table.method == method, 'accuracy'].iloc[0])
    gain = get('Shared_probe') - get('Base')
    atomic_json(run.OUT / 'numerical_acceptance.json', {
        'pilot_only': False, 'active_endpoint': plan['active_endpoint'], 'conditions': 140,
        'datasets': 35, 'Base': get('Base'), 'Shared_supervised_probe': get('Shared_probe'),
        'gain': gain, 'gain_vs_Native_same_supervision': get('Shared_probe')-get('Native_probe'),
        'required_gain_strictly_greater_than': .02, 'active_endpoint_pass': gain > .02,
        'all_twelve_pass': False, 'extra_source_and_test_subset_supervision_explicit': True})
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    run.features.schedule.verify_frozen()
    atomic_json(run.OUT / 'independent_audit.json', {
        'status': 'PASS', 'pilot_only': False, 'conditions': 140, 'TFM_contexts_reconstructed': 700,
        'reference_and_target_and_eval_row_ids_checked': True,
        'linear_heads_and_four_fold_validation_predictions_independently_refitted': True,
        'cached_OOF_predictions_reverified_by_full_independent_refits': True,
        'all_target_adaptation_rows_excluded_from_evaluation': True,
        'original_frozen_Base_predictions_bitwise_preserved': True,
        'all_six_disjoint_shards_passed': True, 'Markdown_bytes': 0,
        'not_every_neural_prediction_recomputed': True,
        'audit_script_sha256': plan['audit_script_sha256'], 'driver_sha256': plan['driver_sha256']})
    atomic_json(run.OUT / 'parallel_audit_execution.json', {'status': 'COMPLETE', 'conditions': 140, 'workers': results})
    print(table.to_string(index=False), flush=True)
    print('GAIN', gain, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['launch', 'worker'])
    p.add_argument('--shard', type=int, default=0)
    args = p.parse_args()
    launch() if args.phase == 'launch' else worker(args.shard)
