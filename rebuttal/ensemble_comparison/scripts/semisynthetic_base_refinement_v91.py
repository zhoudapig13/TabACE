"""Four neighboring Base-only settings, declared before their evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
import subprocess
import sys

import pandas as pd

import semisynthetic_shortcut_v90 as generator
from prepare_data import RUN, atomic_json
from abcd_mechanism_v84 import sha

OUT = RUN / 'results/semisynthetic_base_refinement_v91'
CONFIGS = [dict(name=f'realnoise1_signal{signal:g}_amplitude{amp:g}', real_noise=1., signal=signal, amplitude=amp)
           for signal, amp in itertools.product((.6, .65), (1.25, 1.5))]


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text())['code_sha256'] == sha(__file__)
        return
    atomic_json(p, {'status': 'FROZEN_BEFORE_BASE_ONLY_REFINEMENT', 'seed': 0,
        'datasets': 72, 'context_cap': 256, 'new_configurations': CONFIGS,
        'first_grid_path': str(generator.OUT / 'plan.json'), 'generator_sha256': sha(generator.__file__),
        'benchmark_protocol_unchanged': True, 'selection_criterion': 'same maximum four-metric discrepancy as v90',
        'ABCD_predictions_for_this_benchmark_exist_at_freeze': False,
        'target_labels_used_only_for_semisynthetic_generation_and_development_metrics': True,
        'independent_confirmation': False, 'original_paper_replication': False,
        'maximum_configurations_in_this_development_cycle': 16,
        'code_sha256': sha(__file__), 'Markdown_written': False})


def worker(shard, shards):
    generator.OUT = OUT
    generator.CONFIGS = CONFIGS
    generator.worker(shard, shards)


def report():
    paths = sorted(OUT.glob('detail_*.csv'))
    assert len(paths) == 8
    new = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert len(new) == 1440
    new.to_csv(OUT / 'detail.csv', index=False)
    old = pd.read_csv(generator.OUT / 'detail.csv')
    combined = pd.concat([old, new], ignore_index=True)
    assert len(combined) == 5760
    assert not combined.duplicated(['task_id', 'configuration', 'rho']).any()
    combined.to_csv(OUT / 'all_16_configuration_detail.csv', index=False)
    ref = json.loads((generator.OUT / 'plan.json').read_text())['reference']
    rows = []
    for name, group in combined.groupby('configuration'):
        clean = group[group.rho == 0][['accuracy', 'auc']].mean()
        disturbed = group[group.rho > 0].groupby('data_id')[['accuracy', 'auc']].mean().mean()
        values = {'clean_accuracy': clean.accuracy, 'clean_auc': clean.auc,
                  'shortcut_accuracy': disturbed.accuracy, 'shortcut_auc': disturbed.auc}
        errors = [abs(v - ref[k]) for k, v in values.items()]
        rows.append({'configuration': name, **values, 'max_abs_error': max(errors), 'squared_error': sum(v*v for v in errors)})
    table = pd.DataFrame(rows).sort_values(['max_abs_error', 'squared_error'])
    table.to_csv(OUT / 'all_16_configuration_summary.csv', index=False)
    selected = table.iloc[0].to_dict(); name = selected['configuration']
    config = next(c for c in generator.CONFIGS + CONFIGS if c['name'] == name)
    directory = OUT if name in {c['name'] for c in CONFIGS} else generator.OUT
    atomic_json(OUT / 'Base_selection.json', {'status': 'FROZEN_BEFORE_ABCD', 'configuration': config,
        'prediction_root': str(directory / 'predictions'), 'development_result': selected,
        'reference': ref, 'tolerance': .05, 'aligned_within_tolerance': selected['max_abs_error'] <= .05,
        'ABCD_metrics_used_for_selection': False, 'candidate_count': 16,
        'semisynthetic_generator_uses_target_labels': True, 'original_paper_replication': False,
        'independent_confirmation': False, 'seed': 0})
    print(table.to_string(index=False), flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(i)
        p = OUT / f'worker_{i}.log'
        with p.open('a') as handle:
            r = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i), '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'job': i, 'returncode': r.returncode, 'log': str(p)}
    results = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            r = future.result(); results.append(r); print(r, flush=True)
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE' if all(r['returncode'] == 0 for r in results) else 'FAILED', 'workers': results})
    if any(r['returncode'] for r in results):
        raise SystemExit(1)
    report()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['launch', 'worker', 'report'])
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    if a.phase == 'worker':
        worker(a.shard, a.shards)
    elif a.phase == 'report':
        report()
    else:
        launch(a.shards)
