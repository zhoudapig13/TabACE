"""Fast progress reports on complete per-dataset grids, retaining both variants."""
import csv
import json
from pathlib import Path

import numpy as np

from prepare_data import atomic_json
from tabpfn_official_benchmark_v2 import OUT, PLAN, tasks, conditions, path_for

METHODS = {'classification': ['Native6', 'PHE6', 'RF_PFN6', 'MixturePFN_MICP6_no_CaPFN'],
           'regression': ['Native6', 'PHE6', 'RF_PFN6']}
KEYS = {'classification': ['accuracy', 'auc'], 'regression': ['r2', 'nrmse']}


def read(path):
    return json.loads(path.read_text())


def mean(rows, keys):
    if not rows:
        return None
    return {key: float(np.mean([row[key] for row in rows if row.get(key) is not None]))
            if any(row.get(key) is not None for row in rows) else None for key in keys}


def baseline(task, track, meta, family, rho):
    block = 'clean' if track == 'five' and rho == 0 else track
    frozen = read(OUT / f'frozen_{task}_{block}.json')
    return read(path_for(task, block, frozen['settings']['name'], meta, family, rho).with_suffix('.json'))['metrics']


def official_records(task):
    methods = METHODS[task]
    keys = KEYS[task]
    rows = {('Base', 'original'): {'task': task, 'method': 'Base', 'variant': 'original'}}
    for method in methods:
        for variant in ['raw', 'adapted']:
            rows[(method, variant)] = {'task': task, 'method': method, 'variant': variant,
                'backbone_gradient_finetuning': False, 'extra_clean_source_supervision_in_fit_or_selection': variant == 'adapted'}
    progress = {}
    for track in ['five', 'seven']:
        datasets = tasks(task, track)
        path = OUT / 'official_predictions' / task / track
        complete = []
        for meta in datasets:
            grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
            if all((path / str(meta['task_id']) / f'{f}_{r}__{method}.json').exists()
                   for f, r in grid for method in methods):
                complete.append(meta)
        progress[track] = {'completed_datasets': len(complete), 'total_datasets': len(datasets),
            'completed_method_conditions': len(list(path.glob('*/*.json'))),
            'expected_method_conditions': len(datasets) * len(grid) * len(methods)}
        accum = {k: [] for k in rows}
        clean = {k: [] for k in rows}
        for meta in complete:
            grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
            for family, rho in grid:
                target = clean if rho == 0 else accum
                target[('Base', 'original')].append(baseline(task, track, meta, family, rho))
                for method in methods:
                    observed = read(path / str(meta['task_id']) / f'{family}_{rho}__{method}.json')
                    for variant in ['raw', 'adapted']:
                        target[(method, variant)].append(observed[variant + '_metrics'])
        for key, row in rows.items():
            row[track] = mean(accum[key], keys)
            row[track + '_datasets'] = len(complete)
            if track == 'five':
                row['clean'] = mean(clean[key], keys)
                row['clean_datasets'] = len(complete)
    return list(rows.values()), progress


def oligo_records():
    rows = []
    for total in [60, 400]:
        records = {}
        for name in [f'OligoICP_{total}to{total // 10}_groupadapted', f'FullMean_{total}_same_candidate_budget']:
            for variant in ['raw', 'adapted']:
                records[(name, variant)] = {'task': 'regression', 'method': name, 'variant': variant,
                    'candidate_count': total, 'extra_clean_source_supervision_in_fit_or_selection': variant == 'adapted',
                    'backbone_gradient_finetuning': False}
        for track in ['five', 'seven']:
            accum = {k: [] for k in records}; clean = {k: [] for k in records}
            complete = []
            for meta in tasks('regression', track):
                grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
                folder = OUT / 'oligo_predictions' / track / str(meta['task_id'])
                if all((folder / f'{f}_{r}' / f'results_{total}.json').exists() for f, r in grid):
                    complete.append(meta)
            for meta in complete:
                grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
                for family, rho in grid:
                    observed = read(OUT / 'oligo_predictions' / track / str(meta['task_id']) / f'{family}_{rho}' / f'results_{total}.json')
                    target = clean if rho == 0 else accum
                    for (name, variant) in records:
                        field = variant + '_metrics' if name.startswith('Oligo') else 'equal_candidate_budget_full_mean' + ('_adapted' if variant == 'adapted' else '') + '_metrics'
                        target[(name, variant)].append(observed[field])
            for key, row in records.items():
                row[track] = mean(accum[key], KEYS['regression'])
                row[track + '_datasets'] = len(complete)
                if track == 'five':
                    row['clean'] = mean(clean[key], KEYS['regression'])
                    row['clean_datasets'] = len(complete)
        rows.extend(records.values())
    return rows


def main():
    plan = read(PLAN)
    records = []; progress = {}
    for task in ['classification', 'regression']:
        rows, progress[task] = official_records(task)
        records.extend(rows)
    records.extend(oligo_records())
    frozen = {task: {track: read(OUT / f'frozen_{task}_{track}.json')['metrics']
                    for track in ['clean', 'five', 'seven']} for task in ['classification', 'regression']}
    for row in records:
        primary = 'accuracy' if row['task'] == 'classification' else 'r2'
        row['historical_range_checks_descriptive_only'] = {}
        for track in ['five', 'seven']:
            if row.get(track):
                low = frozen[row['task']][track][primary]
                high = plan['historical_Shared_references'][row['task']][track][primary]
                row['historical_range_checks_descriptive_only'][track] = low <= row[track][primary] <= high
    complete = all(p['completed_datasets'] == p['total_datasets'] for t in progress.values() for p in t.values())
    oligo_complete = all(row['five_datasets'] == 35 and row['seven_datasets'] == 33 for row in records if row['method'].startswith('OligoICP_400'))
    result = {'status': 'COMPUTATION_COMPLETE_AUDIT_PENDING' if complete and oligo_complete else 'RUNNING',
        'seed': 0, 'outer_repetitions': 1, 'internal_development_only': True, 'paper_protocol_reproduction': False,
        'cross_backbone_benchmark_configurations_differ': True, 'benchmark_frozen_Base_full_grid': frozen,
        'progress': progress, 'records': records,
        'partial_scope': 'Each column averages only complete per-dataset grids shared by the official methods; counts are explicit',
        'range_checks_select_or_stop_experiments': False, 'raw_and_adapted_outputs_both_preserved': True,
        'adaptation_is_auxiliary_supervision_not_backbone_weight_finetuning': True, 'Markdown_written': False}
    dest = OUT / 'reports'; dest.mkdir(exist_ok=True)
    atomic_json(dest / 'latest_summary.json', result)
    fields = ['task', 'method', 'variant', 'clean_datasets', 'five_datasets', 'seven_datasets',
              'Clean_primary', 'Clean_secondary', 'Five_primary', 'Five_secondary', 'Seven_primary', 'Seven_secondary',
              'extra_clean_source_supervision_in_fit_or_selection']
    with (dest / 'averages.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in records:
            flat = {key: row.get(key, '') for key in fields}
            for track in ['clean', 'five', 'seven']:
                pair = row.get(track)
                for key, metric in zip(['primary', 'secondary'], KEYS[row['task']]):
                    flat[track.capitalize() + '_' + key] = pair[metric] if pair else ''
            writer.writerow(flat)
    print('PROGRESS', json.dumps(progress), flush=True)
    for row in records:
        def pair(track):
            values = row.get(track)
            if not values:
                return '--'
            return '/'.join(f'{values[k]:.6f}' if values.get(k) is not None else 'NA' for k in KEYS[row['task']])
        if any(row.get(t) for t in ['clean', 'five', 'seven']):
            print(row['task'], row['method'], row['variant'], *[pair(t) for t in ['clean', 'five', 'seven']],
                  'datasets', row['five_datasets'], row['seven_datasets'], flush=True)


if __name__ == '__main__':
    main()
