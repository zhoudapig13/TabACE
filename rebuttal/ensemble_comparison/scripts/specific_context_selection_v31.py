"""Nested dataset-OOF selection of deployed Specific context members.

Shared is frozen to the measured repair+5% condition. Specific's candidate pool
contains all Shared members, repaired family members, and the canonical clean
64/128-reference contexts. Only source-calibration datasets select members.
Selection may prune Shared members; candidate inclusion is not a monotonicity
claim for a fixed deployed list. Every deployed list and count is exported.
"""
from pathlib import Path
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import traceback

os.environ['TABACE_REPAIR_MIX_RATE'] = '0.05'
import reference_repair_contexts_v28 as guided
import numpy as np
import pandas as pd
import torch
import tabace_fusion_v28 as fusion
from prepare_data import atomic_json
from run_tabicl import metrics

core = guided.core
SOURCE = guided.OUT
OUT = core.RUN / 'results/specific_context_selection_v31'
VARIANT = 'nested_specific_selection'


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    if p.exists():
        return
    plan = json.loads((SOURCE / 'plan.json').read_text())
    atomic_json(p, plan | dict(
        status='DECLARED_BEFORE_MEMBER_SELECTION_AND_NEW_SPECIFIC_SCORING',
        variants=[VARIANT], Shared_frozen_source=str(SOURCE),
        candidate_alternatives=12,
        candidates=['five inherited Shared', 'five repaired family-directed', 'canonical clean128', 'canonical clean64'],
        candidate_subsets=['Shared fallback', 'Base fallback', 'all first11', 'canonical128 only',
                           'canonical64 only', 'best-one Shared plus canonical128', 'best-three Shared plus canonical128'],
        best_Shared_member_ranking='inner-training calibration loss only',
        selection='outer five-fold dataset OOF; within outer train use every fourth sorted dataset ID as inner validation',
        selection_loss='classification NLL / regression target-scale-normalized MSE, averaged per dataset',
        selected_members_refit_on_all_outer_training_calibration=True,
        Specific_source_reference_access=128, Shared_source_reference_access=64,
        actual_Specific_branch_count='selected per family; fully exported; Base fallback has zero alternatives',
        all_Shared_contexts_in_candidate_pool=True,
        all_Shared_contexts_required_in_deployed_subset=False,
        query_labels_used_for_member_selection=False,
        paper_target_values_used_for_selection=False,
        original_fusion_formula_unchanged=True,
        development_followup_not_original_archived_member_lists=True,
        start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    ))


def prepare():
    for task in ['classification', 'regression']:
        for role in ['validation', 'query']:
            path = OUT / f'cells_{task}_{role}.pkl'
            if path.exists():
                continue
            cells = guided.few.load_cells(task, '', role)
            for c in cells:
                raw = Path(c['path'])
                other = core.RUN / 'results/few_reference_contexts_v27_ref128' / raw.relative_to(core.RAW)
                m = json.loads(other.with_suffix('.json').read_text())
                old = json.loads(raw.with_suffix('.json').read_text())
                assert m['frozen_input_hashes'] == old['input_hashes']
                with np.load(other) as z:
                    canonical = z['reference']
                c['bank'] = np.concatenate([c['bank'], canonical[None], c['reference'][None]], axis=0)
            with path.open('wb') as stream:
                pickle.dump(cells, stream, protocol=pickle.HIGHEST_PROTOCOL)
            print('PREPARED', task, role, len(cells), flush=True)


def subset(cells, indices):
    return [c | dict(bank=c['bank'][indices]) for c in cells]


def score(cells, policy, indices):
    by_dataset = {}
    for c in cells:
        p = c['base'] if not indices else fusion.predict(c | dict(bank=c['bank'][indices]), policy)
        if c['task'] == 'classification':
            y = c['y'].astype(int)
            loss = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        else:
            loss = float(np.mean(((p-c['y'])/c['meta']['scale'])**2))
        by_dataset.setdefault(c['meta']['data_id'], []).append(loss)
    return float(np.mean([np.mean(v) for v in by_dataset.values()]))


def member_loss(cells, index):
    by_dataset = {}
    for c in cells:
        p = c['bank'][index]
        if c['task'] == 'classification':
            y = c['y'].astype(int)
            value = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        else:
            value = float(np.mean(((p-c['y'])/c['meta']['scale'])**2))
        by_dataset.setdefault(c['meta']['data_id'], []).append(value)
    return float(np.mean([np.mean(v) for v in by_dataset.values()]))


def fit(cells, task, indices):
    return core.fit(subset(cells, indices), task, len(indices), 'cuda:0')


def run_fold(task, fold):
    torch.set_num_threads(1)
    dest = OUT / 'folds' / task / str(fold)
    if (dest / 'complete.json').exists():
        return
    with (OUT / f'cells_{task}_validation.pkl').open('rb') as stream:
        cal = pickle.load(stream)
    folds = json.loads((OUT / 'plan.json').read_text())['folds'][task]
    train = [c for c in cal if folds[str(c['meta']['data_id'])] != fold]
    outer_ids = sorted({c['meta']['data_id'] for c in train})
    inner_ids = set(outer_ids[::4])
    inner_train = [c for c in train if c['meta']['data_id'] not in inner_ids]
    inner_val = [c for c in train if c['meta']['data_id'] in inner_ids]
    assert inner_train and inner_val
    shared_inner = fit(inner_train, task, [0, 1, 2, 3, 4])
    old_policies = json.loads((SOURCE / 'policies/reference_repair064_mix005' / task /
                              f'fold{fold}/policies.json').read_text())
    shared_full = old_policies['Shared']
    policies = {}
    for group in sorted({c['group'] for c in train}):
        a = [c for c in inner_train if c['group'] == group]
        b = [c for c in inner_val if c['group'] == group]
        full = [c for c in train if c['group'] == group]
        assert a and b, group
        ranked = sorted(range(5), key=lambda i: member_loss(a, i))
        candidates = [
            ('Shared', [0, 1, 2, 3, 4], shared_inner),
            ('Base', [], dict(kind='base')),
            ('full11', list(range(11)), None),
            ('canonical128', [10], None),
            ('canonical64', [11], None),
            ('top1Shared_canonical128', [ranked[0], 10], None),
            ('top3Shared_canonical128', ranked[:3] + [10], None),
        ]
        trials = []
        for name, indices, policy in candidates:
            if policy is None:
                policy = fit(a, task, indices)
            value = score(b, policy, indices)
            trials.append(dict(name=name, indices=indices, validation_loss=value, model=policy))
        best = min(trials, key=lambda t: (t['validation_loss'], len(t['indices'])))
        if best['name'] == 'Shared':
            final = shared_full
        elif best['name'] == 'Base':
            final = dict(kind='base')
        else:
            final = fit(full, task, best['indices'])
        key = '/'.join(group)
        policies[key] = dict(indices=best['indices'], name=best['name'], model=final,
                             deployed_alternatives=len(best['indices']),
                             trials=[{k: t[k] for k in ['name', 'indices', 'validation_loss']} for t in trials])
        print('SELECTED', task, fold, key, best['name'], best['indices'], flush=True)
    atomic_json(dest / 'policies.json', dict(
        Shared=shared_full, Specific=policies, outer_train_data_ids=outer_ids,
        inner_validation_data_ids=sorted(inner_ids),
        query_labels_used_for_selection=False))
    with (OUT / f'cells_{task}_query.pkl').open('rb') as stream:
        queries = pickle.load(stream)
    queries = [c for c in queries if folds[str(c['meta']['data_id'])] == fold]
    assert not set(outer_ids) & {c['meta']['data_id'] for c in queries}
    rows = []
    for c in queries:
        choice = policies['/'.join(c['group'])]
        p = c['base'] if not choice['indices'] else fusion.predict(
            c | dict(bank=c['bank'][choice['indices']]), choice['model'])
        predictions = dict(Base=c['base'], Shared=fusion.predict(c, shared_full), Specific=p)
        for name, prediction in predictions.items():
            rows.append({k: c['meta'][k] for k in ['task', 'block', 'task_id', 'dataset', 'family', 'rho']} |
                        dict(data_id=c['meta']['data_id'], fold=fold, variant=VARIANT, method=name,
                             deployed_specific_alternatives=choice['deployed_alternatives'],
                             **metrics(c['y'], prediction, task, c['meta']['scale'])))
    pd.DataFrame(rows).to_csv(dest / 'detail.csv', index=False)
    atomic_json(dest / 'complete.json', dict(status='COMPLETE', conditions=len(rows)//3,
                                            outer_test_data_ids=sorted({c['meta']['data_id'] for c in queries})))
    print('FOLD_COMPLETE', task, fold, len(rows)//3, flush=True)


def worker(shard):
    for task, fold in [(t, f) for t in ['classification', 'regression'] for f in range(5)][shard::8]:
        try:
            run_fold(task, fold)
        except Exception:
            error = traceback.format_exc()
            atomic_json(OUT / 'folds' / task / str(fold) / 'error.json', dict(error=error))
            print(error, flush=True)


def report():
    paths = list((OUT / 'folds').glob('*/*/detail.csv'))
    if not paths:
        return
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df.to_csv(OUT / 'detail.csv', index=False)
    for task in ['classification', 'regression']:
        d = df[df.task == task]
        if d.empty:
            continue
        keys = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        avg = d.groupby(['task', 'block', 'method'], as_index=False).agg(
            datasets=('data_id', 'nunique'), conditions=('task_id', 'size'), **{k: (k, 'mean') for k in keys})
        avg.to_csv(OUT / f'averages_{task}.csv', index=False)
        print(avg.to_string(index=False), flush=True)


def launch():
    (OUT / 'logs').mkdir(exist_ok=True)
    workers = []
    for gpu in range(8):
        with (OUT / 'logs' / f'gpu{gpu}.log').open('a') as log:
            p = subprocess.Popen([sys.executable, '-u', __file__, 'worker', '--shard', str(gpu)],
                                 env=os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu)},
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        workers.append(dict(gpu=gpu, pid=p.pid))
    atomic_json(OUT / 'workers.json', workers)
    print(workers, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['declare', 'prepare', 'launch', 'worker', 'report'])
    parser.add_argument('--shard', type=int, default=0)
    args = parser.parse_args()
    declare()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'launch':
        launch()
    elif args.phase == 'worker':
        worker(args.shard)
    elif args.phase == 'report':
        report()
