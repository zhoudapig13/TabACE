"""Class-covered TRAIN holdout for Specific above an immutable Shared.

Only the source calibration population changes. Its size is the old holdout
size, and all 72 tasks are retained. Models/priors exclude the new held rows.
The final query data, frozen Base, Shared, and old context banks stay fixed.
No query label or score chooses a calibration split or deployment recipe.
"""
import argparse
import hashlib
import json
import os
import time
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name, '1')
import numpy as np
import pandas as pd
import full_pool_contexts_v34 as full
import tabace_context_adaptation_v29 as generic
import weighted_gate_v62 as parent
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor, normalize, metrics

OUT = RUN / 'results/tabace_balanced_clean_calibration_v71'
BUDGETS = [1024, 4096, 8192]
POWERS = [0., .25, .5, .75, 1.]
TEMPERATURES = [.7, 1., 1.4, 2.]
BLENDS = [.1, .25, .5, .75, 1.]


def cohort():
    return full.reference.frozen.cohort('classification', 'clean')


def partition(meta, pool):
    path = OUT / 'partitions' / f'{meta["task_id"]}.npz'
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    old = full.validation.validation_split('classification', 'five', meta, pool)
    y = pool['y'].astype(int)
    k = int(y.max()) + 1
    count = np.bincount(y, minlength=k)
    # Keep at least half of each class for contexts, and singleton classes
    # entirely for contexts. Record unavoidable missing calibration classes.
    cap = count // 2
    n = len(old['y_validation'])
    assert cap.sum() >= n
    target = np.bincount(y[:256], minlength=k).astype(float)
    target /= target.sum()
    allocation = ((cap > 0) & (target > 0)).astype(int)
    assert allocation.sum() <= n
    for _ in range(n - int(allocation.sum())):
        eligible = allocation < cap
        deficit = n * target - allocation
        deficit[~eligible] = -np.inf
        chosen = int(np.argmax(deficit))
        assert eligible[chosen]
        allocation[chosen] += 1
    rng = np.random.default_rng(np.random.SeedSequence([0, int(meta['task_id']), 7100]))
    held = []
    for label, amount in enumerate(allocation):
        if not amount:
            continue
        positions = np.flatnonzero(y == label)
        tail = positions[positions >= 256]
        core = positions[positions < 256]
        rng.shuffle(tail)
        rng.shuffle(core)
        held.extend(np.r_[tail, core][:amount].tolist())
    held = np.sort(np.asarray(held, dtype=int))
    train = np.setdiff1d(np.arange(len(y)), held)
    ids = pool['source_ids']
    assert len(held) == n and len(np.unique(held)) == n
    assert not np.intersect1d(ids[held], ids[train]).size
    assert not np.intersect1d(ids[held], pool['query_ids']).size
    assert np.array_equal(np.bincount(y[held], minlength=k), allocation)
    core_positions = np.intersect1d(np.arange(min(256, len(y))), train)
    assert len(core_positions) > 0
    result = dict(train_positions=train, held_positions=held,
                  X_validation=pool['X'][held], y_validation=pool['y'][held],
                  source_original_ids=ids[train], validation_original_ids=ids[held],
                  unheld_core_positions=core_positions)
    full.reference.scaling.npwrite(path, **result)
    atomic_json(path.with_suffix('.json'), dict(status='FROZEN_BEFORE_NEW_INFERENCE',
        task_id=meta['task_id'], dataset=meta['dataset'], seed=0,
        calibration_rows=n, old_calibration_rows=len(old['y_validation']),
        calibration_label_counts=allocation.tolist(), available_TRAIN_label_counts=count.tolist(),
        singleton_classes_unavailable_to_calibration=np.flatnonzero((count == 1) & (target > 0)).tolist(),
        old_missing_classes=np.flatnonzero((np.bincount(old['y_validation'].astype(int), minlength=k) == 0) & (target > 0)).tolist(),
        missing_classes=np.flatnonzero((allocation == 0) & (target > 0)).tolist(),
        original_core_rows_used_in_calibration=int(np.sum(held < 256)),
        unheld_core_label_counts=np.bincount(y[core_positions], minlength=k).tolist(),
        source_query_disjoint=True, held_rows_excluded_from_contexts_and_priors=True,
        replacement=False, calibration_selection_uses_TRAIN_labels_only=True,
        hashes={name: protocol.array_sha256(value) for name, value in result.items()}))
    return result


def construct(meta, pool, split, budget, base):
    X = pool['X'][split['train_positions']].copy()
    y = pool['y'][split['train_positions']].copy()
    Q = split['X_validation']
    n = min(budget, len(y))
    h = min(max(1, 3 * len(y) // 4), len(y)-1)
    payload = dict(X_pool=X, y_pool=y, base=base, history=(X[:h], y[:h]),
                   reference=(X[h:], y[h:]), actual_Base_context=n,
                   canonical=dict(X_source=X[:n], y_source=y[:n], X_target=Q))
    common, specific = full.contexts(payload, 'classification', 'clean', 'original')
    extra = generic.new_views(X[:n], y[:n], Q, 'classification', 'clean', 'original', meta)
    views = common + extra[:5] + specific + extra[5:]
    assert len(views) == 21
    return views, payload['canonical']


def source_worker(index, count, limit=0):
    import torch
    import warnings
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    warnings.filterwarnings('ignore')
    model = Predictor('classification', 'cuda:0')
    completed = 0
    for meta in cohort()[index::count]:
        tid = str(meta['task_id'])
        pool = full.reference.scaling.prepare('classification', 'five', meta)
        split = partition(meta, pool)
        k = int(pool['y'].max()) + 1
        for budget in BUDGETS:
            out = OUT / 'source' / str(budget) / f'{tid}.npz'
            if out.exists() and out.with_suffix('.json').exists():
                continue
            started = time.monotonic()
            X = pool['X'][split['train_positions']][:budget]
            y = pool['y'][split['train_positions']][:budget]
            Q = split['X_validation']
            base, _ = model(X, y, Q, members=1, random_state=0, n_classes=k)
            views, canonical = construct(meta, pool, split, budget, base)
            key = tuple(protocol.array_sha256(v) for v in [X, y, Q])
            cache = {key: base}
            predictions, hashes, names, rows, label_counts, aliases = [], [], [], [], [], []
            for name, A, b, T in views:
                A, b, T = np.asarray(A, dtype=np.float32), np.asarray(b), np.asarray(T, dtype=np.float32)
                assert 0 < len(b) <= budget
                key = tuple(protocol.array_sha256(v) for v in [A, b, T])
                alias = key in cache
                if alias:
                    p = cache[key]
                else:
                    p, _ = model(A, b, T, members=1, random_state=0, n_classes=k)
                cache[key] = p
                predictions.append(p); hashes.append(key); names.append(name); rows.append(len(b)); aliases.append(alias)
                label_counts.append(np.bincount(b.astype(int), minlength=k))
            assert np.isfinite(predictions).all()
            full.reference.scaling.npwrite(out, base=base, members=np.asarray(predictions),
                y_validation=split['y_validation'], validation_original_ids=split['validation_original_ids'],
                core_label_counts=np.bincount(pool['y'][split['unheld_core_positions']].astype(int), minlength=k),
                canonical_label_counts=np.bincount(y.astype(int), minlength=k), member_label_counts=np.asarray(label_counts))
            atomic_json(out.with_suffix('.json'), dict(status='SOURCE_BANK_COMPLETE', task='classification', block='clean',
                task_id=meta['task_id'], data_id=meta['data_id'], dataset=meta['dataset'], requested_context=budget,
                actual_context=len(y), family='original', rho=0, scale=1., seed=0,
                source_input_hashes={key: protocol.array_sha256(v) for key, v in canonical.items()},
                member_input_hashes=hashes, member_names=names, context_rows=rows, aliases=aliases,
                physical_Native1_calls=1+sum(not value for value in aliases),
                calibration_rows=len(Q), calibration_partition=str(OUT/'partitions'/f'{tid}.npz'),
                held_rows_excluded_from_all_contexts=True, same_per_branch_row_cap=True,
                query_labels_loaded=False, checkpoint_finetuned=False, seconds=time.monotonic()-started))
            print('BALANCED_CLEAN_SOURCE_COMPLETE', budget, tid, flush=True)
            completed += 1
            if limit and completed >= limit:
                return
    atomic_json(OUT/'source_workers'/f'{index}.json', dict(status='COMPLETE'))


def source_cells(budget):
    order = json.loads((full.OUT/'policies'/str(budget)/'classification/clean/generic_scale_selection.json').read_text())['permutation']
    cells = []
    for meta in cohort():
        path = OUT/'source'/str(budget)/f'{meta["task_id"]}.npz'
        record = json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            c = dict(base=z['base'], bank=z['members'][order], y=z['y_validation'].astype(int), ids=z['validation_original_ids'])
            core = z['core_label_counts'].astype(float)
            context = z['canonical_label_counts'].astype(float)
            member = z['member_label_counts'][order].astype(float)
        c.update(core_prior=(core+.5)/(core.sum()+.5*len(core)),
                 metric_core_prior=(core+.5)/(core.sum()+.5*len(core)),
                 context_prior=(context+.5)/(context.sum()+.5*len(context)),
                 member_priors=(member+.5)/(member.sum(1,keepdims=True)+.5*member.shape[1]),
                 meta=record, path=path, key=f'classification/clean/{meta["task_id"]}/original_0.npz',
                 order=order, task='classification', role='validation')
        cells.append(c)
    assert len(cells) == 72
    return cells


def parent_source(cells, budget):
    root = parent.OUT/'policies'/str(budget)/'classification/clean'
    policy = json.loads((root/'all_policies_frozen.json').read_text())
    datasets = sorted({int(c['meta']['data_id']) for c in cells})
    folds = {d: i % 4 for i, d in enumerate(datasets)}
    models = {name: {f: json.loads((root/name/f'fold_{f}.json').read_text()) for f in range(4)}
              for name in ['Shared', 'Specific_original']}
    shared, specific = {}, {}
    for c in cells:
        dataset = int(c['meta']['data_id']); fold = folds[dataset]
        sh = models['Shared'][fold]; sp = models['Specific_original'][fold]
        assert dataset not in sh['train_datasets'] and dataset not in sp['train_datasets']
        shared[c['key']] = parent.apply(c, sh['model'], policy['Shared'], c['base'])
        specific[c['key']] = parent.apply(c, sp['model'], policy['Specific'], shared[c['key']])
    return shared, specific


def member_mean(c, power):
    logs = np.log(np.clip(c['bank'], 1e-12, 1.)) + power*np.log(c['core_prior'][None]/c['member_priors'])[:,None,:]
    return np.mean([normalize(np.exp(p-p.max(1,keepdims=True))) for p in logs], axis=0)


def candidate(c, previous, source, power, temperature):
    if source == 'member_mean':
        p = member_mean(c, power)
        return parent.transform(c, p, 0., temperature)
    p = previous if source == 'previous_specific' else c['base']
    return parent.transform(c, p, power, temperature)


def fit(budget):
    parent.configure()
    cells = source_cells(budget)
    anchors, previous = parent_source(cells, budget)
    initial, best_value = parent.prior.score(cells, anchors)
    best, trials = None, []
    for source in ['previous_specific', 'member_mean', 'canonical']:
        for power in POWERS:
            for temperature in TEMPERATURES:
                values = {c['key']: candidate(c, previous[c['key']], source, power, temperature) for c in cells}
                for blend in BLENDS:
                    mixed = {key: normalize((1-blend)*anchors[key]+blend*values[key]) for key in anchors}
                    scores, utility = parent.prior.score(cells, mixed)
                    eligible = scores['accuracy'] >= initial['accuracy']-1e-10 and scores['auc'] >= initial['auc']-1e-10
                    trials.append(dict(source=source, prior_power=power, temperature=temperature, blend=blend,
                                       source_scores=scores, utility=utility, eligible=eligible))
                    if eligible and utility > best_value+1e-10:
                        best_value = utility
                        best = dict(source=source, prior_power=power, temperature=temperature, blend=blend)
    root = OUT/'policies'/str(budget)/'classification/clean'
    atomic_json(root/'all_policies_frozen.json', dict(status='FROZEN_BEFORE_QUERY_BANKS_LOADED', Specific=best,
        source_baseline=initial, trials=trials, Shared_unchanged_from_v62=True,
        same_calibration_row_count=True, balanced_class_covered_TRAIN_calibration=True,
        priors_exclude_all_held_rows=True, parent_source_models_exclude_entire_target_dataset=True,
        query_labels_used_for_fit_or_selection=False, configured_estimator_upper_bound=22,
        clean_TRAIN_supervision=True, seed=0))
    rows = []
    import member_priors_v63 as priors
    for c in parent.load('query', budget):
        original = parent.OUT/'predictions'/str(budget)/c['key']
        with np.load(original) as z:
            base, shared, old, truth = z['Base'], z['Shared'], z['Specific'], z['y_target']
        exact = json.loads((priors.OUT/'query'/str(budget)/f'{c["meta"]["task_id"]}.json').read_text())
        hist = np.asarray(exact['member_label_counts'], dtype=float)[c['order']]
        c['member_priors'] = (hist+.5)/(hist.sum(1,keepdims=True)+.5*hist.shape[1])
        specific = shared
        if best is not None:
            value = candidate(c, old, best['source'], best['prior_power'], best['temperature'])
            a = best['blend']; specific = normalize((1-a)*shared+a*value)
        out = OUT/'predictions'/str(budget)/c['key']
        full.reference.scaling.npwrite(out, Base=base, Shared=shared, Specific=specific, y_target=truth)
        record = json.loads(original.with_suffix('.json').read_text())
        record.update(adaptation_version='v71', Shared_bitwise_unchanged_from_v62=True,
            balanced_class_covered_TRAIN_calibration=True, same_calibration_row_count=True,
            query_labels_used_for_fit_or_selection=False,
            methods={name: metrics(truth, p, 'classification', 1.) for name, p in [('Base',base),('Shared',shared),('Specific',specific)]})
        atomic_json(out.with_suffix('.json'), record)
        for name, values in record['methods'].items():
            rows.append(dict(context=budget, block='clean', method=name, task_id=c['meta']['task_id'], **values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv', index=False)
    atomic_json(root/'completion.json', dict(status='COMPLETE', conditions=72, Shared_unchanged=True))
    print('BALANCED_CLEAN_FIT_COMPLETE', budget, best, flush=True)


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json', dict(status='DECLARED_BEFORE_SOURCE_INFERENCE', seed=0, budgets=BUDGETS,
        full_goal_checks=48, changed_factor='TRAIN calibration sample composition only; matched old holdout size',
        motivation='12 old calibration datasets miss some classes; reweighting cannot recover absent positive rows',
        sampling='without replacement, toward protected TRAIN core prior, cover every nonsingleton class, retain half per class in TRAIN',
        source_TFM_contexts='rebuild existing21 views with new held rows excluded; same per-context row caps',
        target_priors='source original core256 minus held rows; query original core256; no query labels',
        candidates=['previous_specific', 'member_mean', 'canonical'], powers=POWERS,
        temperatures=TEMPERATURES, blends=BLENDS, additional_native_ensemble=False,
        parent_source='v62 whole-dataset OOF gate models; target dataset excluded',
        Shared_query_predictions_frozen_from_v62=True, Clean256_frozen_from_v62=True,
        regression_frozen_from_v59=True, classification_five_frozen_from_v51=True,
        final_query_data_and_Base_unchanged=True, checkpoint_finetuned=False,
        clean_TRAIN_calibration_supervision=True, query_labels_used_for_fit_or_selection=False,
        query_feedback_used_for_development_design=True, development_only=True,
        original_paper_replication=False, independent_test_claim=False, Markdown_written=False))
    (OUT/'paper_targets.json').write_text((parent.OUT/'paper_targets.json').read_text())
    for b in [256,1024,4096,8192]:
        for task in ['classification','regression']:
            for block in ['clean','five','seven']:
                if task=='classification' and block=='clean' and b!=256:
                    continue
                origin = (parent.OUT if block=='clean' else RUN/'results/tabace_probability_head_v51'
                          if block=='five' else RUN/'results/tabace_balanced_seven_v66') if task=='classification' else RUN/'results/tabace_clean_calibration_v59'
                for name in ['policies','predictions']:
                    dest = OUT/name/str(b)/task/block
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.symlink_to(origin/name/str(b)/task/block, target_is_directory=True)


def check_partitions():
    rows = []
    for meta in cohort():
        pool = full.reference.scaling.prepare('classification', 'five', meta)
        split = partition(meta, pool)
        assert len(split['y_validation']) == len(np.unique(split['validation_original_ids']))
        rows.append(json.loads((OUT/'partitions'/f'{meta["task_id"]}.json').read_text()))
    atomic_json(OUT/'partition_audit.json', dict(status='PASS', datasets=len(rows),
        old_tasks_with_missing_classes=sum(bool(row['old_missing_classes']) for row in rows),
        new_tasks_with_missing_classes=sum(bool(row['missing_classes']) for row in rows),
        same_calibration_row_counts=all(row['calibration_rows']==row['old_calibration_rows'] for row in rows),
        all_original_ids_unique_and_source_query_disjoint=True,
        held_rows_removed_from_source_priors=True, rows=rows, query_labels_used=False))
    print('BALANCED_CLEAN_PARTITIONS_PASS', len(rows), sum(bool(r['missing_classes']) for r in rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['plan','partitions','source','fit'])
    p.add_argument('--index', type=int, default=0); p.add_argument('--count', type=int, default=1)
    p.add_argument('--limit', type=int, default=0); p.add_argument('--budget', type=int)
    a = p.parse_args()
    if a.phase == 'plan': plan()
    elif a.phase == 'partitions': check_partitions()
    elif a.phase == 'source': source_worker(a.index, a.count, a.limit)
    else: fit(a.budget)
