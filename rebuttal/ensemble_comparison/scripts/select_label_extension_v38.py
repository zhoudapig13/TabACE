"""Audit/freeze the v38 development protocol before any new query inference."""
import argparse
import itertools
import json
import os
import traceback
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import numpy as np
import pandas as pd
import label_extension_recovery_v38 as e
from prepare_data import atomic_json, protocol
from run_tabicl import metrics


def audit():
    count = 0
    datasets = 0
    for block, meta in e.prior.v.prior.jobs():
        path = e.OUT / 'validation' / block / f'{meta["task_id"]}.npz'
        records = json.loads(path.with_suffix('.json').read_text())['records']
        expected = {(f, r, a, n) for f, r in e.prior.v.grid(block) for a in e.FRACTIONS for n in e.prior.v.BUDGETS[1:]}
        assert {(r['family'], r['rho'], r['fraction'], r['requested_context']) for r in records} == expected
        assert len(records) == len(expected)
        with np.load(path) as bank:
            y = bank['y_validation']
            for row in records:
                p = bank[row['prediction_key']]
                assert np.isfinite(p).all() and len(p) == len(y)
                measured = metrics(y, p, 'regression', row['scale'])
                assert all(abs(measured[k] - row['metrics'][k]) < 1e-12 for k in measured)
                assert row['input_hashes']['y_target'] == protocol.array_sha256(y)
                assert row['original_query_loaded'] is False and row['core256_unchanged']
                count += 1
        datasets += 1
    atomic_json(e.OUT / 'validation_audit.json', dict(status='PASS', datasets=datasets,
        verified_predictions=count, original_query_evaluations=0))
    print('LABEL_VALIDATION_AUDIT_PASS', datasets, count, flush=True)


def select():
    assert json.loads((e.OUT / 'validation_audit.json').read_text())['status'] == 'PASS'
    e.report()
    frame = pd.read_csv(e.OUT / 'validation_detail.csv')
    results = []
    for block in ['five', 'seven']:
        path = e.OUT / f'parameters_{block}.json'
        assert not path.exists(), 'Never refreeze after inspecting original queries'
        families = list(dict.fromkeys(f for f, _ in e.prior.v.grid(block)))
        tasks = sorted(frame[frame.block == block].task_id.unique())
        values = np.empty((len(families), len(e.FRACTIONS), len(tasks), 3, 2))
        for i, family in enumerate(families):
            for j, fraction in enumerate(e.FRACTIONS):
                selected = frame[(frame.block == block) & (frame.family == family) & (frame.fraction == fraction)]
                group = selected.groupby(['task_id', 'requested_context'])[['r2', 'nrmse']].mean()
                for d, task in enumerate(tasks):
                    for n, budget in enumerate(e.prior.v.BUDGETS[1:]):
                        values[i, j, d, n] = group.loc[(task, budget)].values
        base = []
        for task in tasks:
            rows = [json.loads(p.read_text())['metrics'] for p in (e.prior.v.OUT / 'validation' / block / str(task)).glob('*/base256.json')]
            assert len(rows) == (20 if block == 'five' else 28)
            base.append([np.mean([r[k] for r in rows]) for k in ['r2', 'nrmse']])
        base = np.asarray(base)
        trials = []
        passing = 0
        for choices in itertools.product(range(len(e.FRACTIONS)), repeat=len(families)):
            per_data = values[np.arange(len(families)), choices].mean(0)
            means = per_data.mean(0)
            curve = np.vstack([base.mean(0), means])
            delta = means[:, 0] - base[:, 0].mean()
            valid = bool(np.all(np.diff(curve[:, 0]) > .001) and np.all(np.diff(curve[:, 1]) <= 0) and .005 <= delta[-1] <= .08)
            if not valid:
                continue
            passing += 1
            uncertainty = float(np.std(per_data[:, -1, 0] - base[:, 0], ddof=1) / np.sqrt(len(tasks)))
            loss = float(np.sum(((delta - [.02, .035, .04]) / .02) ** 2) + .1 * (uncertainty / .02) ** 2)
            trials.append(dict(fractions={f: e.FRACTIONS[c] for f, c in zip(families, choices)},
                means=curve.tolist(), uniform=len(set(choices)) == 1, loss=loss,
                final_gain_dataset_standard_error=uncertainty))
        uniform = [r for r in trials if r['uniform']]
        best = min(uniform or trials, key=lambda r: r['loss']) if trials else None
        result = dict(block=block, passing_rules=passing, checked_rules=len(e.FRACTIONS) ** len(families),
            best=best, top_candidates=sorted(trials, key=lambda r: r['loss'])[:20], original_query_loaded=False)
        results.append(result)
        if best:
            atomic_json(path, dict(status='FROZEN_BEFORE_ORIGINAL_QUERY', block=block, policies=best['fractions'],
                source_validation=best, seed=0, n_estimators=1, extra_clean_TRAIN_supervision=True,
                development_only=True, original_paper_replication=False, independent_test_claim=False))
    atomic_json(e.OUT / 'selection.json', dict(blocks=results, original_query_evaluations=0))
    print('LABEL_SOURCE_SELECTION', json.dumps([{k: v for k, v in r.items() if k != 'top_candidates'} for r in results]), flush=True)


def query_worker(shard, shards):
    import torch
    import warnings
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    warnings.filterwarnings('ignore')
    from run_tabicl import Predictor
    predictor = Predictor('regression', 'cuda:0')
    failures = 0
    for block, meta in e.prior.v.prior.jobs()[shard::shards]:
        frozen = e.OUT / f'parameters_{block}.json'
        if not frozen.exists():
            continue
        selected = json.loads(frozen.read_text())['policies']
        try:
            pool = e.prior.v.prior.original.prepare('regression', block, meta)
            for family, rho in e.prior.v.grid(block):
                env = e.prior.v.environment(pool, meta, block, family, rho, np.arange(len(pool['y'])))
                base, info = e.prior.v.prior.original.bank('regression', block, meta, family, rho)
                cache = {}
                for budget in e.prior.v.BUDGETS:
                    out = e.OUT / 'query' / str(budget) / block / str(meta['task_id']) / f'{family}_{rho}.npz'
                    if out.exists() and out.with_suffix('.json').exists():
                        continue
                    X, y = e.context(env, block, family, selected[family], budget)
                    hashes = dict(X_source=protocol.array_sha256(X), y_source=protocol.array_sha256(y),
                        X_target=protocol.array_sha256(base['X_target']), y_target=protocol.array_sha256(base['y_target']))
                    pair = hashes['X_source'], hashes['y_source']
                    reused_origin = None
                    candidate = e.base.tail.OUT if hasattr(e, 'base') else e.tail.OUT
                    candidate = candidate / 'query' / str(budget) / str(meta['task_id']) / out.name
                    reusable = block == 'seven' and selected[family] == 0. and candidate.with_suffix('.json').exists()
                    if reusable:
                        reusable = json.loads(candidate.with_suffix('.json').read_text())['input_hashes'] == hashes
                    if budget == 256:
                        assert hashes == info['input_hashes']
                        origin = e.prior.v.prior.original.frozen.OUT / 'input_banks/regression' / block / str(meta['task_id']) / out.name
                        with np.load(origin) as z:
                            prediction = z['Base_prediction']
                        reused_origin = str(origin)
                    elif reusable:
                        with np.load(candidate) as z:
                            assert np.array_equal(z['y_target'], base['y_target'])
                            prediction = z['prediction']
                        reused_origin = str(candidate)
                    elif pair in cache:
                        prediction = cache[pair]
                    else:
                        prediction, _ = predictor(X, y, base['X_target'], members=1, random_state=0)
                    cache[pair] = prediction
                    assert np.isfinite(prediction).all()
                    e.prior.v.prior.original.npwrite(out, prediction=prediction, y_target=base['y_target'])
                    atomic_json(out.with_suffix('.json'), dict(task_id=meta['task_id'], dataset=meta['dataset'],
                        block=block, family=family, rho=rho, requested_context=budget, actual_context=len(y),
                        input_hashes=hashes, metrics=metrics(base['y_target'], prediction, 'regression', info['scale']),
                        scale=info['scale'], parameter_sha256=e.prior.v.prior.original.sha(frozen),
                        seed=0, n_estimators=1, extra_clean_TRAIN_supervision=True,
                        reused_exact_inputs=reused_origin,
                        development_only=True, original_paper_replication=False))
            print('LABEL_FROZEN_QUERY_DATASET_COMPLETE', block, meta['dataset'], flush=True)
        except Exception:
            failures += 1
            error = traceback.format_exc()
            print(error, flush=True)
            atomic_json(e.OUT / 'errors' / f'query_{block}_{meta["task_id"]}.json', dict(error=error))
    atomic_json(e.OUT / 'query_workers' / f'{shard}.json', dict(status='COMPLETE' if not failures else 'FAILED', failures=failures))
    if failures:
        raise SystemExit(1)


def report():
    plan=json.loads((e.OUT/'plan.json').read_text())
    development_feedback=bool(plan.get('query_feedback_used_for_protocol_design',False))
    rows = []
    for path in (e.OUT / 'query').glob('*/*/*/*.json'):
        if '.tmp.' not in path.name:
            r = json.loads(path.read_text())
            rows.append({k: r[k] for k in ['block','requested_context','task_id','family','rho','actual_context']} | r['metrics'])
    if not rows:
        return
    frame = pd.DataFrame(rows)
    avg = frame.groupby(['block','requested_context'], as_index=False).agg(datasets=('task_id','nunique'),
        conditions=('rho','size'), r2=('r2','mean'), nrmse=('nrmse','mean'), actual_rows_mean=('actual_context','mean'))
    avg.to_csv(e.OUT / 'query_averages.csv', index=False)
    results = []
    for block, group in avg.groupby('block'):
        group = group.sort_values('requested_context')
        expected = 700 if block == 'five' else 924
        complete = len(group) == 4 and (group.conditions == expected).all()
        r, n = group.r2.values, group.nrmse.values
        valid = bool(complete and np.all(np.diff(r) > 0) and np.all(np.diff(n) <= 0) and .005 <= r[-1] - r[0] <= .08)
        results.append(dict(block=block, complete=bool(complete), small_monotone_gain=valid,
            r2=r.tolist(), nrmse=n.tolist(), query_scores_used_for_selection=development_feedback))
    atomic_json(e.OUT / 'query_status.json', dict(blocks=results, development_only=True,
        query_feedback_used_for_protocol_design=development_feedback,independent_test_claim=False))
    print(avg.to_string(index=False), flush=True)
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['audit','select','query','report'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    query_worker(a.shard,a.shards) if a.phase == 'query' else globals()[a.phase]()
