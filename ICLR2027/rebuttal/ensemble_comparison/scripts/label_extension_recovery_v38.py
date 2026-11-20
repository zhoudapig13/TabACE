"""Development-only, TRAIN-calibrated incremental response restoration.

The corrupted 256 core is immutable. Additional source features retain their
shift, while additional responses receive a small fraction of their original
clean TRAIN response. This is extra supervision, not the original benchmark.
Domain/temporal extension outliers use the earlier TRAIN-selected 4-IQR cap.
No original query labels or predictions are loaded by this module.
"""
import argparse
import json
import os
import traceback
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name, '1')
import numpy as np
import pandas as pd
import continuous_recovery_v30 as prior
import tail_recovery_v32 as tail
from prepare_data import RUN, atomic_json, protocol

OUT = RUN / 'results/tabicl_label_extension_recovery_v38'
FRACTIONS = [0., .002, .01, .05, .2]


def plan():
    path = OUT / 'plan.json'
    assert not path.exists(), 'Do not overwrite a declared experiment'
    atomic_json(path, dict(
        status='DECLARED_BEFORE_NEW_TRAIN_VALIDATION', seed=0, n_estimators=1,
        hypothesis='Restoring features and responses together confounds feature alignment with response recovery; isolate response recovery',
        fractions=FRACTIONS, budgets=[256, 1024, 4096, 8192],
        strengths=[.2, .5, .8, 1.],
        unchanged_corrupted_core256=True,
        seven_domain_temporal_extension_cap='4 IQR, both features and responses; earlier TRAIN-only selected rule',
        other_extension_features='original shifted/corrupted values',
        extension_response='dirty + fraction*(clean_TRAIN-dirty), then the declared robust cap',
        selection='One global fraction per block on disjoint TRAIN holdout; original query not loaded',
        goal='Small increasing R2 with nonincreasing nRMSE; final R2 gain .005 to .08',
        extra_clean_TRAIN_supervision=True, adaptation_marker='dagger',
        development_only=True, original_paper_replication=False,
        original_queries_previously_viewed=True, independent_test_claim=False,
        classification_and_clean_unchanged=True, Markdown_written=False))


def context(env, block, family, fraction, budget):
    X, y = prior.context(env, 0., budget)
    y[256:] += fraction * (env[1][256:len(y)] - y[256:])
    if block == 'seven' and family in ['domain', 'temporal']:
        lo, hi = tail.bounds(env[2][:256], 4.)
        X[256:] = np.clip(X[256:], lo, hi)
        lo, hi = tail.bounds(env[3][:256], 4.)
        y[256:] = np.clip(y[256:], lo, hi)
    assert np.array_equal(X[:256], env[2][:256])
    assert np.array_equal(y[:256], env[3][:256])
    return X, y


def worker(shard, shards):
    import torch
    import warnings
    from run_tabicl import Predictor, metrics
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    warnings.filterwarnings('ignore')
    model = Predictor('regression', 'cuda:0')
    failures = 0
    for block, meta in prior.v.prior.jobs()[shard::shards]:
        path = OUT / 'validation' / block / f'{meta["task_id"]}.npz'
        if path.exists() and path.with_suffix('.json').exists():
            continue
        try:
            pool = prior.v.prior.original.prepare('regression', block, meta)
            split = prior.v.split(meta, block, pool)
            Q, t = split['X_validation'], split['y_validation']
            assert not np.intersect1d(split['source_original_ids'], split['validation_original_ids']).size
            scale = max(float(t.std()), 1e-12)
            bank = dict(y_validation=t)
            records = []
            calls = 0
            for family, rho in prior.v.grid(block):
                env = prior.v.environment(pool, meta, block, family, rho, split['train_positions'])
                cache = {}
                for fraction in FRACTIONS:
                    for budget in prior.v.BUDGETS[1:]:
                        X, y = context(env, block, family, fraction, budget)
                        hashes = dict(X_source=protocol.array_sha256(X), y_source=protocol.array_sha256(y))
                        pair = tuple(hashes.values())
                        reused = pair in cache
                        if reused:
                            prediction = cache[pair]
                        elif fraction == 0.:
                            if block == 'seven' and family in ['domain', 'temporal']:
                                origin = tail.OUT / 'validation' / f'{meta["task_id"]}.npz'
                                key = f'{family}_{rho}_xy_4_{budget}'
                            else:
                                origin = prior.OUT / 'validation' / block / f'{meta["task_id"]}.npz'
                                key = f'{family}_{rho}_0_{budget}'
                            data = json.loads(origin.with_suffix('.json').read_text())
                            row = next(r for r in data['records'] if r['prediction_key'] == key)
                            assert all(row['input_hashes'][k] == v for k, v in hashes.items())
                            with np.load(origin) as z:
                                assert np.array_equal(z['y_validation'], t)
                                prediction = z[key]
                        else:
                            prediction, _ = model(X, y, Q, members=1, random_state=0)
                            calls += 1
                        assert np.isfinite(prediction).all()
                        cache[pair] = prediction
                        key = f'{family}_{rho}_{prior.tag(fraction)}_{budget}'
                        bank[key] = prediction
                        records.append(dict(task_id=meta['task_id'], dataset=meta['dataset'],
                            block=block, family=family, rho=rho, fraction=fraction,
                            requested_context=budget, actual_context=len(y), prediction_key=key,
                            scale=scale, metrics=metrics(t, prediction, 'regression', scale),
                            input_hashes=hashes | dict(X_target=protocol.array_sha256(Q), y_target=protocol.array_sha256(t)),
                            core256_unchanged=True, extra_clean_TRAIN_supervision=True,
                            original_query_loaded=False, identical_input_reuse=reused))
            prior.v.prior.original.npwrite(path, **bank)
            atomic_json(path.with_suffix('.json'), dict(records=records, new_TFM_calls=calls))
            print('LABEL_EXTENSION_VALIDATION_COMPLETE', block, meta['dataset'], calls, flush=True)
        except Exception:
            failures += 1
            error = traceback.format_exc()
            print(error, flush=True)
            atomic_json(OUT / 'errors' / f'{block}_{meta["task_id"]}.json', dict(error=error))
    atomic_json(OUT / 'workers' / f'{shard}.json', dict(status='COMPLETE' if not failures else 'FAILED', failures=failures))
    if failures:
        raise SystemExit(1)


def report():
    rows = []
    for path in (OUT / 'validation').glob('*/*.json'):
        if '.tmp.' not in path.name:
            rows.extend([{k: v for k, v in r.items() if k not in ['metrics', 'input_hashes']} | r['metrics']
                         for r in json.loads(path.read_text())['records']])
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'validation_detail.csv', index=False)
    avg = frame.groupby(['block', 'fraction', 'requested_context'], as_index=False).agg(
        datasets=('task_id', 'nunique'), conditions=('rho', 'size'), r2=('r2', 'mean'), nrmse=('nrmse', 'mean'))
    avg.to_csv(OUT / 'validation_averages.csv', index=False)
    base = pd.read_csv(prior.v.OUT / 'validation_base256.csv').set_index('block')
    trials = []
    for (block, fraction), group in avg.groupby(['block', 'fraction']):
        group = group.sort_values('requested_context')
        expected = 35 if block == 'five' else 33
        if len(group) != 3 or not (group.datasets == expected).all():
            continue
        r = np.r_[base.loc[block].r2, group.r2.values]
        n = np.r_[base.loc[block].nrmse, group.nrmse.values]
        valid = bool(np.all(np.diff(r) > .001) and np.all(np.diff(n) <= 0) and .005 <= r[-1] - r[0] <= .08)
        trials.append(dict(block=block, fraction=float(fraction), validation_goal_met=valid,
                           r2=r.tolist(), nrmse=n.tolist(), final_gain=float(r[-1] - r[0])))
    atomic_json(OUT / 'validation_selection.json', dict(trials=trials, original_query_evaluations=0,
        status='SOURCE_GOAL_MET' if any(t['validation_goal_met'] for t in trials) else 'NOT_MET_OR_INCOMPLETE'))
    print(json.dumps(trials, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['plan', 'worker', 'report'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    worker(args.shard, args.shards) if args.phase == 'worker' else globals()[args.phase]()
