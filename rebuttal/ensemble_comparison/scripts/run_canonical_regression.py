"""A separate input-access control: six bootstraps of Base's exact 256 rows.

The frozen shift generator and complete-pool experiments are unchanged. Query
labels enter metrics only, after predictions and IQR selection are complete.
"""
import argparse
import hashlib
import json
import logging
import time
import traceback
import warnings

import numpy as np
import torch

from prepare_data import RUN, atomic_json
from run_tabicl import Predictor, metrics
from shared_reference import load_condition, specifications
from seven_shifts import FAMILIES


def jobs():
    conditions = [('covariate', 0)] + [(f, r) for f in FAMILIES for r in [20, 50, 80, 100]]
    return [(m, f, r) for m in specifications('seven_calibrated_v2') for f, r in conditions]


def fingerprint(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--replace-fraction', type=float, default=1.)
    args = parser.parse_args()
    assert args.replace_fraction in [1., .1]
    result_root = 'canonical_input' if args.replace_fraction == 1. else 'canonical_partial_010'
    torch.set_num_threads(1)
    logging.disable(logging.INFO)
    warnings.filterwarnings('ignore')
    predictor = Predictor('regression', 'cuda:0')
    work = jobs()[args.shard::args.shards]
    if args.limit:
        work = work[:args.limit]
    failures = 0
    for meta, family, rho in work:
        out = RUN/'results'/result_root/'tabicl/seven_calibrated_v2/regression'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():
            continue
        try:
            data, info = load_condition(meta, family, rho)
            X, y, Q = (data[k] for k in ['X_source', 'y_source', 'X_target'])
            raw = RUN/'results/tabicl/seven_calibrated_v2/regression'/str(meta['task_id'])/out.name
            tick = time.perf_counter()
            # This API receives source rows and unlabelled queries only.
            bank, iqrs, hashes = [], [], []
            for i in range(6):
                rng = np.random.default_rng(1009*(i+1))
                if args.replace_fraction == 1.:
                    rows = rng.choice(len(y), size=len(y), replace=True)
                else:
                    rows = np.arange(len(y))
                    positions = rng.choice(len(y), size=int(np.ceil(.1*len(y))), replace=False)
                    rows[positions] = rng.choice(len(y),size=len(positions),replace=True)
                pred, iqr = predictor(X[rows], y[rows], Q, quantiles=True)
                bank.append(pred)
                iqrs.append(iqr)
                hashes.append(fingerprint(rows))
            bank, iqrs = np.stack(bank), np.stack(iqrs)
            selected = int(np.argmin(iqrs.mean(axis=1)))
            assert np.isfinite(bank).all() and np.isfinite(iqrs).all()
            with np.load(raw) as f:
                base = f['prediction__base_protocol_check']
                native = f['prediction__native_6']
                truth = f['y_target']
            assert np.array_equal(truth, data['y_target'])
            scale = max(float(truth.std()), 1e-12)
            predictions = {'base_protocol_check': base, 'native_6': native,
                           'bootstrap_mean_6': bank.mean(0),
                           'oligoicp_iqr_6_keep1_adapted': bank[selected]}
            arrays = {'bootstrap_bank': bank, 'bootstrap_iqr': iqrs, 'y_target': truth,
                      'source_y_mean': np.asarray(float(y.mean())),
                      'source_y_std': np.asarray(max(float(y.std()), 1e-12))}
            arrays.update({'prediction__'+k: v for k, v in predictions.items()})
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, **arrays)
            tmp.replace(out)
            atomic_json(out.with_suffix('.json'), info | {
                'status': 'COMPLETE', 'seed': 0, 'family': family, 'rho': rho,
                'access_regime': 'Base canonical rows only; no history/reference pool',
                'context_rows': len(y), 'candidate_count': 6,
                'resampling_fraction':args.replace_fraction,
                'resampling_policy': 'ordinary bootstrap' if args.replace_fraction == 1. else 'fixed 10% source-row replacement',
                'original_accessible_pool_rows': len(data.get('X_pool', X)),
                'source_X_sha256': fingerprint(X), 'source_y_sha256': fingerprint(y),
                'query_X_sha256': fingerprint(Q), 'bootstrap_indices_sha256': hashes,
                'base_native_reused_from': str(raw.relative_to(RUN)),
                'target_labels_used_for_inference_or_selection': False,
                'Shared_run': False, 'paper_scores_used_for_selection': False,
                'selected_iqr_member': selected, 'seconds': time.perf_counter()-tick,
                'methods': {k: metrics(truth, v, 'regression', scale) for k, v in predictions.items()}})
            print('CANONICAL_DONE', meta['task_id'], family, rho, flush=True)
        except Exception as exc:
            failures += 1
            atomic_json(RUN/'results/inference_failures'/f"canonical_reg_{meta['task_id']}_{family}_{rho}.json",
                        {'error': repr(exc), 'traceback': traceback.format_exc()})
            print('CANONICAL_FAILED', repr(exc), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
