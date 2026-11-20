"""Oligo-specific source validation; original outputs remain unchanged.

This is an additional development variant designed after observing the internal
benchmark. Its coefficient is fit exclusively on disjoint SOURCE validation.
"""
import argparse
import json
import logging
import time
import warnings

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from prepare_data import atomic_json, protocol
from tabpfn_official_benchmark_v2 import OUT, tasks, conditions, frozen_input
from tabpfn_official_methods_v2 import source_clean, source_validation_condition
from tabpfn_oligoicp_v2 import grouping, candidate_spec, infer
from run_tabpfn import model, aligned
from run_tabicl import metrics


def read(path):
    return json.loads(path.read_text())


def coefficient(cells, which):
    cross = 0.; square = 0.
    for i, cell in enumerate(cells):
        weight = .25 if i == 0 else .75 / (len(cells) - 1)
        pred = cell[which]; delta = cell['auxiliary'] - pred
        scale = max(float(cell['normalization_variance']), 1e-12)
        cross += weight * float(np.mean((pred - cell['truth']) * delta)) / scale
        square += weight * float(np.mean(delta ** 2)) / scale
    return float(np.clip(-cross / max(square, 1e-30), 0., .5))


def fit(track, meta):
    folder = OUT / 'oligo_specific_adapters' / track / str(meta['task_id'])
    policy_path = folder / 'policy.json'
    if policy_path.exists():
        return
    pristine_X, pristine_y = source_clean('regression', track, meta)
    common = OUT / 'source_adapters/regression' / track / str(meta['task_id'])
    with np.load(common / 'source_split.npz') as z:
        tr, val = z['train_relative_ids'], z['validation_relative_ids']
    assert set(tr).isdisjoint(val)
    args = read(common / 'policy.json')['ExtraTrees_parameters']
    inner_auxiliary = ExtraTreesRegressor(**args).fit(pristine_X[tr], pristine_y[tr])
    est = model('regression', 'cuda:0')
    from seven_shifts import FAMILIES
    families = protocol.FAMILIES if track == 'five' else FAMILIES
    grid = [('original' if track == 'five' else 'covariate', 0)] + [(f, r) for f in families for r in [50, 100]]
    cells = []
    folder.mkdir(parents=True, exist_ok=True)
    for family, rho in grid:
        path = folder / f'{family}_{rho}.npz'
        if not (path.exists() and path.with_suffix('.json').exists()):
            X, y, Q, truth = source_validation_condition('regression', track, meta,
                pristine_X[tr], pristine_y[tr], pristine_X[val], pristine_y[val], family, rho)
            labels, active = grouping(X)
            means, spreads, seeds = [], [], []
            tick = time.perf_counter()
            for member in range(60):
                rows, groups, seed = candidate_spec(meta['task_id'], track, member, labels, active)
                pred, iqr, _ = infer(est, X[rows], y[rows], Q, seed)
                means.append(pred); spreads.append(iqr); seeds.append(seed)
            means, spreads = np.stack(means), np.stack(spreads)
            selected = np.argsort(spreads.mean(1), kind='stable')[:6]
            auxiliary = inner_auxiliary.predict(Q)
            temp = path.with_suffix('.tmp.npz')
            np.savez_compressed(temp, means=means, iqrs=spreads, raw=means[selected].mean(0),
                full_mean=means.mean(0), auxiliary=auxiliary, truth=truth,
                normalization_variance=max(float(np.var(pristine_y[val])), 1e-12), selected=selected,
                seeds=np.asarray(seeds, dtype=np.uint32), source_group_labels=labels)
            temp.replace(path)
            atomic_json(path.with_suffix('.json'), {'task_id': meta['task_id'], 'track': track, 'family': family,
                'rho': rho, 'seed': 0, 'source_validation_only': True,
                'input_hashes': {key: protocol.array_sha256(value) for key, value in zip(['X', 'y', 'Q', 'truth'], [X, y, Q, truth])},
                'train_relative_ids': tr.tolist(), 'validation_relative_ids': val.tolist(),
                'benchmark_query_labels_used': False, 'candidate_count': 60, 'seconds': time.perf_counter() - tick})
        with np.load(path) as z:
            cells.append({key: z[key] for key in ['raw', 'full_mean', 'auxiliary', 'truth', 'normalization_variance']})
        print('OLIGO_SOURCE_VALIDATION_CELL', track, meta['dataset'], family, rho, flush=True)
    alphas = {key: coefficient(cells, key) for key in ['raw', 'full_mean']}
    losses = {}
    for key, alpha in alphas.items():
        losses[key] = []
        for cell in cells:
            pred = (1 - alpha) * cell[key] + alpha * cell['auxiliary']
            losses[key].append(float(np.mean((pred - cell['truth']) ** 2) / max(float(cell['normalization_variance']), 1e-12)))
    atomic_json(policy_path, {'status': 'FROZEN', 'seed': 0, 'outer_repetitions': 1,
        'alphas': alphas, 'validation_losses_after_adaptation': losses,
        'objective': 'normalized SOURCE-validation MSE, clean .25, shifted .75, equal family/strength weights',
        'alpha_constraint': [0, .5], 'validation_pool_size': 60, 'inner_split_shared_with_common_adapter': True,
        'validation_auxiliary_trained_on_inner_training_only': True,
        'development_variant_after_internal_results_observed': True,
        'benchmark_query_labels_used_to_fit_coefficient': False, 'historical_Shared_used_for_selection': False,
        'extra_clean_source_supervision': True, 'backbone_parameter_updates': False,
        'transfer_to400_is_budget_adaptation': True, 'original_outputs_preserved': True})
    print('OLIGO_SPECIFIC_COEFFICIENTS_FROZEN', track, meta['dataset'], json.dumps(alphas), flush=True)


def apply(track, meta, total):
    policy = read(OUT / 'oligo_specific_adapters' / track / str(meta['task_id']) / 'policy.json')
    common = OUT / 'source_adapters/regression' / track / str(meta['task_id'])
    auxiliary = joblib.load(common / 'extra_trees.joblib')
    grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
    for family, rho in grid:
        original = OUT / 'oligo_predictions' / track / str(meta['task_id']) / f'{family}_{rho}' / f'results_{total}.npz'
        path = OUT / 'oligo_specific_predictions' / str(total) / track / str(meta['task_id']) / f'{family}_{rho}.npz'
        if path.exists() and path.with_suffix('.json').exists():
            continue
        assert original.exists() and original.with_suffix('.json').exists()
        data, info, base = frozen_input('regression', track, meta, family, rho)
        source_meta = read(original.with_suffix('.json'))
        assert source_meta['input_hashes'] == {key: protocol.array_sha256(value) for key, value in data.items()}
        aux = aligned(auxiliary, data['X_target'], 'regression', None)
        with np.load(original) as z:
            assert np.array_equal(z['y_target'], data['y_target'])
            raw, full_mean = z['raw_prediction'], z['full_ensemble_mean']
            a, b = policy['alphas']['raw'], policy['alphas']['full_mean']
            frozen_column = track == 'five' or rho == 0
            if frozen_column:
                # User steering: preserve every existing Clean/five output.
                # Only the seven-shift regression column may change.
                specific = z['adapted_prediction'].copy()
                mean_specific = z['full_ensemble_mean_adapted'].copy()
                applied_alphas = {'raw': source_meta['adapter_alpha'], 'full_mean': source_meta['adapter_alpha']}
            else:
                specific = (1 - a) * raw + a * aux
                mean_specific = (1 - b) * full_mean + b * aux
                applied_alphas = policy['alphas']
        path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, adapted_prediction=specific, full_mean_adapted_prediction=mean_specific,
                            original_Base=base, y_target=data['y_target'])
        temp.replace(path)
        atomic_json(path.with_suffix('.json'), info | {'status': 'COMPLETE', 'seed': 0,
            'OligoICP_specific_metrics': metrics(data['y_target'], specific, 'regression', info['scale']),
            'FullMean_specific_metrics': metrics(data['y_target'], mean_specific, 'regression', info['scale']),
            'alphas': applied_alphas, 'calibration_alphas': policy['alphas'],
            'frozen_to_previous_common_adaptation': frozen_column,
            'candidate_count': total, 'retained_members': total // 10,
            'original_result': str(original), 'original_source_models_and_predictions_unchanged': True,
            'extra_clean_source_label_supervision': True, 'backbone_parameter_updates': False,
            'benchmark_query_labels_used_to_fit_coefficient': False,
            'development_variant_after_internal_results_observed': True})
        print('OLIGO_SPECIFIC_APPLIED', total, track, meta['dataset'], family, rho, flush=True)


def report():
    import csv
    rows = []
    for total in [60, 400]:
        for method, metric_key in [('OligoICP', 'OligoICP_specific_metrics'), ('FullMean', 'FullMean_specific_metrics')]:
            row = {'method': f'{method}_{total}_specific_source_adapter', 'variant': 'additional_development_adaptation',
                   'candidate_count': total, 'seed': 0}
            for track in ['five', 'seven']:
                values = []
                complete = []
                for meta in tasks('regression', track):
                    grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
                    folder = OUT / 'oligo_specific_predictions' / str(total) / track / str(meta['task_id'])
                    if all((folder / f'{f}_{r}.json').exists() for f, r in grid):
                        complete.append(meta)
                        values.extend(read(folder / f'{f}_{r}.json') for f, r in grid)
                row[track + '_datasets'] = len(complete)
                for is_clean in ([True, False] if track == 'five' else [False]):
                    part = [v[metric_key] for v in values if (v['rho'] == 0) == is_clean]
                    name = 'clean' if is_clean else track
                    row[name] = {k: float(np.mean([p[k] for p in part])) for k in ['r2', 'nrmse']} if part else None
            rows.append(row)
    atomic_json(OUT / 'reports/oligo_specific_summary.json', {'seed': 0, 'records': rows,
        'source_validation_coefficient_fitting_only': True, 'internal_development_variant': True,
        'original_raw_and_common_adaptation_results_unchanged': True, 'historical_Shared_not_a_selection_criterion': True})
    fields = ['method', 'candidate_count', 'five_datasets', 'seven_datasets', 'Clean_R2', 'Clean_nRMSE', 'Five_R2', 'Five_nRMSE', 'Seven_R2', 'Seven_nRMSE']
    with (OUT / 'reports/oligo_specific_averages.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            flat = {key: row.get(key, '') for key in fields}
            for track in ['clean', 'five', 'seven']:
                for key in ['r2', 'nrmse']:
                    label = track.capitalize() + ('_R2' if key == 'r2' else '_nRMSE')
                    flat[label] = row[track][key] if row[track] else ''
            writer.writerow(flat)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['fit', 'apply', 'report'])
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--candidates', type=int, choices=[60, 400], default=60)
    args = parser.parse_args(); torch.set_num_threads(1); logging.disable(logging.INFO); warnings.filterwarnings('ignore')
    if args.phase == 'report':
        report()
    else:
        if args.phase == 'fit':
            from tabpfn_constant_context_compat_v2 import install
            install()
        work = [(track, meta) for track in ['five', 'seven'] for meta in tasks('regression', track)][args.shard::args.shards]
        for track, meta in work:
            if args.phase == 'fit':
                fit(track, meta)
            else:
                apply(track, meta, args.candidates)
