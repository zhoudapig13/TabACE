"""Verify the additional source-only Oligo coefficient and derived metrics."""
import json

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from prepare_data import atomic_json, protocol, ROOT
from tabpfn_official_benchmark_v2 import OUT, tasks, conditions, frozen_input
from tabpfn_official_methods_v2 import source_clean, source_validation_condition
from tabpfn_oligo_source_adaptation_v2 import coefficient
from run_tabpfn import aligned
from run_tabicl import metrics


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(1)
    collected = {}; validation_count = 0; vector_count = 0
    for track in ['five', 'seven']:
        for meta in tasks('regression', track):
            folder = OUT / 'oligo_specific_adapters' / track / str(meta['task_id'])
            policy = read(folder / 'policy.json')
            assert not policy['benchmark_query_labels_used_to_fit_coefficient']
            assert not policy['historical_Shared_used_for_selection'] and not policy['backbone_parameter_updates']
            X, y = source_clean('regression', track, meta)
            common = OUT / 'source_adapters/regression' / track / str(meta['task_id'])
            with np.load(common / 'source_split.npz') as z:
                tr, val = z['train_relative_ids'], z['validation_relative_ids']
            assert set(tr).isdisjoint(val) and set(tr) | set(val) == set(range(len(y)))
            inner = ExtraTreesRegressor(**read(common / 'policy.json')['ExtraTrees_parameters']).fit(X[tr], y[tr])
            cells = []
            from seven_shifts import FAMILIES
            families = protocol.FAMILIES if track == 'five' else FAMILIES
            grid = [('original' if track == 'five' else 'covariate', 0)] + [(f, r) for f in families for r in [50, 100]]
            for family, rho in grid:
                A, b, V, truth = source_validation_condition('regression', track, meta, X[tr], y[tr], X[val], y[val], family, rho)
                path = folder / f'{family}_{rho}.npz'; info = read(path.with_suffix('.json'))
                assert info['input_hashes'] == {key: protocol.array_sha256(value) for key, value in zip(['X', 'y', 'Q', 'truth'], [A, b, V, truth])}
                assert not info['benchmark_query_labels_used'] and info['source_validation_only']
                assert info['train_relative_ids'] == tr.tolist() and info['validation_relative_ids'] == val.tolist()
                with np.load(path) as z:
                    assert z['means'].shape == z['iqrs'].shape == (60, len(truth))
                    assert np.isfinite(z['means']).all() and np.isfinite(z['iqrs']).all() and np.all(z['iqrs'] >= 0)
                    selected = np.argsort(z['iqrs'].mean(1), kind='stable')[:6]
                    assert np.array_equal(selected, z['selected'])
                    assert np.allclose(z['raw'], z['means'][selected].mean(0), rtol=1e-10, atol=1e-10)
                    assert np.allclose(z['full_mean'], z['means'].mean(0), rtol=1e-10, atol=1e-10)
                    assert np.array_equal(z['truth'], truth)
                    assert np.allclose(z['auxiliary'], inner.predict(V), rtol=1e-10, atol=1e-10)
                    cells.append({key: z[key] for key in ['raw', 'full_mean', 'auxiliary', 'truth', 'normalization_variance']})
                validation_count += 1
            for key in ['raw', 'full_mean']:
                assert np.isclose(coefficient(cells, key), policy['alphas'][key], rtol=1e-12, atol=1e-12)
                assert 0 <= policy['alphas'][key] <= .5
            auxiliary = joblib.load(common / 'extra_trees.joblib')
            for total in [60, 400]:
                for family, rho in ([('original', 0)] if track == 'five' else []) + conditions(track):
                    data, info, base = frozen_input('regression', track, meta, family, rho)
                    original = OUT / 'oligo_predictions' / track / str(meta['task_id']) / f'{family}_{rho}' / f'results_{total}.npz'
                    path = OUT / 'oligo_specific_predictions' / str(total) / track / str(meta['task_id']) / f'{family}_{rho}.npz'
                    observed = read(path.with_suffix('.json'))
                    assert observed['calibration_alphas'] == policy['alphas'] and not observed['benchmark_query_labels_used_to_fit_coefficient']
                    extra = aligned(auxiliary, data['X_target'], 'regression', None)
                    with np.load(original) as source, np.load(path) as z:
                        assert np.array_equal(z['y_target'], data['y_target']) and np.array_equal(z['original_Base'], base)
                        for key, field, metric_key in [('raw_prediction', 'adapted_prediction', 'OligoICP_specific_metrics'),
                                                       ('full_ensemble_mean', 'full_mean_adapted_prediction', 'FullMean_specific_metrics')]:
                            if observed['frozen_to_previous_common_adaptation']:
                                assert track == 'five' or rho == 0
                                pred = source['adapted_prediction' if key == 'raw_prediction' else 'full_ensemble_mean_adapted']
                            else:
                                alpha = policy['alphas']['raw' if key == 'raw_prediction' else 'full_mean']
                                pred = (1 - alpha) * source[key] + alpha * extra
                            assert np.allclose(pred, z[field], rtol=1e-10, atol=1e-10)
                            actual = metrics(data['y_target'], pred, 'regression', info['scale'])
                            for metric in ['r2', 'nrmse', 'nmae']:
                                assert np.isclose(actual[metric], observed[metric_key][metric], rtol=1e-10, atol=1e-10)
                            group = 'clean' if rho == 0 else track
                            collected.setdefault((total, 'OligoICP' if key == 'raw_prediction' else 'FullMean', group), []).append(actual)
                            vector_count += 1
            print('OLIGO_SPECIFIC_AUDIT_DATASET_PASS', track, meta['dataset'], flush=True)
    summary = read(OUT / 'reports/oligo_specific_summary.json')
    for row in summary['records']:
        name = 'OligoICP' if row['method'].startswith('OligoICP') else 'FullMean'
        assert row['five_datasets'] == 35 and row['seven_datasets'] == 33
        for track in ['clean', 'five', 'seven']:
            for key in ['r2', 'nrmse']:
                actual = float(np.mean([x[key] for x in collected[(row['candidate_count'], name, track)]]))
                assert np.isclose(actual, row[track][key], rtol=1e-10, atol=1e-10)
    assert validation_count == 880 and vector_count == 6636
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'reports/audit_oligo_specific.json', {'status': 'PASS', 'seed': 0,
        'source_validation_banks_checked': validation_count, 'prediction_vectors_and_metrics_recomputed': vector_count,
        'coefficients_reconstructed_from_source_validation_only': True,
        'inner_auxiliary_models_refit_on_training_partition_only_for_audit': True,
        'original_raw_and_common_adaptation_outputs_preserved': True,
        'development_variant_after_internal_results_observed': True, 'Markdown_bytes': 0})
    summary['status'] = 'COMPLETE_AND_AUDITED'; atomic_json(OUT / 'reports/oligo_specific_summary.json', summary)
    print('OLIGO_SPECIFIC_SOURCE_ADAPTATION_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    main()
