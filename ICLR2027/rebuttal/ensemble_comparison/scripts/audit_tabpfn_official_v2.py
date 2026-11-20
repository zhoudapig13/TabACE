"""Independent input, prediction, metric, supervision and candidate checks."""
import argparse
from collections import defaultdict
import hashlib
import json

import joblib
import numpy as np
import torch

from prepare_data import RUN, ROOT, atomic_json, protocol
from tabpfn_official_benchmark_v2 import OUT, tasks, conditions, frozen_input
from tabpfn_official_methods_v2 import source_clean
from tabpfn_oligoicp_v2 import candidate_spec, grouping
from run_tabpfn import aligned, CLASS_CKPT, REG_CKPT
from run_tabicl import metrics

METHODS = {'classification': ['Native6', 'PHE6', 'RF_PFN6', 'MixturePFN_MICP6_no_CaPFN'],
           'regression': ['Native6', 'PHE6', 'RF_PFN6']}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_locks():
    original = read(OUT / 'run_provenance_start.json')
    for group in ['scripts_and_official_library_sha256', 'frozen_benchmark_and_plan_sha256']:
        for relative, expected in original[group].items():
            assert sha(RUN / relative) == expected, relative
    for relative, expected in read(OUT / 'previous_result_lock.json').items():
        assert sha(RUN / relative) == expected
    dependencies = OUT / 'dependency_provenance.json'
    if dependencies.exists():
        pinned = read(dependencies)
        for relative, expected in pinned['dependencies_sha256'].items():
            assert sha(RUN / relative) == expected
        for absolute, expected in pinned['checkpoint_sha256'].items():
            from pathlib import Path
            assert sha(Path(absolute)) == expected
    runtime = OUT / 'runtime_compat_provenance.json'
    if runtime.exists():
        for relative, expected in read(runtime)['files_sha256'].items():
            assert sha(RUN / relative) == expected
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0


def source_policy(task, track, meta):
    folder = OUT / 'source_adapters' / task / track / str(meta['task_id'])
    policy = read(folder / 'policy.json')
    assert not policy['query_labels_used_for_fit_selection_or_stopping']
    assert not policy['backbone_gradient_finetuning']
    assert policy['extra_clean_source_TRAIN_supervision']
    X, y = source_clean(task, track, meta)
    with np.load(folder / 'source_split.npz') as split:
        train, val = split['train_relative_ids'], split['validation_relative_ids']
        assert set(train).isdisjoint(val)
        assert set(train) | set(val) == set(range(len(y)))
        assert str(split['pristine_source_X_sha256']) == protocol.array_sha256(X)
        assert str(split['pristine_source_y_sha256']) == protocol.array_sha256(y)
    scores = {key: .25 * values[0] + .75 * float(np.mean(values[1:]))
              for key, values in policy['all_validation_losses'].items()}
    assert all(np.isclose(scores[key], value) for key, value in policy['validation_loss_by_alpha'].items())
    assert float(min(scores, key=scores.get)) == policy['alpha']
    assert policy['alpha'] in [0., .1, .25]
    return joblib.load(folder / 'extra_trees.joblib'), policy


def check_metrics(truth, pred, observed, task, scale):
    assert np.isfinite(pred).all()
    actual = metrics(truth, pred, task, scale)
    for key in ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse', 'nmae']:
        if actual[key] is None:
            assert observed[key] is None
        else:
            assert np.isclose(actual[key], observed[key], rtol=1e-10, atol=1e-10), (key, actual[key], observed[key])
    return actual


def official(task):
    check_locks()
    count, vectors, policies = 0, 0, 0
    supervision_notes = 0
    collected = defaultdict(list)
    for track in ['five', 'seven']:
        for meta in tasks(task, track):
            auxiliary, policy = source_policy(task, track, meta)
            policies += 1
            grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
            for family, rho in grid:
                data, info, base = frozen_input(task, track, meta, family, rho)
                hashes = {key: protocol.array_sha256(value) for key, value in data.items()}
                auxpred = aligned(auxiliary, data['X_target'], task, info.get('n_classes'))
                group = 'clean' if rho == 0 else track
                collected[('Base', 'original', group)].append(metrics(data['y_target'], base, task, info['scale']))
                for method in METHODS[task]:
                    path = OUT / 'official_predictions' / task / track / str(meta['task_id']) / f'{family}_{rho}__{method}.npz'
                    observed = read(path.with_suffix('.json'))
                    assert observed['input_hashes'] == hashes and observed['seed'] == 0
                    assert not observed['benchmark_query_labels_used_for_training_or_selection']
                    assert not observed['backbone_gradient_finetuning']
                    assert observed['adapter_alpha'] == policy['alpha']
                    # Early metadata used this boolean for nonzero auxiliary use.
                    # Reports correctly mark ALL source-calibrated adaptations as
                    # extra supervision, including validation-selected alpha=0.
                    if not observed['extra_clean_source_TRAIN_supervision']:
                        assert policy['alpha'] == 0.
                        supervision_notes += 1
                    with np.load(path) as z:
                        assert np.array_equal(z['y_target'], data['y_target'])
                        assert np.array_equal(z['original_Base'], base)
                        expected = (1 - policy['alpha']) * z['raw_prediction'] + policy['alpha'] * auxpred
                        assert np.allclose(expected, z['adapted_prediction'], rtol=1e-10, atol=1e-10)
                        for variant in ['raw', 'adapted']:
                            pred = z[variant + '_prediction']
                            if task == 'classification':
                                assert pred.shape == (len(data['y_target']), info['n_classes'])
                                assert np.all(pred >= 0) and np.allclose(pred.sum(1), 1, atol=1e-6)
                            actual = check_metrics(data['y_target'], pred, observed[variant + '_metrics'], task, info['scale'])
                            collected[(method, variant, group)].append(actual)
                            vectors += 1
                    details = observed['method_details']
                    if not details.get('analytical_case'):
                        if method == 'PHE6':
                            assert len(details['portfolio']) == 6 and 0 < details['active_members'] <= 6
                        if method.startswith('Mixture'):
                            assert not details['trained_CaPFN'] and not details['full_MixturePFN_reproduction']
                            assert len(details['support_indices']) == min(6, len(data['y_source']))
                            for ids in details['support_indices']:
                                assert len(ids) <= 128 and set(ids) <= set(range(len(data['y_source'])))
                    count += 1
            print('OFFICIAL_AUDIT_DATASET_PASS', task, track, meta['dataset'], flush=True)
    assert count == (9968 if task == 'classification' else 4977)
    for row in read(OUT / 'reports/latest_summary.json')['records']:
        if row['task'] != task or (row['method'] != 'Base' and row['method'] not in METHODS[task]):
            continue
        if row['variant'] == 'adapted':
            assert row['extra_clean_source_supervision_in_fit_or_selection']
        for track in ['clean', 'five', 'seven']:
            values = collected[(row['method'], row['variant'], track)]
            for key in ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']:
                valid = [v[key] for v in values if v.get(key) is not None]
                assert np.isclose(float(np.mean(valid)), row[track][key], rtol=1e-10, atol=1e-10)
    result = {'status': 'PASS', 'task': task, 'seed': 0, 'outer_repetitions': 1,
        'complete_method_conditions': count, 'prediction_vectors_and_metrics_recomputed': vectors,
        'source_adapter_policies_checked': policies, 'all_canonical_inputs_hash_checked': True,
        'all_adapted_predictions_rebuilt_from_raw_plus_saved_auxiliary': True,
        'extra_supervision_in_selection_even_when_alpha_zero': True,
        'early_boolean_field_means_nonzero_auxiliary_weight_for_this_many_conditions': supervision_notes,
        'report_supervision_annotations_verified': True, 'benchmark_and_previous_results_unchanged': True,
        'checkpoint_sha256': sha(CLASS_CKPT if task == 'classification' else REG_CKPT), 'Markdown_bytes': 0}
    atomic_json(OUT / 'reports' / f'audit_{task}.json', result)
    print('OFFICIAL_TASK_AUDIT_PASS', task, count, flush=True)


def oligo():
    check_locks()
    profiles = {}; count = 0; vectors = 0; grouping_checks = []
    for track in ['five', 'seven']:
        items = tasks('regression', track)
        for index, meta in enumerate(items):
            auxiliary, policy = source_policy('regression', track, meta)
            grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
            for family, rho in grid:
                data, info, base = frozen_input('regression', track, meta, family, rho)
                folder = OUT / 'oligo_predictions' / track / str(meta['task_id']) / f'{family}_{rho}'
                progress = read(folder / 'progress.json')
                assert progress['completed_candidates'] == 400 and progress['seed'] == 0
                assert progress['input_hashes'] == {key: protocol.array_sha256(value) for key, value in data.items()}
                arrays = {name: np.load(folder / (name + '.npy'), mmap_mode='r')
                          for name in ['means', 'iqrs', 'seeds', 'group_masks', 'row_counts', 'seconds', 'calls']}
                labels = np.load(folder / 'group_labels.npy'); active = np.unique(labels)
                assert len(labels) == len(data['y_source']) and len(active) <= 20
                key = (meta['task_id'], track, tuple(active))
                if key not in profiles:
                    masks = np.zeros((400, 20), dtype=bool); seeds = []
                    for member in range(400):
                        _, groups, seed = candidate_spec(meta['task_id'], track, member, labels, active)
                        masks[member, groups] = True; seeds.append(seed)
                    profiles[key] = masks, np.asarray(seeds, dtype=np.uint32)
                masks, seeds = profiles[key]
                assert np.array_equal(masks, arrays['group_masks']) and np.array_equal(seeds, arrays['seeds'])
                expected_counts = masks.astype(np.int64) @ np.bincount(labels, minlength=20)
                assert np.array_equal(expected_counts, arrays['row_counts'])
                assert arrays['means'].shape == arrays['iqrs'].shape == (400, len(data['y_target']))
                assert np.isfinite(arrays['means']).all() and np.isfinite(arrays['iqrs']).all()
                assert np.all(arrays['iqrs'] >= 0)
                # Fixed source-only reconstruction sample, independent of scores.
                if index % 9 == 0 and (family, rho) == grid[0]:
                    rebuilt_labels, _ = grouping(data['X_source'])
                    assert np.array_equal(rebuilt_labels, labels)
                    grouping_checks.append({'track': track, 'task_id': meta['task_id'], 'family': family, 'rho': rho})
                auxpred = aligned(auxiliary, data['X_target'], 'regression', None)
                for total in [60, 400]:
                    observed = read(folder / f'results_{total}.json')
                    assert not observed['benchmark_query_labels_used_for_training_or_selection']
                    selected = np.argsort(arrays['iqrs'][:total].mean(1), kind='stable')[:total // 10]
                    assert observed['selected_indices'] == selected.tolist()
                    raw = arrays['means'][selected].mean(0); full = arrays['means'][:total].mean(0)
                    expected = {'raw_prediction': raw, 'adapted_prediction': (1 - policy['alpha']) * raw + policy['alpha'] * auxpred,
                        'full_ensemble_mean': full, 'full_ensemble_mean_adapted': (1 - policy['alpha']) * full + policy['alpha'] * auxpred}
                    fields = ['raw_metrics', 'adapted_metrics', 'equal_candidate_budget_full_mean_metrics', 'equal_candidate_budget_full_mean_adapted_metrics']
                    with np.load(folder / f'results_{total}.npz') as z:
                        assert np.array_equal(z['y_target'], data['y_target']) and np.array_equal(z['original_Base'], base)
                        for (name, pred), field in zip(expected.items(), fields):
                            assert np.allclose(pred, z[name], rtol=1e-10, atol=1e-10)
                            check_metrics(data['y_target'], pred, observed[field], 'regression', info['scale']); vectors += 1
                    assert observed['actual_predict_api_calls'] == int(arrays['calls'][:total].sum())
                    assert observed['includes_all_rejected_candidate_costs']
                count += 1
            print('OLIGO_AUDIT_DATASET_PASS', track, meta['dataset'], flush=True)
    assert count == 1659
    atomic_json(OUT / 'reports/audit_oligo.json', {'status': 'PASS', 'seed': 0, 'outer_repetitions': 1,
        'complete_condition_grids': count, 'candidate_predictions_checked': count * 400,
        'prediction_vectors_and_metrics_recomputed': vectors,
        'all_member_seeds_groups_and_row_counts_reconstructed': True,
        'source_only_KMeans_reconstruction_sample': grouping_checks,
        'all_IQR_selected_indices_reconstructed_without_labels': True,
        'both_raw_and_adapted_and_equal_candidate_budget_means_checked': True,
        'original_molecular_grouping_not_claimed': True, 'Markdown_bytes': 0})
    print('OLIGO_BOTH_BUDGETS_AUDIT_PASS', count, flush=True)


def final():
    check_locks()
    for name in ['classification', 'regression', 'oligo']:
        assert read(OUT / 'reports' / f'audit_{name}.json')['status'] == 'PASS'
    summary = read(OUT / 'reports/latest_summary.json')
    assert summary['status'] == 'COMPUTATION_COMPLETE_AUDIT_PENDING'
    summary['status'] = 'COMPLETE_AND_AUDITED'
    summary['audits'] = ['audit_classification.json', 'audit_regression.json', 'audit_oligo.json']
    atomic_json(OUT / 'reports/latest_summary.json', summary)
    atomic_json(OUT / 'completion_status.json', {'status': 'COMPLETE_AND_AUDITED', 'seed': 0,
        'outer_repetitions': 1, 'historical_Shared_not_rerun': True, 'Markdown_bytes': 0})
    print('ALL_TABPFN_OFFICIAL_AND_ADAPTED_COMPARISONS_AUDITED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['classification', 'regression', 'oligo', 'final'])
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.phase in ['classification', 'regression']:
        official(args.phase)
    elif args.phase == 'oligo':
        oligo()
    else:
        final()
