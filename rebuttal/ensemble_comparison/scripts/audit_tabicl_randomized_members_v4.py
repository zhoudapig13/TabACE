"""Verify the fixed single-seed sensitivity run against retained defaults."""
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch

from prepare_data import ROOT, RUN, atomic_json, protocol
from internal_regression_benchmark import manifest
from randomized_tabicl_members_v4 import OUT, OLD, PLAN, TAG, member_spec
from evaluate_tabicl_finetuned_ensembles_v3 import METHODS, prediction
from report_tabicl_actual_finetuning_v3 import average
from tabicl_regression_comparison_v2 import load_condition
from run_tabicl import metrics


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_sampling(z, task_id, n=None, p=None):
    for bank_id, name in enumerate(['feature', 'bootstrap']):
        stored_rows = z[name + '_row_indices']
        stored_cols = z[name + '_column_masks']
        if n is None:
            n = stored_rows.shape[1]
        if p is None:
            p = stored_cols.shape[1]
        assert stored_rows.shape == (6, n)
        assert stored_cols.shape == (6, p)
        for member in range(6):
            rows, cols, seed, rf, cf = member_spec(task_id, bank_id, member, n, p)
            padded = np.full(n, -1, dtype=np.int64)
            padded[:len(rows)] = rows
            mask = np.zeros(p, dtype=bool)
            mask[cols] = True
            assert np.array_equal(stored_rows[member], padded)
            assert np.array_equal(stored_cols[member], mask)
            assert z[name + '_member_seeds'][member] == seed
            assert np.array_equal(z[name + '_fractions'][member], [rf, cf])
            assert np.array_equal(z[name + '_counts'][member], [len(rows), len(cols), len(np.unique(rows))])
        assert z[name].shape[0] == 6 and np.isfinite(z[name]).all()


def diversity(values, target):
    scale = max(float(np.std(target)), 1e-12)
    disagreement = float(np.sqrt(np.mean((values - values.mean(0)) ** 2)) / scale)
    centered = values - values.mean(1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    pairs = []
    skipped = 0
    for i in range(6):
        for j in range(i):
            if norms[i] <= 1e-12 or norms[j] <= 1e-12:
                skipped += 1
            else:
                pairs.append(float(centered[i] @ centered[j] / (norms[i] * norms[j])))
    return {'normalized_member_disagreement': disagreement,
            'mean_nonconstant_pair_correlation': float(np.mean(pairs)) if pairs else None,
            'valid_pairs': len(pairs), 'constant_pairs_omitted': skipped}


def main():
    torch.set_num_threads(1)
    dest = OUT / 'reports' / TAG
    summary = json.loads((dest / 'latest_summary.json').read_text())
    plan = json.loads(PLAN.read_text())
    assert plan['outer_seed'] == 0 and plan['outer_repetitions'] == 1
    assert plan['new_sensitivity_configurations'] == 1
    for relative, expected in json.loads((OUT / 'previous_result_lock.json').read_text())['sha256'].items():
        assert sha(RUN / relative) == expected
    prior = json.loads((OLD / 'reports/initial128/audit.json').read_text())
    assert prior['status'] == 'PASS'
    prior_checkpoints = {r['task_id']: r['checkpoint_sha256'] for r in prior['checkpoint_weight_changes_verified']}
    tasks = {m['task_id']: m for m in manifest()}
    checkpoint_checks = []
    validation_count = 0
    for task_id in tasks:
        folder = OUT / 'checkpoints' / TAG / str(task_id)
        expected = prior_checkpoints[task_id]
        assert sha(folder / 'best.ckpt') == expected
        split = json.loads((folder / 'source_split.json').read_text())
        train = set(split['train_original_row_ids'])
        val = set(split['validation_original_row_ids'])
        assert train.isdisjoint(val)
        with np.load(RUN / 'cache/five/regression' / f'{task_id}.npz') as z:
            assert not (train | val).intersection(z['target_indices'].tolist())
        policy = json.loads((OUT / 'combiner_policies' / TAG / f'{task_id}.json').read_text())
        assert policy['checkpoint_sha256'] == expected
        assert not policy['benchmark_query_labels_used'] and not policy['original_Base_fallback']
        paths = list((OUT / 'source_validation_banks' / TAG / str(task_id)).glob('*.npz'))
        assert len(paths) == 26
        for path in paths:
            meta = json.loads(path.with_suffix('.json').read_text())
            assert set(meta['context_row_ids']) <= train and set(meta['query_row_ids']) <= val
            assert not meta['benchmark_query_labels_used']
            assert meta['checkpoint_sha256'] == expected
            with np.load(path) as z:
                check_sampling(z, task_id, n=len(meta['context_row_ids']))
                assert np.isfinite(z['y']).all()
            validation_count += 1
        checkpoint_checks.append({'task_id': task_id, 'sha256': expected, 'unchanged_from_audited_v3': True})
    print('SENSITIVITY_CHECKPOINT_AND_VALIDATION_PASS', validation_count, flush=True)
    all_rows = defaultdict(list)
    diagnostics = defaultdict(list)
    control_max = {'single': 0., 'native': 0.}
    vectors = 0
    for track, expected_count in [('five', 735), ('seven', 957)]:
        paths = sorted((OUT / 'benchmark_predictions' / TAG / track).glob('*/*.npz'))
        assert len(paths) == expected_count
        for idx, path in enumerate(paths):
            meta = json.loads(path.with_suffix('.json').read_text())
            task_id = meta['task_id']
            data, info, base = load_condition(track, tasks[task_id], meta['family'], meta['rho'])
            assert meta['seed'] == 0
            assert not meta['benchmark_query_labels_used_for_training_or_selection']
            assert meta['input_hashes'] == {k: protocol.array_sha256(data[k]) for k in ['X_source', 'y_source', 'X_target', 'y_target']}
            policy = json.loads((OUT / 'combiner_policies' / TAG / f'{task_id}.json').read_text())
            old_path = OLD / 'benchmark_predictions/initial128' / track / str(task_id) / path.name
            group = ('clean' if track == 'five' else 'seven_clean_control') if meta['rho'] == 0 else track
            with np.load(path) as z, np.load(old_path) as old:
                check_sampling(z, task_id, len(data['y_source']), data['X_source'].shape[1])
                assert np.array_equal(z['y_target'], data['y_target'])
                assert np.array_equal(z['original_Base'], base)
                target = z['y_target']
                scale = max(1., float(np.std(target)))
                for control in ['single', 'native']:
                    error = float(np.max(np.abs(z[control] - old[control])) / scale)
                    control_max[control] = max(control_max[control], error)
                    assert np.allclose(z[control], old[control], rtol=1e-6, atol=1e-6 * scale), (path, control, error)
                bank = {k: z[k] for k in ['feature', 'bootstrap', 'iqr', 'single', 'native']}
                assert bank['feature'].shape == bank['bootstrap'].shape == bank['iqr'].shape == (6, len(target))
                all_rows[('Original_Base', track)].append({'rho': meta['rho'], **metrics(target, base, 'regression', info['scale'])})
                for method in METHODS:
                    pred = z['prediction__' + method]
                    assert np.isfinite(pred).all()
                    assert np.allclose(pred, prediction(bank, method, policy['models'][method]), rtol=1e-10, atol=1e-10)
                    observed = metrics(target, pred, 'regression', info['scale'])
                    for key in ['r2', 'nrmse']:
                        assert np.isclose(observed[key], meta['methods'][method][key], rtol=1e-10, atol=1e-10)
                    all_rows[(method, track)].append({'rho': meta['rho'], **observed})
                    vectors += 1
                for variant, arrays in [('previous_v3', old), ('randomized_v4', z)]:
                    for name in ['feature', 'bootstrap']:
                        diagnostics[(variant, group, name)].append(diversity(arrays[name], target))
            if (idx + 1) % 200 == 0:
                print('SENSITIVITY_AUDIT_PROGRESS', track, idx + 1, flush=True)
    for row in summary['records']:
        for track in ['five', 'seven']:
            observed = average(all_rows[(row['method'], track)], False)
            for key in observed:
                assert np.isclose(observed[key], row[track][key], rtol=1e-10, atol=1e-10)
        observed = average(all_rows[(row['method'], 'five')], True)
        for key in observed:
            assert np.isclose(observed[key], row['clean'][key], rtol=1e-10, atol=1e-10)
    for relative, expected in json.loads((OLD / 'benchmark_lock.json').read_text()).items():
        assert sha(RUN / relative) == expected
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    diagnostic_rows = []
    for (variant, group, bank_name), rows in sorted(diagnostics.items()):
        correlations = [r['mean_nonconstant_pair_correlation'] for r in rows if r['mean_nonconstant_pair_correlation'] is not None]
        diagnostic_rows.append({'variant': variant, 'track': group, 'bank': bank_name, 'conditions': len(rows),
            'normalized_member_disagreement': float(np.mean([r['normalized_member_disagreement'] for r in rows])),
            'mean_nonconstant_pair_correlation': float(np.mean(correlations)) if correlations else None,
            'conditions_with_valid_correlations': len(correlations),
            'valid_pairs': sum(r['valid_pairs'] for r in rows),
            'constant_pairs_omitted': sum(r['constant_pairs_omitted'] for r in rows)})
    atomic_json(dest / 'member_diversity.json', {'role': 'descriptive only; never used for configuration selection', 'records': diagnostic_rows})
    result = {'status': 'PASS', 'outer_seed': 0, 'outer_repetitions': 1,
        'unchanged_checkpoint_checks': checkpoint_checks, 'validation_banks_checked': validation_count,
        'canonical_input_hashes_verified': 1692, 'prediction_vectors_recomputed': vectors,
        'member_sample_and_seed_reconstruction': 'PASS', 'native_and_single_control_max_normalized_error': control_max,
        'all_previous_default_report_hashes_unchanged': True, 'benchmark_frozen_files_unchanged': True,
        'extra_clean_TRAIN_supervision_inherited_from_v3': True, 'Markdown_bytes': 0}
    atomic_json(dest / 'audit.json', result)
    summary['status'] = 'COMPLETE_AND_AUDITED'
    summary['audit'] = 'audit.json'
    summary['candidate_pools'] = {'random_subspaces': 6, 'random_patches': 6}
    summary['deployed_members'] = {'FT_single': 1, 'Native6': 6, 'IQR6': 'TRAIN validation selects 1, 3 or 6', 'other_ensembles': 6}
    summary['multiple_banks_computed_for_shared_evaluation_not_equal_total_runtime_budget'] = True
    atomic_json(dest / 'latest_summary.json', summary)
    atomic_json(dest / 'provenance.json', {'plan_sha256': sha(PLAN),
        'scripts': {str(p.relative_to(RUN)): sha(p) for p in [RUN / 'scripts/randomized_tabicl_members_v4.py', RUN / 'scripts/audit_tabicl_randomized_members_v4.py', RUN / 'scripts/evaluate_tabicl_finetuned_ensembles_v3.py']},
        'purpose': 'One fixed single-seed member-randomization sensitivity experiment',
        'setting': plan, 'results': 'averages.csv', 'analysis': 'member_diversity.json; numerical separation not a selection objective',
        'next_steps': 'Retain both default and sensitivity results; report limitations of shared updated backbone and additional TRAIN supervision.'})
    print('SINGLE_SEED_RANDOMIZED_SENSITIVITY_AUDIT_PASS', vectors, flush=True)
    print(json.dumps(diagnostic_rows, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
