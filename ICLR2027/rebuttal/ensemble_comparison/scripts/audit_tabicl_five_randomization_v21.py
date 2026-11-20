"""Verify sampling, unchanged checkpoints, data separation and all new metrics."""
import json
import numpy as np
import pandas as pd

from prepare_data import RUN, atomic_json
from tabicl_five_randomization_v21 import OUT, PRIOR, TRAINED, TAG, PLAN, sha, member_spec
from evaluate_tabicl_finetuned_ensembles_v3 import prediction, METHODS


def sampling(z, task_id):
    n = z['feature_row_indices'].shape[1]
    p = z['feature_column_masks'].shape[1]
    for bank_id, name in [(0, 'feature'), (1, 'bootstrap')]:
        for member in range(6):
            rows, cols, seed, rf, cf = member_spec(task_id, bank_id, member, n, p)
            padded = np.full(n, -1, dtype=np.int64); padded[:len(rows)] = rows
            mask = np.zeros(p, dtype=bool); mask[cols] = True
            assert np.array_equal(z[name + '_row_indices'][member], padded)
            assert np.array_equal(z[name + '_column_masks'][member], mask)
            assert z[name + '_member_seeds'][member] == seed
            assert np.array_equal(z[name + '_fractions'][member], [rf, cf])


def main():
    previous_audit = json.loads((TRAINED / 'reports/initial128/audit.json').read_text())
    expected = {m['task_id']: m['checkpoint_sha256'] for m in previous_audit['checkpoint_weight_changes_verified']}
    validation = 0
    for task_id, digest in expected.items():
        root = TRAINED / 'checkpoints/initial128' / str(task_id)
        assert sha(root / 'best.ckpt') == digest
        split = json.loads((root / 'source_split.json').read_text())
        train = set(split['train_original_row_ids']); val = set(split['validation_original_row_ids'])
        assert train.isdisjoint(val)
        with np.load(RUN / 'cache/five/regression' / f'{task_id}.npz') as z:
            assert (train | val).isdisjoint(z['target_indices'])
        policy = json.loads((OUT / 'combiner_policies' / f'{task_id}.json').read_text())
        assert policy['checkpoint_sha256'] == digest and not policy['benchmark_query_labels_used']
        paths = [p for p in (OUT / 'source_validation_banks' / TAG / str(task_id)).glob('*.json') if not p.name.endswith('.tmp.json')]
        assert len(paths) == 26
        for path in paths:
            m = json.loads(path.read_text())
            assert set(m['context_row_ids']) <= train and set(m['query_row_ids']) <= val
            assert not m['benchmark_query_labels_used']
            with np.load(path.with_suffix('.npz')) as z: sampling(z, task_id)
            validation += 1
    rows, diagnostics = [], []
    paths = [p for p in (OUT / 'predictions/five').glob('*/*.json') if not p.name.endswith('.tmp.json')]
    assert len(paths) == 700
    for path in paths:
        m = json.loads(path.read_text()); task_id = m['task_id']
        assert m['seed'] == 0 and m['rho'] in [20, 50, 80, 100]
        assert m['checkpoint_sha256'] == expected[task_id] and not m['checkpoint_retrained']
        assert not m['benchmark_query_labels_used_for_training_or_selection']
        old = RUN / m['control_predictions_reused_from']
        assert m['input_hashes'] == json.loads(old.with_suffix('.json').read_text())['input_hashes']
        models = json.loads((OUT / 'combiner_policies' / f'{task_id}.json').read_text())['models']
        with np.load(path.with_suffix('.npz')) as z, np.load(old) as prior:
            sampling(z, task_id)
            for key in ['single', 'native', 'original_Base', 'y_target']:
                assert np.array_equal(z[key], prior[key])
            bank = {k: z[k] for k in ['feature', 'bootstrap', 'iqr', 'single', 'native']}
            truth = z['y_target']
            for method in METHODS:
                pred = z['prediction__' + method]
                assert np.allclose(pred, prediction(bank, method, models[method]), atol=1e-10, rtol=1e-10)
                error = pred - truth
                r2 = 1 - float(np.sum(error ** 2)) / max(float(np.sum((truth - truth.mean()) ** 2)), 1e-12)
                nrmse = float(np.sqrt(np.mean(error ** 2))) / m['scale']
                assert np.isclose(r2, m['methods'][method]['r2'], atol=1e-10, rtol=1e-10)
                assert np.isclose(nrmse, m['methods'][method]['nrmse'], atol=1e-10, rtol=1e-10)
                rows.append({'task_id': task_id, 'family': m['family'], 'rho': m['rho'], 'method': method})
            for name in ['feature', 'bootstrap']:
                for variant, arrays in [('previous_v4', prior), ('stronger_v21', z)]:
                    p = arrays[name]
                    value = float(np.sqrt(np.mean((p - p.mean(0)) ** 2))) / m['scale']
                    diagnostics.append({'bank': name, 'variant': variant, 'normalized_member_disagreement': value})
    frame = pd.DataFrame(rows)
    assert not frame.duplicated().any() and frame.task_id.nunique() == 35
    for method, g in frame.groupby('method'): assert len(g) == 700
    plan = json.loads(PLAN.read_text())
    for relative, size, mtime in plan['Clean_seven_frozen_file_index']:
        s = (RUN / relative).stat(); assert (s.st_size, s.st_mtime_ns) == (size, mtime)
    for relative, digest in plan['previous_reports_sha256'].items(): assert sha(RUN / relative) == digest
    assert (RUN.parents[1] / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    diversity = pd.DataFrame(diagnostics).groupby(['bank', 'variant'], as_index=False).mean()
    diversity.to_csv(OUT / 'member_diversity.csv', index=False)
    atomic_json(OUT / 'audit.json', {'status': 'PASS', 'benchmark_conditions': 700,
                'prediction_vectors_checked': len(rows), 'TRAIN_validation_banks_checked': validation,
                'frozen_Clean_seven_artifacts': len(plan['Clean_seven_frozen_file_index']),
                'checkpoints_unchanged': 35, 'Base_Shared_not_rerun': True,
                'diversity': diversity.to_dict(orient='records'),
                'all_outcomes_preserved_without_decimal_or_interval_selection': True})
    print('PASS', diversity.to_dict(orient='records'), flush=True)


if __name__ == '__main__': main()
