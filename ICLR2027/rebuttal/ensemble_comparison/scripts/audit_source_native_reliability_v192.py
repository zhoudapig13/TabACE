"""Source-CV replay audit and descriptive transfer diagnosis.

The audit reads source labels only. Existing query scores are joined afterward
for a diagnostic correlation; they do not enter the reliability feature.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import source_native_reliability_v191 as run
from prepare_data import ROOT, RUN, atomic_json, protocol


def main():
    source = run.source
    plan = json.loads((run.OUT/'plan.json').read_text())
    assert source.sha(run.__file__) == plan['script_sha256']
    assert source.sha(run.crossfit.__file__) == plan['source_folding_sha256']
    assert json.loads((run.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    for name, digest in plan['frozen_checkpoints_sha256'].items():
        assert source.sha(source.CHECKPOINTS/f'tabicl-{name}-v2-20260212.ckpt') == digest
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    rows, count, fold_count = [], 0, 0
    for c in cases:
        assert source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as data:
            assert 'y_target' not in data.files
            assert not set(data['source_indices']) & set(data['target_indices'])
            y = data['y_source']
        expected_folds, rare = run.crossfit.folds(y, c['task'])
        for path in sorted((run.OUT/'features'/c['task']/str(c['task_id'])).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert source.sha(path) == meta['output_sha256']
            assert source.sha(meta['Base_CV_bank']) == meta['Base_CV_bank_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            assert meta['final_query_outcome_files_opened'] is False
            assert meta['source_y_sha256'] == protocol.array_sha256(y)
            assert len(expected_folds) == meta['Native8_source_CV_calls']
            np.testing.assert_array_equal(meta['mandatory_rare_rows'], rare)
            for (train, valid), recorded in zip(expected_folds, meta['source_crossfit_partitions']):
                np.testing.assert_array_equal(train, recorded['training_rows'])
                np.testing.assert_array_equal(valid, recorded['validation_rows'])
                assert not set(train) & set(valid)
                if c['task'] == 'classification':
                    assert set(y[valid]) <= set(y[train])
                fold_count += 1
            with np.load(path) as z, np.load(meta['Base_CV_bank']) as old:
                np.testing.assert_array_equal(z['SourceCV_Native1'], old['SourceCV_prediction'])
                a, b = z['SourceCV_Native1'], z['SourceCV_Native8']
                assert a.shape == b.shape
                valid = np.isfinite(b).all(axis=1) if b.ndim == 2 else np.isfinite(b)
                assert set(np.flatnonzero(~valid)) == set(rare)
                if b.ndim == 2:
                    np.testing.assert_allclose(b[valid].sum(axis=1), 1., atol=1e-9)
                values, info = run.reliability(y, a, b, c['task'])
                np.testing.assert_array_equal(values, z['source_reliability'])
                assert info == meta['features']
            rows.append({k: c[k] for k in ('task', 'task_id', 'data_id')} |
                {'family': meta['family'], 'rho': meta['rho']} |
                {name: float(value) for name, value in zip(run.NAMES, values)})
            count += 1
    assert count == 1327
    evidence = pd.DataFrame(rows)
    evidence.to_csv(run.OUT/'source_evidence_detail.csv', index=False)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(run.OUT/'independent_source_audit.json', {'status': 'PASS', 'conditions': count,
        'matched_source_folds': fold_count, 'Native1_source_predictions_byte_equal_to_parent': True,
        'source_fold_exclusions_and_rare_rows_checked': True,
        'evidence_recomputed_without_final_query_outcomes': True,
        'query_outcome_files_opened_during_source_audit': False,
        'Native8_predictions_provenance_not_reinferred_by_this_CPU_audit': True,
        'extra_source_calibration_compute_disclosed': True, 'Markdown_bytes': 0})
    # These are already-scored development outcomes, joined only for diagnosis.
    query = pd.read_csv(RUN/'results/shared_native8_generic_v171/calibration/detail.csv')
    query = query[(query['mode'] == 'pooled') & query.method.isin(['Base', 'Native8'])]
    diagnoses = []
    for task in ('classification', 'regression'):
        metric = 'auc' if task == 'classification' else 'r2'
        selected = query[query.task == task].pivot_table(index=['task_id', 'data_id', 'family', 'rho'],
            columns='method', values=metric).reset_index()
        selected['query_primary_gain_Native8'] = selected.Native8-selected.Base
        joined = evidence[evidence.task == task].merge(selected[['task_id', 'data_id', 'family', 'rho',
            'query_primary_gain_Native8']], on=['task_id', 'data_id', 'family', 'rho'], validate='one_to_one')
        joined.to_csv(run.OUT/f'{task}_transfer_diagnostic_detail.csv', index=False)
        for family, group in joined[joined.rho > 0].groupby('family'):
            units = group.groupby('data_id')[[run.NAMES[0], 'query_primary_gain_Native8']].mean()
            a, b = units[run.NAMES[0]].to_numpy(), units.query_primary_gain_Native8.to_numpy()
            nonconstant = min(a.std(), b.std()) > 1e-12
            diagnoses.append({'task': task, 'family': family, 'datasets': len(units),
                'source_CV_mean_primary_gain': float(a.mean()), 'query_mean_primary_gain': float(b.mean()),
                'Pearson_r': float(np.corrcoef(a, b)[0, 1]) if nonconstant else None,
                'Spearman_r': float(spearmanr(a, b).statistic) if nonconstant else None,
                'diagnostic_only_no_new_significance_test': True})
    pd.DataFrame(diagnoses).to_csv(run.OUT/'cv_to_query_diagnostic_correlations.csv', index=False)
    print(pd.DataFrame(diagnoses).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
