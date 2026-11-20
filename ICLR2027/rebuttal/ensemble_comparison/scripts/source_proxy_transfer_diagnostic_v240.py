"""Descriptive transfer check of fixed source-CV gains versus observed query gains.

Query truths are opened for post-hoc analysis only. No fitted model, gate
threshold, eligibility rule, or new primary hypothesis is changed here.
"""
import json
import os

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import source_proxy_repair_features_v234 as source_cv
import shared_task_residual_heads_v155 as fast

OUT = source_cv.OUT/'transfer_diagnostic_v240'


def correlation(x, y, rank=False):
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float((spearmanr if rank else pearsonr)(x, y).statistic)


def main():
    assert json.loads((source_cv.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = json.loads((source_cv.source.OUT/'case_manifest.json').read_text())
    for c in manifest:
        if c['track'] != 'perturbation':
            continue
        with np.load(c['truth_file']) as truth:
            y = truth['y_target']
        cell = {'task': c['task'], 'y': y}
        for family in c['families']:
            for rho in source_cv.source.novel.STRENGTHS:
                path = source_cv.bank.OUT/'banks'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
                meta = json.loads(path.with_suffix('.json').read_text())
                assert source_cv.source.sha(path) == meta['output_sha256']
                qpath = source_cv.OUT/'features'/c['task']/str(c['task_id'])/path.name
                qmeta = json.loads(qpath.with_suffix('.json').read_text())
                assert source_cv.source.sha(qpath) == qmeta['output_sha256']
                assert qmeta['feature_bank_sha256'] == meta['output_sha256']
                losses = qmeta['source_proxy_loss_mean_observed_mild_full']
                with np.load(path) as data:
                    baseline = fast.metric(cell, data['Base'])[0]
                    for i, method in enumerate(('mild', 'full')):
                        query_gain = fast.metric(cell, data['members'][5+i])[0]-baseline
                        source_gain = losses[0]-losses[1+i]
                        rows.append({k: c[k] for k in ('task', 'task_id', 'data_id')} |
                            {'family': family, 'rho': rho, 'method': method,
                                'source_relative_loss_gain': source_gain/max(losses[0], 1e-12),
                                'query_primary_gain': query_gain, 'source_guard_enabled': source_gain > 0.})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT/'condition_transfer.csv', index=False)
    units = detail.groupby(['task', 'family', 'method', 'data_id'])[['source_relative_loss_gain', 'query_primary_gain']].mean()
    summaries = []
    for (task, family, method), frame in units.groupby(['task', 'family', 'method']):
        x, y = frame.source_relative_loss_gain.to_numpy(), frame.query_primary_gain.to_numpy()
        condition = detail[(detail.task == task) & (detail.family == family) & (detail.method == method)]
        enabled = condition.source_guard_enabled
        summaries.append({'task': task, 'family': family, 'method': method, 'datasets': len(frame),
            'dataset_Pearson': correlation(x, y), 'dataset_Spearman': correlation(x, y, True),
            'source_guard_enabled_fraction': float(enabled.mean()),
            'query_primary_gain_mean': float(y.mean()),
            'query_gain_mean_enabled': float(condition.loc[enabled, 'query_primary_gain'].mean()) if enabled.any() else None,
            'query_gain_mean_blocked': float(condition.loc[~enabled, 'query_primary_gain'].mean()) if (~enabled).any() else None,
            'harmful_query_fraction_among_enabled': float((condition.loc[enabled, 'query_primary_gain'] < 0).mean()) if enabled.any() else None})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'transfer_summary.csv', index=False)
    source_cv.atomic_json(OUT/'analysis_scope.json', {
        'status': 'COMPLETE', 'script_sha256': source_cv.source.sha(__file__),
        'source_quality_plan_sha256': source_cv.source.sha(source_cv.OUT/'plan.json'),
        'feature_bank_plan_sha256': source_cv.source.sha(source_cv.bank.OUT/'plan.json'),
        'task_query_labels_opened_for_posthoc_analysis_only': True,
        'no_model_or_threshold_fit': True, 'no_new_primary_hypothesis_tests': True,
        'each_dataset_has_four_severities_in_each_eligible_family': True,
        'Markdown_bytes': (source_cv.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size})
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
