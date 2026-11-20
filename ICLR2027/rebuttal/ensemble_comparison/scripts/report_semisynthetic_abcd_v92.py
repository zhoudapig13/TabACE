"""Report both preregistered versions; Holm over all twelve A-D comparisons."""
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd

import semisynthetic_abcd_v92 as study
import abcd_concept_calibrated_v87 as concept
import tabace_fusion_v28 as fusion
from prepare_data import atomic_json, protocol
from report_abcd_v84 import paired_statistics
from run_tabicl import metrics


def main():
    frames = []
    for version in study.VERSIONS:
        files = list((study.OUT / version / 'folds/classification').glob('*/detail.csv'))
        assert len(files) == 5, (version, len(files))
        frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        assert len(frame) == 2160
        frames.append(frame)
    regression = pd.read_csv(concept.OUT / 'detail.csv')
    regression['version'] = 'concept_calibrated_adapted'
    data = pd.concat([*frames, regression], ignore_index=True)
    assert len(data) == 5310
    rows = []
    for (task, version), group in data.groupby(['task', 'version']):
        primary, secondary = ('auc', 'accuracy') if task == 'classification' else ('r2', 'nrmse')
        scored = group[group.rho > 0].groupby(['data_id', 'method'])[[primary, secondary]].mean()
        clean = group[group.rho == 0].set_index(['data_id', 'method'])[[primary, secondary]]
        base = scored.xs('Base', level='method'); clean_base = clean.xs('Base', level='method')
        for method in ['Base', 'Single-matched', 'A', 'B', 'C', 'D']:
            values = scored.xs(method, level='method').loc[base.index]
            c = clean.xs(method, level='method').loc[base.index]
            rows.append({'task': task, 'version': version, 'method': method, 'datasets': len(base),
                primary: values[primary].mean(), secondary: values[secondary].mean(),
                'clean_primary': c[primary].mean(), 'clean_secondary': c[secondary].mean(),
                'clean_primary_change': (c[primary] - clean_base[primary]).mean(),
                'clean_secondary_change': (c[secondary] - clean_base[secondary]).mean(),
                **paired_statistics(values[primary] - base[primary])})
    table = pd.DataFrame(rows)
    tested = table.index[table.method.isin(['A', 'B', 'C', 'D'])].tolist(); assert len(tested) == 12
    last = 0.
    for rank, i in enumerate(sorted(tested, key=lambda j: table.loc[j, 'p_two_sided'])):
        last = max(last, min(1., (12 - rank) * table.loc[i, 'p_two_sided']))
        table.loc[i, 'p_holm'] = last
    table['positive_significant'] = (table.gain_ci_low > 0) & (table.p_holm < .05)
    primary = table[table.version != 'original_registry'].copy()
    data.to_csv(study.OUT / 'all_versions_detail.csv', index=False)
    table.to_csv(study.OUT / 'all_versions_tables.csv', index=False)
    primary.to_csv(study.OUT / 'primary_tables.csv', index=False)
    selected = study.selection(); meta_by_id = {m['task_id']: m for m in study.generator.cohort()}
    cell_sets = {(version, matched): {(c['meta']['task_id'], c['meta']['rho']): c
                  for c in study.load(matched, version) if c['meta']['track'] == 'five_semisynthetic'}
                 for version in study.VERSIONS for matched in (True, False)}
    # Audit every classifier row, including the predeclared original control.
    for frame in frames:
        for r in frame.to_dict('records'):
            version = r['version']; matched = r['method'] not in ('C', 'D')
            cell = cell_sets[(version, matched)][(r['task_id'], r['rho'])]
            folder = study.OUT / version / 'folds/classification' / str(r['fold'])
            path = folder / 'predictions' / str(r['task_id']) / f'{r["rho"]}_{r["method"]}.npz'
            with np.load(path) as z:
                prediction, y = z['prediction'], z['y']
            assert np.array_equal(y, cell['y'])
            if r['method'] == 'Base':
                expected = cell['base']
                original = Path(selected['prediction_root']) / str(r['task_id']) / selected['configuration']['name'] / f'{r["rho"]}.npz'
                with np.load(original) as z:
                    assert np.array_equal(y, z['y']) and np.array_equal(prediction, z['prediction'])
            elif r['method'] == 'Single-matched':
                expected = cell['bank'][3]
            else:
                p = json.loads((folder / f'{r["method"]}_policy.json').read_text())
                assert set(p['training_data_ids']).isdisjoint(p['excluded_data_ids'])
                assert p['unseen_mechanism_audit']
                assert p['policy']['regularization'] == min(p['trials'])[1]
                expected = fusion.predict(cell, p['policy'])
            assert np.allclose(prediction, expected, atol=1e-12, rtol=1e-12), path
            for key, value in metrics(y, prediction, 'classification', 1.).items():
                assert np.isclose(value, r[key], atol=1e-11, rtol=1e-11), path
    checked = 0
    # Recreate all semisynthetic inputs, and verify each density context is a
    # resampling of source rows with unchanged columns and source labels.
    for task_id, meta in meta_by_id.items():
        latent = study.generator.latent(meta)
        for rho in (0, 20, 50, 80, 100):
            cell = cell_sets[('original_registry', True)][(task_id, rho)]
            info = cell['meta']; case = study.generator.materialize(meta, selected['configuration'], rho, latent)
            assert protocol.array_sha256(case['X_source']) == info['density_source_X_sha256']
            assert protocol.array_sha256(case['y_source']) == info['density_source_y_sha256']
            assert protocol.array_sha256(case['X_target']) == info['query_X_sha256']
            A, b, ids, columns = study.density_context(case['X_source'], case['y_source'], case['X_target'])
            with np.load(cell['path']) as z:
                assert np.array_equal(ids, z['density_rows']) and np.array_equal(columns, z['density_columns'])
            assert A.shape == case['X_source'].shape and np.array_equal(A, case['X_source'][ids])
            assert np.array_equal(b, case['y_source'][ids])
            assert protocol.array_sha256(A) == info['selected_X_sha256']
            assert protocol.array_sha256(b) == info['selected_y_sha256']
            checked += 1
    assert checked == 360
    plan = json.loads((study.OUT / 'plan.json').read_text())
    assert plan['code_sha256'] == study.original.sha(study.__file__)
    assert plan['Base_selection_sha256'] == study.original.sha(study.calibration.OUT / 'Base_selection.json')
    md = study.original.ROOT / 'REBUTTAL_EXPERIMENTS.md'; assert md.stat().st_size == 0
    main_tests = primary[primary.method.isin(['A', 'B', 'C', 'D'])]
    result = {'status': 'PASS', 'classifier_rows_independently_recomputed': 4320,
        'semisynthetic_input_and_density_contexts_recreated': checked,
        'Base_bitwise_equal_to_pre_ABCD_calibration': True, 'outer_dataset_disjointness': True,
        'Base_alignment_max_abs_error': selected['development_result']['max_abs_error'],
        'Base_alignment_tolerance': .05, 'paired_dataset_inference': True, 'Holm_family': 12,
        'all_eight_primary_ABCD_positive_significant': bool(main_tests.positive_significant.all()),
        'target_labels_used_for_semisynthetic_feature_generation': True,
        'target_labels_passed_to_predictor_or_constructor': False,
        'regression_audit_reused': str(concept.OUT / 'audit.json'),
        'original_paper_replication': False, 'independent_confirmation': False,
        'single_seed': 0, 'Markdown_bytes': 0}
    atomic_json(study.OUT / 'audit.json', result)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
