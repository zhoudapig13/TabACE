"""Replay sealed v127 predictions and compare v126 banks to their parent."""
import json

import numpy as np
import pandas as pd

import shared_hampel_group_calibration_v127 as run
from prepare_data import atomic_json, ROOT


def audit():
    manifest = json.loads((run.original.OUT/'case_manifest.json').read_text())
    rows, banks, predictions = [], 0, 0
    policy_cache = {}
    seal = json.loads((run.OUT/'policy_seal.json').read_text())
    for name, digest in seal.items():
        assert run.original.sha(run.OUT/name) == digest
    for task in ('classification', 'regression'):
        cases = {c['task_id']: c for c in manifest
            if c['task'] == task and c['track'] == 'perturbation'}
        for c in run.load_cells(task):
            path = run.Path(c['path'])
            meta = json.loads(path.with_suffix('.json').read_text())
            assert run.original.sha(path) == meta['output_sha256']
            old = run.Path(meta['previous_bank'])
            assert run.original.sha(old) == meta['previous_bank_sha256']
            source = cases[c['task_id']]
            assert meta['original_input_sha256'] == source['input_sha256']
            assert meta['source_labels_unchanged'] and not meta['query_labels_opened_for_inference']
            assert not meta['constructor_uses_query_labels']
            assert meta['context_rows'] <= 256
            previous = run.original.OUT/'predictions/perturbation'/task/str(c['task_id'])/path.name
            with np.load(previous) as z:
                np.testing.assert_array_equal(c['P'][0], z['Base'])
            with np.load(old) as z:
                np.testing.assert_array_equal(c['P'][0], z['Base'])
                retained = [i for i in range(5) if i != meta['replacement_slot']]
                np.testing.assert_array_equal(c['P'][1:][retained], z['members'][retained])
            banks += 1
            for mode in ('frozen_fusion', 'pooled', 'lomo'):
                if mode == 'lomo' and c['family'] == 'clean':
                    continue
                if mode == 'frozen_fusion':
                    pred = c['frozen']
                else:
                    key = 'all' if mode == 'pooled' else c['family']
                    name = f'policies/{task}/{c["fold"]}/{mode}/{key}.json'
                    if name not in policy_cache:
                        policy_cache[name] = json.loads((run.OUT/name).read_text())
                    policy = policy_cache[name]
                    assert c['data_id'] not in policy['training_data_ids']
                    if mode == 'lomo':
                        assert c['family'] not in policy['training_families']
                    pred = run.heads.predict({k: v for k, v in c.items() if k != 'y'}, policy['model'])
                f = run.OUT/'predictions'/task/mode/str(c['task_id'])/path.name
                with np.load(f) as z:
                    np.testing.assert_array_equal(pred, z['prediction'])
                assert np.isfinite(pred).all()
                if task == 'classification':
                    np.testing.assert_allclose(pred.sum(axis=1), 1.)
                predictions += 1
                for method, q in [('Base', c['P'][0]), ('Shared_adapted', pred)]:
                    rows.append({k: c[k] for k in ('task', 'data_id', 'family', 'rho')}
                        | {'mode': mode, 'method': method,
                            **run.metrics(c['y'], q, task, max(float(c['y'].std()), 1e-12))})
    detail = pd.DataFrame(rows)
    table = pd.read_csv(run.OUT/'family_averages.csv')
    for r in table.to_dict('records'):
        group = detail[(detail.task == r['task']) & (detail.family == r['family']) &
            (detail['mode'] == r['mode']) & (detail.method == r['method']) & (detail.rho > 0)]
        columns = ['accuracy', 'auc'] if r['task'] == 'classification' else ['r2', 'nrmse']
        values = group.groupby('data_id')[columns].mean()
        assert len(values) == r['datasets']
        for column in columns:
            assert np.isclose(values[column].mean(), r[column], atol=1e-12)
    assert banks == 1327 and predictions == 3874 and len(table) == 36
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'banks_replayed': banks, 'predictions_replayed': predictions,
        'macro_rows_recomputed': len(table), 'Base_bitwise_unchanged': True,
        'four_other_context_predictions_bitwise_unchanged_from_parent': True,
        'sealed_policies': True, 'deployment_without_query_labels_replayed': True,
        'dataset_OOF_and_LOMO_exclusions_checked': True, 'Markdown_bytes': 0,
        'goal_achievement_not_asserted': True}
    atomic_json(run.OUT/'independent_prediction_audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    audit()
