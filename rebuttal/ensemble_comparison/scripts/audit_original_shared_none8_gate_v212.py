"""Independent replay of a sealed six-context OOF adaptation report."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(module):
    run = importlib.import_module(module)
    out = run.OUT
    plan = json.loads((out/'plan.json').read_text())
    assert run.original.sha(run.__file__) == plan['script_sha256']
    assert run.original.sha(run.heads.__file__) == plan['head_sha256']
    assert run.original.sha(run.heads.fusion.__file__) == plan['fusion_implementation_sha256']
    assert run.original.sha(run.heads.fast.__file__) == plan['fast_metrics_sha256']
    assert run.original.sha(run.original.constructors.__file__) == plan['original_constructor_sha256']
    assert run.original.sha(run.calibration.__file__) == plan['calibration_selection_implementation_sha256']
    manifest = json.loads((run.original.OUT/'case_manifest.json').read_text())
    seal = json.loads((out/'policy_seal.json').read_text())
    policies, inner_count = {}, 0
    for name, digest in seal.items():
        path = out/name
        assert run.original.sha(path) == digest
        policy = json.loads(path.read_text())
        policies[name] = policy
        own = {c['data_id'] for c in manifest if c['task'] == policy['task'] and
            c['track'] == 'perturbation' and c['fold'] == policy['outer_fold']}
        assert own.isdisjoint(policy['training_data_ids'])
        if policy['mode'] == 'lomo':
            assert policy['excluded_family'] not in policy['training_families']
        for trial in policy['trials']:
            for inner in trial['inner']:
                train, valid = set(inner['training_data_ids']), set(inner['validation_data_ids'])
                assert train.isdisjoint(valid) and own.isdisjoint(train|valid)
                for family, score in inner['scores_by_family'].items():
                    if policy['mode'] == 'lomo':
                        assert family != policy['excluded_family']
                    assert np.isclose(score['gain'], score['candidate']['primary']-score['Base']['primary'])
                inner_count += 1
            gains = trial['family_gains']
            for family, gain in gains.items():
                values = [(i['scores_by_family'][family]['candidate']['datasets'],
                    i['scores_by_family'][family]['gain']) for i in trial['inner']]
                assert np.isclose(gain, sum(n*v for n, v in values)/sum(n for n, v in values))
            assert np.isclose(min(gains.values()), trial['score']['worst_family_gain'])
            assert np.isclose(np.mean(list(gains.values())), trial['score']['macro_family_gain'])
        best = max(policy['trials'], key=lambda t: (t['score']['worst_family_gain'], t['score']['macro_family_gain']))
        assert best['specification'] == policy['selected_specification']
        assert best['score'] == policy['selection_score']
    rows, banks, count = [], 0, 0
    for task in ('classification', 'regression'):
        for cell in run.load_cells(task):
            path = Path(cell['path'])
            assert run.original.sha(path) == cell['bank_sha256']
            assert run.original.sha(cell['native8_path']) == cell['native8_sha256']
            with np.load(cell['native8_path']) as native:
                np.testing.assert_array_equal(cell['native8_base'], native['Canonical_native8_none'])
                np.testing.assert_array_equal(cell['native8_bank'], native['members'])
                np.testing.assert_array_equal(cell['P'][0], native['Base'])
            assert run.original.sha(cell['native8_default_path']) == cell['native8_default_sha256']
            with np.load(cell['native8_default_path']) as defaults:
                np.testing.assert_array_equal(cell['native8_default'], defaults['Native8'])
            original = run.original.OUT/'predictions/perturbation'/task/str(cell['task_id'])/path.name
            with np.load(original) as z:
                np.testing.assert_array_equal(cell['P'][0], z['Base'])
            banks += 1
            for mode in ('pooled', 'lomo'):
                if mode == 'lomo' and cell['family'] == 'clean':
                    continue
                if mode == 'frozen_fusion':
                    prediction = cell['frozen']
                else:
                    key = 'all' if mode == 'pooled' else cell['family']
                    name = f'policies/{task}/{cell["fold"]}/{mode}/{key}.json'
                    policy = policies[name]
                    assert cell['data_id'] not in policy['training_data_ids']
                    if mode == 'lomo':
                        assert cell['family'] not in policy['training_families']
                    prediction = run.heads.predict({k: v for k, v in cell.items() if k != 'y'}, policy['model'])
                saved = out/'predictions'/task/mode/str(cell['task_id'])/path.name
                with np.load(saved) as z:
                    np.testing.assert_array_equal(prediction, z['prediction'])
                assert np.isfinite(prediction).all()
                if task == 'classification':
                    np.testing.assert_allclose(prediction.sum(axis=1), 1., atol=1e-9)
                count += 1
                for method, value in [('Base', cell['P'][0]), ('Shared_adapted', prediction), ('Native8_none_control', cell['native8_base']), ('Native8_default_control', cell['native8_default'])]:
                    score = run.metrics(cell['y'], value, task, max(float(cell['y'].std()), 1e-12))
                    if task == 'regression':
                        assert np.isclose(score['r2'], 1-score['nrmse']**2)
                    rows.append({k: cell[k] for k in ('task', 'data_id', 'family', 'rho')}
                        | {'mode': mode, 'method': method, **score})
    frame = pd.DataFrame(rows)
    table = pd.read_csv(out/'family_averages.csv')
    for row in table.to_dict('records'):
        selected = frame[(frame.task == row['task']) & (frame.family == row['family']) &
            (frame['mode'] == row['mode']) & (frame.method == row['method']) & (frame.rho > 0)]
        metrics = ['accuracy', 'auc'] if row['task'] == 'classification' else ['r2', 'nrmse']
        values = selected.groupby('data_id')[metrics].mean()
        assert len(values) == row['datasets']
        for metric in metrics:
            assert np.isclose(values[metric].mean(), row[metric], atol=1e-12)
    assert banks == 1327 and count == 2547 and len(table) == 48
    assert len(policies) == 40 and inner_count == 3360
    assert (run.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    selections = [{'task': p['task'], 'fold': p['outer_fold'], 'mode': p['mode'], 'specification': p['selected_specification']} for p in policies.values()]
    result = {'selected_original_gate_policies': selections, 'status': 'PASS', 'policies': len(policies), 'inner_selections': inner_count,
        'banks': banks, 'predictions_replayed_without_query_labels': count,
        'macro_rows_recomputed': len(table), 'Base_bitwise_unchanged': True,
        'outer_dataset_and_inner_dataset_exclusions': True, 'LOMO_generator_exclusion': True,
        'maxmin_candidate_selection': True, 'Markdown_bytes': 0,
        'development_adaptation_not_independent_confirmation': True,
        'original_five_contexts_and_reconstructed_fusion_formula_retained': True, 'internal_Native8_compute_change_explicit': True, 'None8_and_default_Native8_controls_separate': True, 'goal_achievement_not_asserted_by_this_audit': True}
    run.atomic_json(out/'independent_replay_audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('module')
    audit(parser.parse_args().module)
