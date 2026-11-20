"""Replay v269 provenance/fusion and score the frozen extra-reference experiment.

Uses the run_tabicl.metrics(y,pred,task,scale) signature explicitly. This avoids
the argument-order error in the unused report() helper of frozen v269; no model
input, inference code, saved forecasts, or selection is changed by this repair.
"""
import json
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd

import shared_clean_reference_contexts_v269 as run
from prepare_data import ROOT, atomic_json, protocol
from run_tabicl import metrics


def main():
    out = run.OUT
    assert json.loads((out/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((out/'plan.json').read_text())
    assert run.source.sha(run.__file__) == plan['script_sha256']
    assert run.source.sha(run.source.constructors.__file__) == plan['constructors_sha256']
    assert run.source.sha(run.source.fusion.__file__) == plan['fusion_sha256']
    refs = {(r['task'], r['task_id']): r for r in json.loads((run.reference.OUT/'reference_manifest.json').read_text())}
    records, conditions = [], 0
    methods = ('Base', 'OriginalShared', 'Base_plus_same_reference', 'Shared_extra_reference')
    for c in json.loads((run.source.OUT/'case_manifest.json').read_text()):
        if c['track'] != 'perturbation':
            continue
        assert run.source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as z:
            X0, y0, Q0, numeric, sid, qid = (z[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns', 'source_indices', 'target_indices'))
        r = refs[(c['task'], c['task_id'])]
        assert run.source.sha(r['reference_file']) == r['reference_sha256']
        with np.load(r['reference_file']) as z:
            RX, Ry, rid = (z[k] for k in ('X_reference', 'y_reference', 'reference_indices'))
        assert len(Ry) == 64 and set(map(int, rid)).isdisjoint(map(int, np.r_[sid, qid]))
        rows = run.source_rows(y0, c['task'])
        policy_file = run.source.OUT/'policies'/c['task']/f'{c["fold"]}.json'
        policy = json.loads(policy_file.read_text())
        assert c['data_id'] not in policy['training_data_ids']
        # Labels are only opened here, after all forecast files are fixed.
        assert run.source.sha(c['truth_file']) == c['truth_sha256']
        with np.load(c['truth_file']) as z:
            truth = z['y_target']
        for family, rho in [('clean', 0)] + [(f, s) for f in c['families'] for s in run.source.novel.STRENGTHS]:
            path = out/'predictions'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert run.source.sha(path) == meta['output_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            assert meta['reference_sha256'] == r['reference_sha256']
            assert meta['policy_sha256'] == run.source.sha(policy_file)
            assert not meta['query_labels_opened_for_inference']
            assert meta['extra_labeled_reference_rows'] == 64
            assert meta['context_rows'] == 256 and meta['actual_TFM_calls'] <= 6
            np.testing.assert_array_equal(meta['retained_source_row_positions'], rows)
            X, y = X0, y0
            if rho:
                X, y, Q, _ = run.source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
                np.testing.assert_array_equal(Q, Q0)
            A, b = np.concatenate([X[rows], RX]), np.concatenate([y[rows], Ry])
            assert protocol.array_sha256(A) == meta['context_X_sha256']
            assert protocol.array_sha256(b) == meta['context_y_sha256']
            original = run.source.OUT/'predictions/perturbation'/c['task']/str(c['task_id'])/path.name
            assert run.source.sha(original) == meta['original_prediction_sha256']
            with np.load(path) as z, np.load(original) as old:
                np.testing.assert_array_equal(z['Base'], old['Base'])
                np.testing.assert_array_equal(z['OriginalShared'], old['Shared'])
                replay = run.source.fusion.predict({'task': c['task'],
                    'base': z['Base_plus_same_reference'], 'bank': z['members'],
                    'mu': float(b.mean()), 'sigma': max(float(b.std()), 1e-12)}, policy['Shared'])
                np.testing.assert_allclose(z['Shared_extra_reference'], replay, rtol=0, atol=1e-12)
                for method in methods:
                    assert np.isfinite(z[method]).all()
                    scores = metrics(truth, z[method], c['task'], max(float(np.std(truth)), 1e-12))
                    records.append({'task': c['task'], 'data_id': c['data_id'], 'family': family,
                                    'rho': rho, 'method': method, **scores})
            conditions += 1
    assert conditions == 1327
    df = pd.DataFrame(records)
    df.to_csv(out/'condition_metrics.csv', index=False)
    keys, cols = ['task', 'family', 'method'], ['accuracy', 'auc', 'r2', 'nrmse']
    dataset = df.groupby(keys+['data_id'])[cols].mean().reset_index()
    table = dataset.groupby(keys)[cols].mean().reset_index()
    table = table.merge(dataset.groupby(keys).size().rename('datasets').reset_index(), on=keys)
    table.to_csv(out/'family_averages.csv', index=False)
    gains = []
    for (task, family), g in table[table.family!='clean'].groupby(['task', 'family']):
        metric = 'accuracy' if task=='classification' else 'r2'
        baseline = float(g.loc[g.method=='Base', metric].iloc[0])
        adapted = float(g.loc[g.method=='Shared_extra_reference', metric].iloc[0])
        control = float(g.loc[g.method=='Base_plus_same_reference', metric].iloc[0])
        gains.append({'task': task, 'family': family, 'metric': metric,
            'Base': baseline, 'Shared_extra_reference': adapted,
            'gain_vs_original_Base': adapted-baseline,
            'gain_vs_same_reference_control': adapted-control,
            'required_gain': .05, 'passed': adapted-baseline >= .05})
    assert len(gains)==6
    atomic_json(out/'numerical_acceptance.json', {'family_results': gains,
        'all_six_pass': all(g['passed'] for g in gains),
        'additional_clean_labeled_rows_per_dataset': 64,
        'not_original_frozen_Shared': True, 'numeric_threshold_user_confirmed': True})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(out/'independent_audit.json', {'status': 'PASS', 'conditions': conditions,
        'Base_and_original_Shared_bitwise_unchanged': True,
        'source192_plus_clean_reference64_contexts_reconstructed': True,
        'all_reference_rows_disjoint_from_final_evaluation_and_source': True,
        'fusion_predictions_replayed': conditions,
        'extra_supervision_and_same_reference_control_explicit': True,
        'report_signature_repaired_without_changing_frozen_run': True,
        'report_script_sha256': run.source.sha(__file__), 'Markdown_bytes': 0})
    print(table.to_string(index=False), flush=True)
    print(json.dumps(gains, indent=2), flush=True)


if __name__ == '__main__':
    main()
