"""Replay all joint-transport maps without loading query outcome files."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
import shared_joint_transport_banks_v177 as bank
from prepare_data import ROOT, atomic_json, protocol


def check_case(c):
    source = bank.source
    with np.load(c['input_file']) as inputs:
        assert 'y_target' not in inputs.files
        assert not set(inputs['source_indices']) & set(inputs['target_indices'])
        X0, y0, Q0, numeric = (inputs[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
    count, unconverged, max_error = 0, 0, 0.
    for path in sorted((bank.OUT/'banks'/c['task']/str(c['task_id'])).glob('*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        assert source.sha(path) == meta['output_sha256']
        assert source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
        assert meta['original_input_sha256'] == c['input_sha256']
        assert meta['final_query_labels_opened'] is False
        if meta['rho']:
            X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, c['task'], meta['family'], meta['rho'], c['task_id'])
        else:
            X, y, Q = X0, y0, Q0
        with np.load(path) as data, np.load(meta['parent_bank']) as parent:
            for name in ('Base', 'Canonical_native8'):
                np.testing.assert_array_equal(data[name], parent[name])
            np.testing.assert_array_equal(data['members'][:3], parent['members'][:3])
            repaired, info = bank.constructor.construct(X, y, Q, numeric, c['task'], data['Base'])
        assert info == meta['transport']
        assert [protocol.array_sha256(A) for A in repaired] == meta['repair_hashes']
        assert protocol.array_sha256(y) == meta['source_y_sha256']
        assert meta['new_context_rows'] == [len(y), len(y)] and len(y) <= 256
        categorical = np.setdiff1d(np.arange(X.shape[1]), numeric)
        for A in repaired:
            for j in categorical:
                assert set(A[:, j]) <= set(X[:, j]) | set(Q[:, j])
        unconverged += int(not info['converged'])
        max_error = max(max_error, info['row_marginal_max_error'], info['column_marginal_max_error'])
        count += 1
    return count, unconverged, max_error


def main():
    source = bank.source
    plan = json.loads((bank.OUT/'plan.json').read_text())
    assert source.sha(bank.__file__) == plan['script_sha256']
    assert source.sha(bank.constructor.__file__) == plan['constructor_sha256']
    assert source.sha(bank.RUN/'scripts/domain_overlap_context_v166.py') == plan['feature_encoder_sha256']
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    values = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(check_case, c) for c in cases]):
            values.append(future.result())
    assert sum(v[0] for v in values) == 1327
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions_replayed': 1327, 'query_outcome_files_opened': False,
        'Base_and_first_three_generic_members_and_Native8_control_unchanged': True,
        'source_rows_and_label_values_preserved': True, 'constructor_maps_replayed_exactly': True,
        'finite_iteration_transport_residuals_not_hidden': True,
        'conditions_not_reaching_1e_minus7_marginal_tolerance': sum(v[1] for v in values),
        'maximum_recorded_marginal_error': max(v[2] for v in values), 'Markdown_bytes': 0}
    atomic_json(bank.OUT/'independent_constructor_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
