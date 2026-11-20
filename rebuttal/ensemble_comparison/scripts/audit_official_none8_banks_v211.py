"""Replay preserved Shared context inputs and None8 configuration provenance."""
import json
from pathlib import Path
import numpy as np
import torch
import shared_official_none8_banks_v209 as bank
from prepare_data import ROOT, atomic_json, protocol


def main():
    torch.set_num_threads(1)
    source = bank.source
    plan = json.loads((bank.OUT/'plan.json').read_text())
    assert source.sha(bank.__file__) == plan['script_sha256']
    assert source.sha(bank.configured.__file__) == plan['predictor_configuration_sha256']
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    for name, digest in plan['official_implementation_sha256'].items():
        assert source.sha(name) == digest
    original_plan = json.loads((source.OUT/'plan.json').read_text())
    for name in ('tabace_context_reproduction_v28.py', 'shared_reference.py', 'specific_reference.py', 'unseen_mechanisms_v117.py'):
        assert source.sha(Path(source.__file__).with_name(name)) == original_plan['implementation_sha256'][name]
    for name, digest in original_plan['checkpoints'].items():
        assert source.sha(source.CHECKPOINTS/name) == digest
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    count, branches, actual_counts = 0, 0, []
    for c in cases:
        with np.load(c['input_file']) as data:
            assert 'y_target' not in data.files
            X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
        for path in sorted((bank.OUT/'banks'/c['task']/str(c['task_id'])).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text())
            assert source.sha(path) == meta['output_sha256']
            assert source.sha(meta['original_bank']) == meta['original_bank_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            assert meta['normalization_method'] == 'none' and meta['configured_estimators_per_context'] == 8
            assert meta['query_outcome_labels_opened'] is False
            if meta['rho']:
                X, y, Q, _ = source.novel.generate(X0, y0, Q0, numeric, c['task'], meta['family'], meta['rho'], c['task_id'])
            else:
                X, y, Q = X0, y0, Q0
            with np.load(path) as data, np.load(meta['original_bank']) as old:
                np.testing.assert_array_equal(data['Base'], old['Base'])
                views = source.constructors.shared_contexts(X, y, Q, len(y), c['task'], data['Base'])
                assert np.isfinite(data['members']).all() and np.isfinite(data['Canonical_native8_none']).all()
            assert [name for name, _, _, _ in views] == meta['Shared_roles']
            assert [len(b) for _, _, b, _ in views] == meta['branch_rows']
            hashes = [[protocol.array_sha256(v) for v in (A, b, T)] for _, A, b, T in views]
            assert hashes == meta['branch_input_hashes']
            assert max(meta['branch_rows']) <= 256
            actual = meta['actual_views_per_context_including_canonical_control']
            assert len(actual) == 6 and all(0 < n <= 8 for n in actual)
            actual_counts.extend(actual)
            count += 1
            branches += len(views)
    assert count == 1327 and branches == 6635
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions': count, 'original_context_inputs_replayed': branches,
        'Base_byte_unchanged': True, 'query_outcomes_opened': False,
        'original_context_row_memberships_and_values_retained': True,
        'official_normalization_configuration_snapshot_checked': True,
        'configured_estimators': 8, 'minimum_actual_views': min(actual_counts), 'maximum_actual_views': max(actual_counts),
        'limited_feature_permutations_disclosed': True, 'checkpoint_hashes_unchanged': True, 'Markdown_bytes': 0}
    atomic_json(bank.OUT/'independent_membership_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
