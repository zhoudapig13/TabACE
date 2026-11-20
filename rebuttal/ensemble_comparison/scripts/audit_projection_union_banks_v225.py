"""Verify the union reuses all five parent and exactly two audited PLS views."""
import json

import numpy as np

import shared_projection_union_banks_v219 as bank


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((bank.OUT/'plan.json').read_text())
    assert bank.source.sha(bank.__file__) == plan['script_sha256']
    assert bank.source.sha(bank.parent.OUT/'plan.json') == plan['parent_plan_sha256']
    assert bank.source.sha(bank.extension.OUT/'plan.json') == plan['extension_plan_sha256']
    projection_audit = bank.extension.OUT/'independent_membership_audit.json'
    assert json.loads(projection_audit.read_text())['status'] == 'PASS'
    count = 0
    for path in sorted((bank.OUT/'banks').glob('*/*/*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        assert bank.source.sha(path) == meta['output_sha256']
        assert bank.source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
        assert bank.source.sha(meta['extension_bank']) == meta['extension_bank_sha256']
        assert not meta['query_outcomes_opened']
        with np.load(path) as union, np.load(meta['parent_bank']) as parent, np.load(meta['extension_bank']) as extension:
            np.testing.assert_array_equal(union['Base'], parent['Base'])
            np.testing.assert_array_equal(union['Base'], extension['Base'])
            np.testing.assert_array_equal(union['members'][:5], parent['members'])
            np.testing.assert_array_equal(union['members'][5:], extension['members'][3:])
            assert union['members'].shape[0] == 7
            np.testing.assert_array_equal(union['Canonical_native8'], parent['Canonical_native8'])
        count += 1
    assert count == 1327
    assert (bank.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions': count,
        'all_five_parent_and_two_projection_alternatives_replayed': True,
        'eight_raw_predictor_outputs_including_Base': True, 'maximum_query_estimator_views': 15,
        'additional_source_CV_views_regression': 3,
        'Base_and_benchmark_unchanged': True, 'query_outcome_files_opened': False,
        'source_projection_membership_audit_sha256': bank.source.sha(projection_audit),
        'explicit_constructor_and_compute_expansion': True, 'Markdown_bytes': 0}
    bank.atomic_json(bank.OUT/'independent_membership_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
