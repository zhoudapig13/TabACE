"""Recompute each source-only branch guard and explicit Base fallback."""
import json

import numpy as np

import shared_source_guarded_forest_banks_v236 as bank


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    source_audit = bank.quality.OUT/'independent_source_CV_audit.json'
    assert json.loads(source_audit.read_text())['status'] == 'PASS'
    plan = json.loads((bank.OUT/'plan.json').read_text())
    assert bank.source.sha(bank.__file__) == plan['script_sha256']
    assert bank.source.sha(bank.parent.OUT/'plan.json') == plan['parent_plan_sha256']
    assert bank.source.sha(bank.quality.OUT/'plan.json') == plan['source_quality_plan_sha256']
    count, enabled = 0, np.zeros(2, dtype=int)
    for path in sorted((bank.OUT/'banks').glob('*/*/*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        assert bank.source.sha(path) == meta['output_sha256']
        assert bank.source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
        assert bank.source.sha(meta['source_quality_file']) == meta['source_quality_sha256']
        assert not meta['query_task_labels_opened']
        with np.load(meta['source_quality_file']) as quality:
            gain = quality['source_proxy_gain']
        flags = gain > 0.
        np.testing.assert_array_equal(flags, meta['repair_enabled'])
        np.testing.assert_array_equal(gain, meta['source_proxy_loss_gains'])
        with np.load(path) as data, np.load(meta['parent_bank']) as parent:
            np.testing.assert_array_equal(data['Base'], parent['Base'])
            np.testing.assert_array_equal(data['members'][:5], parent['members'][:5])
            np.testing.assert_array_equal(data['RawRepair_predictions'], parent['members'][5:])
            for i, active in enumerate(flags):
                expected = parent['members'][5+i] if active else parent['Base']
                np.testing.assert_array_equal(data['members'][5+i], expected)
        enabled += flags
        count += 1
    assert count == 1327
    assert (bank.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions': count, 'guard_decisions_recomputed': 2*count,
        'enabled_counts': enabled.tolist(), 'Base_and_five_generic_alternatives_unchanged': True,
        'raw_repair_predictions_preserved': True, 'blocked_repairs_explicitly_alias_Base': True,
        'guard_threshold_fixed_zero': True, 'task_query_labels_opened': False,
        'source_CV_audit_sha256': bank.source.sha(source_audit),
        'extra_source_supervision_is_existing_training_labels_only': True, 'Markdown_bytes': 0}
    bank.atomic_json(bank.OUT/'independent_membership_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
