"""Cached heterogeneous contexts plus optional official Native-8 canonical gate.

This explicitly increases inference compute. It is not a matched-six-forward-
pass method. Native-8 alone must remain a separate descriptive control.
"""
import json
import numpy as np
import shared_generic_pool_banks_v163 as generic
from prepare_data import RUN, atomic_json

source = generic.source
OUT = RUN/'results/shared_native8_generic_v171'
NATIVE = RUN/'results/shared_native8_adaptation_v120'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for parent in (generic.OUT, NATIVE):
        assert json.loads((parent/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'CACHED_NATIVE8_AND_FULL_GENERIC_BANK_ADAPTATION', 'seed': 0,
        'context_cap': 256, 'alternative_contexts': 5, 'alternative_native_estimators_each': 1,
        'canonical_estimators_choices': [1, 8], 'maximum_TFM_estimator_forward_passes': [6, 13],
        'source_CV_extra_training_scoring_TFM_calls': 'up to3, cached',
        'not_original_Shared': True, 'not_compute_matched_to_Native1_Base': True,
        'Native8_only_control_required': True, 'new_TFM_inference': False,
        'Base_and_benchmark_and_source_pool_and_query_cohort_unchanged': True,
        'no_query_outcomes_opened_during_bank_assembly': True,
        'parent_plan_sha256': {str(p): source.sha(p/'plan.json') for p in (generic.OUT, NATIVE)},
        'script_sha256': source.sha(__file__), 'expected_conditions': 1327,
        'previously_evaluated_development_cohort': True, 'Markdown_written': False}
    p = OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)
    count = 0
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    for c in cases:
        task, tid = c['task'], c['task_id']
        for old in sorted((generic.OUT/'banks'/task/str(tid)).glob('*.npz')):
            meta = json.loads(old.with_suffix('.json').read_text())
            native = NATIVE/'predictions'/task/str(tid)/old.name
            nmeta = json.loads(native.with_suffix('.json').read_text())
            assert source.sha(old) == meta['output_sha256']
            assert source.sha(native) == nmeta['output_sha256']
            assert meta['original_input_sha256'] == nmeta['source_input_sha256'] == c['input_sha256']
            assert nmeta['native_estimators_each_call'] == 8
            with np.load(old) as data, np.load(native) as eight:
                np.testing.assert_array_equal(data['Base'], eight['Base1'])
                values = {k: data[k] for k in data.files}
                values['Canonical_native8'] = eight['Native8']
            dest = OUT/'banks'/task/str(tid)/old.name
            source.npwrite(dest, **values)
            atomic_json(dest.with_suffix('.json'), meta | {'output_sha256': source.sha(dest),
                'generic_parent_bank': str(old), 'generic_parent_sha256': source.sha(old),
                'native_parent_bank': str(native), 'native_parent_sha256': source.sha(native),
                'maximum_prediction_estimator_passes_if_native8_gate': 13})
            count += 1
    assert count == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'conditions': count})
    print('NATIVE8_GENERIC_BANK_COMPLETE', count, flush=True)


if __name__ == '__main__':
    main()
