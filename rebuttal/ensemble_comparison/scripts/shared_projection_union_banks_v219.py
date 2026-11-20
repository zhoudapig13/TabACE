"""Keep all five v182 alternatives and append both source-PLS alternatives.

Eight predictor outputs including Base; up to15 internal estimator views.
This explicitly expands the constructor count and is not original Shared.
"""
import json

import numpy as np

import shared_dual_canonical_banks_v182 as parent
import shared_supervised_projection_banks_v215 as extension
from prepare_data import RUN, atomic_json

source = parent.source
OUT = RUN/'results/shared_projection_union_v219'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((extension.OUT/'independent_membership_audit.json').read_text())['status'] == 'PASS'
    plan = {'status': 'EIGHT_OUTPUT_CONSTRUCTOR_UNION_NOT_ORIGINAL_SHARED', 'seed': 0,
        'context_row_cap': 256, 'predictor_outputs_including_Base': 8,
        'alternative_outputs': 7, 'all_five_parent_alternatives_retained': True,
        'added_alternatives': ['robust_scaled_source_PLS', 'marginal_rank_source_PLS'],
        'source_CV_TFM_calls': {'classification': 0, 'regression': 3},
        'maximum_prediction_estimator_passes': 15, 'compute_matched_to_Native1_Base': False,
        'Base_and_benchmark_and_query_cohort_unchanged': True,
        'family_specific_context_selection': False, 'new_TFM_inference': False,
        'extra_labeled_rows': 0, 'local_observed_source_labels_used': True,
        'own_query_outcome_files_opened': False,
        'parent_plan_sha256': source.sha(parent.OUT/'plan.json'),
        'extension_plan_sha256': source.sha(extension.OUT/'plan.json'),
        'script_sha256': source.sha(__file__), 'expected_conditions': 1327,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'Markdown_written': False}
    p = OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)
    count = 0
    for old in sorted((parent.OUT/'banks').glob('*/*/*.npz')):
        meta = json.loads(old.with_suffix('.json').read_text())
        new = extension.OUT/'banks'/old.relative_to(parent.OUT/'banks')
        newmeta = json.loads(new.with_suffix('.json').read_text())
        assert source.sha(old) == meta['output_sha256']
        assert source.sha(new) == newmeta['output_sha256']
        assert meta['original_input_sha256'] == newmeta['original_input_sha256']
        assert newmeta['parent_bank_sha256'] == meta['output_sha256']
        with np.load(old) as raw, np.load(new) as extra:
            np.testing.assert_array_equal(raw['Base'], extra['Base'])
            np.testing.assert_array_equal(raw['members'][:3], extra['members'][:3])
            members = np.concatenate([raw['members'], extra['members'][3:]], axis=0)
            assert members.shape[0] == 7 and np.isfinite(members).all()
            dest = OUT/'banks'/old.relative_to(parent.OUT/'banks')
            # FrozenFusion is a retained five-alternative diagnostic control,
            # not the fitted union prediction and not used by the union report.
            source.npwrite(dest, Base=raw['Base'], members=members,
                Canonical_native8=raw['Canonical_native8'], FrozenFusion=raw['FrozenFusion'],
                source_mean=raw['source_mean'], source_scale=raw['source_scale'])
        atomic_json(dest.with_suffix('.json'), {k: meta[k] for k in ('task', 'task_id', 'data_id', 'fold', 'family', 'rho', 'original_input_sha256')} |
            {'parent_bank': str(old), 'parent_bank_sha256': meta['output_sha256'],
                'extension_bank': str(new), 'extension_bank_sha256': newmeta['output_sha256'],
                'output_sha256': source.sha(dest), 'parent_member_indices': list(range(5)),
                'extension_member_indices': [3, 4], 'query_outcomes_opened': False,
                'raw_TFM_predictor_outputs_including_Base': 8})
        count += 1
    assert count == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'conditions': count})
    print('PROJECTION_UNION_BANK_COMPLETE', count, flush=True)


if __name__ == '__main__':
    main()
