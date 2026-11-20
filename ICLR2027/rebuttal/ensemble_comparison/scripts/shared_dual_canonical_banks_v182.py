"""Keep Native1 and Native8 together, plus four fixed generic alternatives.

This avoids selecting a single canonical estimator for an entire dataset fold.
Six predictors cost up to13 estimator passes; compute differs from Native1 Base.
"""
import json
import numpy as np
import shared_native8_generic_banks_v171 as parent
from prepare_data import RUN, atomic_json

source = parent.source
OUT = RUN/'results/shared_dual_canonical_v182'
INDICES = {'classification': [0, 1, 2, 3], 'regression': [0, 1, 3, 4]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((parent.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = {'status': 'DUAL_NATIVE_CANONICAL_AND_GENERIC_BANK_ADAPTATION', 'seed': 0,
        'context_cap': 256, 'predictors_including_Base': 6,
        'alternative_predictors': 'Native8 plus four fixed generic Native1 views',
        'generic_indices_by_task': INDICES, 'indices_fixed_before_new_fusion_fit': True,
        'source_CV_TFM_training_calls': {'classification': 0, 'regression': 'up to3'},
        'maximum_prediction_estimator_passes': 13, 'compute_matched_to_Native1_Base': False,
        'Native8_only_control_required': True, 'Base_and_benchmark_and_query_cohort_unchanged': True,
        'no_family_or_severity_specific_bank_selection': True,
        'not_original_Shared': True, 'no_new_TFM_inference': True,
        'task_query_outcomes_opened_in_bank_assembly': False,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'parent_plan_sha256': source.sha(parent.OUT/'plan.json'),
        'script_sha256': source.sha(__file__), 'expected_conditions': 1327, 'Markdown_written': False}
    p = OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)
    count = 0
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    for c in cases:
        task, tid = c['task'], c['task_id']
        for old in sorted((parent.OUT/'banks'/task/str(tid)).glob('*.npz')):
            meta = json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old) == meta['output_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            with np.load(old) as data:
                base, native8 = data['Base'], data['Canonical_native8']
                members = np.concatenate([native8[None], data['members'][INDICES[task]]], axis=0)
                mu, sigma = float(data['source_mean']), float(data['source_scale'])
            assert len(members) == 5
            policy = json.loads((source.OUT/'policies'/task/f'{c["fold"]}.json').read_text())
            frozen = source.fusion.predict({'task': task, 'base': base, 'bank': members,
                'mu': mu, 'sigma': sigma}, policy['Shared'])
            dest = OUT/'banks'/task/str(tid)/old.name
            source.npwrite(dest, Base=base, members=members, Canonical_native8=native8,
                FrozenFusion=frozen, source_mean=mu, source_scale=sigma)
            atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
                {'family': meta['family'], 'rho': meta['rho'], 'output_sha256': source.sha(dest),
                'original_input_sha256': c['input_sha256'], 'parent_bank': str(old),
                'parent_bank_sha256': source.sha(old), 'generic_indices': INDICES[task],
                'context_rows': [c['source_rows']]+[meta['context_rows'][j] for j in INDICES[task]],
                'task_query_outcomes_opened': False, 'Native1_and_Native8_both_present': True})
            count += 1
    assert count == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'conditions': count})
    print('DUAL_CANONICAL_BANK_COMPLETE', count, flush=True)


if __name__ == '__main__':
    main()
