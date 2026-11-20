"""Source-CV guard for the two conditional-repair alternatives.

Only a positive paired source loss gain enables a repair. Otherwise its
deployed output explicitly falls back to original Base. Raw repair outputs
are retained for provenance. The other five alternatives never change.
"""
import json

import numpy as np

import shared_feature_forest_banks_v227 as parent
import source_proxy_repair_features_v234 as quality
from prepare_data import RUN, atomic_json

source = parent.source
OUT = RUN/'results/shared_source_guarded_forest_v236'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert json.loads((quality.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((parent.OUT/'independent_membership_audit.json').read_text())['status'] == 'PASS'
    plan = {'status': 'SOURCE_PROXY_CV_GUARDED_REPAIR_NOT_ORIGINAL_SHARED', 'seed': 0,
        'context_cap': 256, 'predictor_outputs_including_Base': 8,
        'all_five_generic_parent_alternatives_retained': True,
        'guard': 'repair enabled iff source-proxy mean loss is strictly below observed-source mean loss',
        'guard_threshold': 0., 'guard_threshold_tuned_on_own_query_labels': False,
        'local_source_training_labels_used': True, 'extra_true_label_rows': 0,
        'blocked_repair_falls_back_to_original_Base_explicitly': True,
        'raw_repair_predictions_preserved': True, 'query_task_labels_opened': False,
        'maximum_query_TFM_estimator_views': 15, 'source_proxy_CV_TFM_views': 9,
        'extra_parent_source_CV_views': {'classification': 0, 'regression': 3},
        'compute_matched_to_Native1_Base': False, 'backbone_weights_updated': False,
        'Base_and_benchmark_unchanged': True, 'new_TFM_inference_in_this_assembly': False,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'parent_plan_sha256': source.sha(parent.OUT/'plan.json'),
        'source_quality_plan_sha256': source.sha(quality.OUT/'plan.json'),
        'script_sha256': source.sha(__file__), 'expected_conditions': 1327, 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)
    count = 0
    enabled_counts = [0, 0]
    for old in sorted((parent.OUT/'banks').glob('*/*/*.npz')):
        meta = json.loads(old.with_suffix('.json').read_text())
        assert source.sha(old) == meta['output_sha256']
        qpath = quality.OUT/'features'/old.relative_to(parent.OUT/'banks')
        qmeta = json.loads(qpath.with_suffix('.json').read_text())
        assert source.sha(qpath) == qmeta['output_sha256']
        assert qmeta['original_input_sha256'] == meta['original_input_sha256']
        assert qmeta['feature_bank_sha256'] == meta['output_sha256']
        with np.load(qpath) as q:
            gains = q['source_proxy_gain']
        assert gains.shape == (2,) and np.isfinite(gains).all()
        enabled = gains > 0.
        enabled_counts = [a+int(b) for a, b in zip(enabled_counts, enabled)]
        with np.load(old) as data:
            members = data['members'].copy()
            raw = members[5:].copy()
            assert len(members) == 7
            for i, active in enumerate(enabled):
                if not active:
                    members[5+i] = data['Base']
            dest = OUT/'banks'/old.relative_to(parent.OUT/'banks')
            source.npwrite(dest, Base=data['Base'], members=members, RawRepair_predictions=raw,
                Canonical_native8=data['Canonical_native8'], FrozenFusion=data['FrozenFusion'],
                source_mean=data['source_mean'], source_scale=data['source_scale'])
        atomic_json(dest.with_suffix('.json'), {k: meta[k] for k in ('task', 'task_id', 'data_id', 'fold', 'family', 'rho', 'original_input_sha256')} |
            {'output_sha256': source.sha(dest), 'parent_bank': str(old), 'parent_bank_sha256': meta['output_sha256'],
                'source_quality_file': str(qpath), 'source_quality_sha256': qmeta['output_sha256'],
                'source_proxy_loss_gains': gains.tolist(), 'repair_enabled': enabled.tolist(),
                'five_parent_alternatives_unchanged': True, 'query_task_labels_opened': False,
                'fallback_alias_is_explicit_Not_a_new_TFM_forward': True})
        count += 1
    assert count == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'conditions': count, 'repair_enabled_counts': enabled_counts})
    print('SOURCE_GUARDED_FOREST_BANK_COMPLETE', count, enabled_counts, flush=True)


if __name__ == '__main__':
    main()
