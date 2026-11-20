"""Full 48-check numerical audit; raw prediction verification remains separate."""
import argparse
import json
import audit_tabace_goals_v28 as audit
from prepare_data import RUN,atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args()
    assert a.stage in ['tabace_recovery_contexts_v44','tabace_calibration_head_v45','tabace_clean_role_head_v46','tabace_sparse_specific_v47',
                       'tabace_probability_head_v51','tabace_meta_calibration_v52',
                       'tabace_source_expert_v53','tabace_augmented_classifier_v54','tabace_bagged_specific_v56',
                       'tabace_classification_risk_v58','tabace_clean_calibration_v59','tabace_prior_fusion_v61',
                       'tabace_weighted_gate_v62','tabace_hierarchical_specific_v64','tabace_equivariant_specific_v65',
                       'tabace_balanced_seven_v66','tabace_native_context_specific_v67',
                       'tabace_balanced_clean_calibration_v71']
    audit.main.OUT=RUN/'results'/a.stage;audit.report()
    path=audit.main.OUT/'goal_audit.json';row=json.loads(path.read_text())
    row.update(methods='Adapted development variants, with extra clean-TRAIN supervision where declared',
        original_paper_replication=False,query_feedback_used_for_benchmark_design=True,
        audit_scope='All 48 paired mean checks from saved metric records; raw arrays require separate verification',
        numerical_audit_is_not_goal_completion=True)
    atomic_json(path,row)
