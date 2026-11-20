"""Predeclared nested64 target labels; remaining64 never enter fitting APIs."""
import json
import numpy as np
import nonlinear_sensor_schedule_v383 as schedule
from prepare_data import RUN, atomic_json, protocol

source = schedule.source
OUT = RUN/'results/nonlinear_sensor_nested_labels_v426'


def prepare():
    original = schedule.endpoint_cases()
    assert len(original)==32
    canonical = json.loads((source.OUT/'case_manifest.json').read_text())
    by_id = {(c['track'],c['task'],c['task_id']):c for c in canonical}
    cases = []
    for c in original:
        with np.load(c['adaptation_file']) as z:
            a,e,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        assert len(a)==32 and len(e)==96
        newa,newe=np.r_[a,e[:32]],e[32:]
        assert len(newa)==len(newe)==64 and set(newa).isdisjoint(newe)
        truth=by_id[(c['track'],c['task'],c['task_id'])]
        assert source.sha(truth['truth_file'])==truth['truth_sha256']
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(truth['truth_file']) as z:y=z['y_target']
        with np.load(c['input_file']) as z:ids=z['target_indices']
        np.testing.assert_array_equal(ay,y[a])
        dest=OUT/'splits'/str(c['task_id'])
        adapt,held=dest/'adaptation.npz',dest/'evaluation_truth.npz'
        if not adapt.exists():
            source.npwrite(adapt,adaptation_positions=newa,evaluation_positions=newe,
                           adaptation_y=y[newa],adaptation_indices=ids[newa],evaluation_indices=ids[newe])
            source.npwrite(held,y_target=y[newe],target_indices=ids[newe])
        with np.load(adapt) as z:
            for key,v in [('adaptation_positions',newa),('evaluation_positions',newe),('adaptation_y',y[newa]),
                          ('adaptation_indices',ids[newa]),('evaluation_indices',ids[newe])]:
                np.testing.assert_array_equal(z[key],v)
        with np.load(held) as z:
            np.testing.assert_array_equal(z['y_target'],y[newe])
            np.testing.assert_array_equal(z['target_indices'],ids[newe])
        cases.append(dict(c,adaptation_file=str(adapt),adaptation_sha256=source.sha(adapt),
                          heldout_truth_file=str(held),heldout_truth_sha256=source.sha(held),
                          previous_adaptation_sha256=c['adaptation_sha256']))
    manifest=OUT/'case_manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==cases
    else:
        atomic_json(manifest,cases)
    plan={'active_endpoint':['shift','classification','nonlinear_sensor'],'seed':0,
          'datasets':32,'conditions':128,'target_adaptation_rows':64,'evaluation_rows':64,
          'nested_split':'original32 adaptation rows, then the first32 positions of the already fixed v311 evaluation permutation; remaining64 evaluation positions',
          'split_uses_no_label_values':True,'original32_adaptation_rows_retained':True,
          'added_target_labels_from_original_test_subset':32,
          'method_fitting_receives_only_adaptation_labels':True,
          'reference_and_TFM_features_and_backbone_and_benchmark_unchanged':True,
          'new_Base_is_original_predictions_on_the_same_remaining64_rows':True,
          'not_comparable_to_old96_row_means_without_this_protocol_annotation':True,
          'previously_scored_development_cohort_not_blind_test':True,
          'manifest_sha256':source.sha(manifest),'script_sha256':source.sha(__file__),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text())==plan
    else:
        atomic_json(p,plan)


def endpoint_cases():
    p=OUT/'case_manifest.json'
    if not p.exists():
        prepare()
    return json.loads(p.read_text())


if __name__=='__main__':
    prepare()
