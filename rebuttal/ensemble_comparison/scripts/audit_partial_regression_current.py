"""Verify fixed partial resampling, OOF shrinkage and saved regression metrics."""
import hashlib
import json
import math
import numpy as np
from prepare_data import RUN,atomic_json
from fit_combiners import fold_mapping
from run_canonical_regression import fingerprint
from run_tabicl import metrics


def main():
    root=RUN/'results/canonical_partial_010'
    summary=json.loads((RUN/'results/canonical_partial_regression_current_summary.json').read_text())
    mapping,_=fold_mapping('regression','seven_calibrated_v2')
    policies={i:json.loads((root/'shrinkage025/policies'/f'fold_{i}.json').read_text()) for i in range(5)}
    for p in policies.values():
        assert set(p['train_data_ids']).isdisjoint(p['test_data_ids'])
        assert not p['outer_test_labels_used_for_fit_or_selection']
        assert not p['paper_Shared_or_requested_score_bounds_used_for_selection']
    expected=[];uniques=[]
    for i in range(6):
        rng=np.random.default_rng(1009*(i+1));ids=np.arange(256)
        positions=rng.choice(256,size=26,replace=False)
        ids[positions]=rng.choice(256,size=26,replace=True)
        expected.append(fingerprint(ids));uniques.append(len(np.unique(ids)))
    rows=[];checked=0
    paths=sorted((root/'tabicl/seven_calibrated_v2/regression').glob('*/*.npz'))
    assert len(paths)==957
    for path in paths:
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['context_rows']==256 and meta['candidate_count']==6 and meta['resampling_fraction']==.1
        assert meta['bootstrap_indices_sha256']==expected
        baseline=RUN/'results/canonical_input/tabicl/seven_calibrated_v2/regression'/str(meta['task_id'])/path.name
        bm=json.loads(baseline.with_suffix('.json').read_text())
        for k in ['source_X_sha256','source_y_sha256','query_X_sha256']:
            assert meta[k]==bm[k]
        with np.load(path) as f:
            bank,y=f['bootstrap_bank'],f['y_target']
            assert np.array_equal(bank.mean(0),f['prediction__bootstrap_mean_6'])
        with np.load(baseline) as f:
            assert np.array_equal(y,f['y_target'])
        p=policies[mapping[meta['data_id']]]
        assert meta['data_id'] in p['test_data_ids']
        for name,beta in p['weights'].items():
            with np.load(root/'shrinkage025/predictions'/name/str(meta['task_id'])/path.name) as f:
                pred,truth=f['prediction'],f['y_target']
            assert np.array_equal(y,truth)
            assert np.max(np.abs(pred-np.asarray(beta)@bank))<1e-10
            observed=metrics(y,pred,'regression',max(float(y.std()),1e-12))
            rows.append(meta | {'method':name} | observed);checked+=1
    assert checked==1914
    for result in summary['averages']:
        if result['method'].startswith('Random'):continue
        for block,nonzero,count in [('clean',False,33),('seven',True,924)]:
            selected=[r for r in rows if r['method']==result['method'] and (r['rho']>0)==nonzero]
            assert len(selected)==count
            for k in ['r2','nrmse']:
                assert abs(math.fsum(r[k] for r in selected)/count-result[block][k])<1e-10
    assert all(r['Clean_within_005'] and r['seven_both_metrics_better_than_Base']
               and r['seven_both_metrics_worse_than_historical_Shared'] for r in summary['averages'])
    atomic_json(RUN/'results/canonical_partial_regression_current_audit.json',{
        'status':'PASS','source_conditions_checked':957,'saved_fusion_predictions_verified':1914,
        'folds_with_disjoint_calibration_test_datasets':5,'seed':0,'contexts':6,'source_rows':256,
        'same_canonical_inputs_and_queries_as_previous_control':True,
        'fixed_replacement_positions':26,'unique_source_rows_per_member':uniques,
        'all_three_current_controls_meet_Clean_and_reported_ordering':True,
        'ordering_bounds_not_used_for_selection':True,'Shared_run':False,
        'source_summary_sha256':hashlib.sha256((RUN/'results/canonical_partial_regression_current_summary.json').read_bytes()).hexdigest()})
    print('PASS',len(paths),checked,uniques,flush=True)


if __name__=='__main__':main()
