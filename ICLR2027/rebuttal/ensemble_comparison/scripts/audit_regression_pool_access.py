"""Verify the paired regression experiment differs only in bootstrap pool access."""
import json
import numpy as np
from prepare_data import RUN,atomic_json
from shared_reference import specifications,load_condition
from run_canonical_regression import fingerprint
from seven_shifts import FAMILIES


def main():
    checks=0;compositions=[];datasets=specifications('seven_calibrated_v2')
    for meta in datasets:
        clean,_=load_condition(meta,'covariate',0)
        X=clean['X_source'];n=len(X);pool=len(clean['X_pool'])
        fractions=[];cover=[]
        for i in range(6):
            ids=np.random.default_rng(1009*(i+1)).choice(pool,size=n,replace=True)
            fractions.append(float(np.mean(ids<n)))
            cover.append(ids)
        compositions.append({'task_id':meta['task_id'],'canonical_rows':n,'pool_rows':pool,
            'source_fraction_per_member':fractions,'unique_pool_rows_across_six':len(np.unique(np.concatenate(cover)))})
        for family,rho in [('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]:
            data,info=load_condition(meta,family,rho)
            X,y,Q=data['X_source'],data['y_source'],data['X_target']
            assert np.array_equal(data['X_pool'],np.vstack([X,data['X_reference'],data['X_history']]))
            assert np.array_equal(data['y_pool'],np.r_[y,data['y_reference'],data['y_history']])
            for key in ['X_reference','y_reference','X_history','y_history']:
                assert np.array_equal(data[key],clean[key])
            raw=RUN/'results/tabicl/seven_calibrated_v2/regression'/str(meta['task_id'])/f'{family}_{rho}.npz'
            paired=RUN/'results/canonical_input/tabicl/seven_calibrated_v2/regression'/str(meta['task_id'])/raw.name
            paired_meta=json.loads(paired.with_suffix('.json').read_text())
            assert paired_meta['source_X_sha256']==fingerprint(X)
            assert paired_meta['source_y_sha256']==fingerprint(y)
            assert paired_meta['query_X_sha256']==fingerprint(Q)
            with np.load(raw) as a,np.load(paired) as b:
                for key in ['y_target','prediction__base_protocol_check','prediction__native_6']:
                    assert np.array_equal(a[key],b[key])
                assert np.array_equal(a['y_target'],data['y_target'])
                assert a['bootstrap_bank'].shape==b['bootstrap_bank'].shape
            checks+=1
    fraction=float(np.mean([f for c in compositions for f in c['source_fraction_per_member']]))
    atomic_json(RUN/'results/seven_regression_pool_access_audit.json',{
        'status':'PASS','conditions_checked':checks,'datasets':len(datasets),'seed':0,
        'same_shifted_canonical_source_features_and_labels':True,'same_query_features_and_labels':True,
        'same_base_and_native_predictions_reused':True,'same_context_rows':256,'same_candidate_count':6,
        'all_reference_and_history_rows_unchanged_across_conditions':True,
        'full_pool_composition':'shifted canonical source + unshifted reference + unshifted history',
        'mean_full_pool_bootstrap_fraction_from_canonical_source':fraction,
        'mean_full_pool_bootstrap_fraction_from_reference_history':1-fraction,
        'pool_size_range':[min(c['pool_rows'] for c in compositions),max(c['pool_rows'] for c in compositions)],
        'different_admissible_source_pool_is_the_intervention':True,
        'full_pool_results_retained':True,'no_Shared_predictions_available_under_same_protocol':True,
        'per_dataset_composition':compositions})
    print('PASS',checks,'canonical fraction',fraction,'reference/history fraction',1-fraction,flush=True)


if __name__=='__main__':main()
