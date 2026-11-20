"""Bounded source-support Native gate and calibrated IQR retention count.

All original variants remain available. Refinement never uses historical Shared
bounds, and never changes the frozen benchmark or the source access budget.
"""
import argparse
import json
import numpy as np
from prepare_data import RUN,atomic_json,protocol
from internal_regression_benchmark import OUT
from run_internal_regression_ensembles import jobs,load_condition
from fit_internal_regression_ensembles import load,clean,feasible
from fit_combiners import sample_weights
from run_tabicl import metrics


def geometry(track,shard,shards):
    for _,m,f,r in [j for j in jobs() if j[0]==track][shard::shards]:
        out=OUT/'source_geometry'/track/str(m['task_id'])/f'{f}_{r}.json'
        if out.exists():continue
        data,_,_=load_condition(track,m,f,r);X=data['X_source'];Q=data['X_target']
        eps=np.maximum(X.std(0)*1e-6,1e-8)
        outside=(Q<X.min(0)-eps)|(Q>X.max(0)+eps)
        score=float(np.mean(outside))
        original=json.loads((OUT/'ensemble_banks'/track/str(m['task_id'])/f'{f}_{r}.json').read_text())
        hashes={k:protocol.array_sha256(data[k]) for k in ['X_source','X_target']}
        assert all(original['input_hashes'][k]==v for k,v in hashes.items())
        atomic_json(out,{'source_support_violation_fraction':score,'source_only_fitted_boundaries':True,
            'query_labels_used':False,'input_hashes':hashes,'rows':len(Q)})
    print('GEOMETRY_FINISHED',track,shard,flush=True)


def predict(c,method,model):
    if method=='Native6_support_gate':
        return c['native_quantile'] if c['support_score']<=model['threshold'] else c['native_raw']
    ids=np.argsort(c['iqr'].mean(1),kind='stable')[:model['keep']]
    raw=c['bootstrap'][ids].mean(0)
    return (1-model['alpha'])*c['base']+model['alpha']*raw


def loss(cells,method,model):
    # Both mean standardized MSE (R2-related) and mean standardized RMSE.
    total=0.
    for c,w in zip(cells,sample_weights(cells)):
        error=(predict(c,method,model)-c['y'])/c['meta']['scale'];nmse=float(np.mean(error**2))
        total+=float(w.sum())*(.5*nmse+.5*np.sqrt(nmse))
    return total


def clean_scores(cells,method,model):
    values=[metrics(c['y'],predict(c,method,model),'regression',c['meta']['scale']) for c in cells if c['meta']['rho']==0]
    return {k:float(np.mean([r[k] for r in values])) for k in ['r2','nrmse']}


def fit(track,fold):
    cells,source=load(track)
    for c in cells:
        p=OUT/'source_geometry'/track/str(c['meta']['task_id'])/c['path'].with_suffix('.json').name
        c['support_score']=json.loads(p.read_text())['source_support_violation_fraction']
    training=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
    train_ids={int(c['meta']['data_id']) for c in training};test_ids={int(c['meta']['data_id']) for c in test}
    assert train_ids.isdisjoint(test_ids);baseline=clean(training)
    for method in ['Native6_support_gate','IQR6_calibrated_keep']:
        out=OUT/'refined_policies'/track/f'fold_{fold}_{method}.json'
        if out.exists():continue
        options=([{'threshold':v} for v in [-1.,0.,.005,.01,.025,.05,.1,1.]] if method=='Native6_support_gate'
                 else [{'keep':k,'alpha':a} for k in [1,3,6] for a in [0.,.25,.5,1.]])
        trials=[]
        for model in options:
            observed=clean_scores(training,method,model);value=loss(training,method,model)
            trials.append({'model':model,'Clean':observed,'loss':value,'Clean_feasible':feasible(observed,baseline)})
        valid=[r for r in trials if r['Clean_feasible']]
        selected=min(valid or trials,key=lambda r:r['loss']);model=selected['model'];rows=[]
        for c in test:
            p=predict(c,method,model);values=metrics(c['y'],p,'regression',c['meta']['scale'])
            row={k:c['meta'][k] for k in ['task_id','data_id','dataset','family','rho']}
            rows.append(row|{'method':method,'fold':fold}|values)
            dest=OUT/'refined_predictions'/track/method/str(c['meta']['task_id'])/c['path'].name
            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=p,y_target=c['y']);tmp.replace(dest)
        atomic_json(out,{'status':'COMPLETE','track':track,'method':method,'fold':fold,'seed':0,'model':model,
            'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),'fold_map_source':source,
            'calibration_uses_other_four_dataset_folds':True,'method_has_no_additional_continuous_fitted_parameters':True,
            'test_labels_used_for_fit':False,'Shared_bounds_used_for_fit':False,'benchmark_changed':False,
            'new_GPU_inference':False,'deployment_context_count':6,'source_support_gate_uses_only_X_and_unlabelled_Q':True,
            'batch_level_gate_one_native_preprocessing_branch_deployed':True,'Clean_feasible':bool(valid),
            'original_variants_retained':True,'internal_only':True,'trials':trials,'rows':rows})
        print('REFINED_METHOD_DONE',track,fold,method,model,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['geometry','fit']);p.add_argument('--track',choices=['five','seven'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.phase=='geometry':geometry(a.track,a.shard,a.shards)
    else:
        assert a.fold in range(5);fit(a.track,a.fold)
