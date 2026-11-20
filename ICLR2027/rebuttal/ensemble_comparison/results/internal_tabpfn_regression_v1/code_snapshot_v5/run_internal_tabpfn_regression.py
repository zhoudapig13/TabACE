"""Frozen-checkpoint TabPFN ensemble banks and official-library references."""
import argparse
from contextlib import contextmanager
import functools
import fcntl
import json
import logging
import time
import warnings
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from prepare_data import RUN,atomic_json,protocol
from internal_tabpfn_regression import OUT,frozen_jobs,frozen_input
from run_tabicl import metrics


@contextmanager
def prediction_counter():
    # Count public prediction invocations, not undocumented internal forwards.
    from tabpfn import TabPFNRegressor
    original=TabPFNRegressor.predict;counter={'predict_api_calls':0,'sum_n_estimators_over_predict_calls':0}
    @functools.wraps(original)
    def counted(self,*args,**kwargs):
        counter['predict_api_calls']+=1;counter['sum_n_estimators_over_predict_calls']+=int(self.n_estimators)
        return original(self,*args,**kwargs)
    TabPFNRegressor.predict=counted
    try:yield counter
    finally:TabPFNRegressor.predict=original


def persist(path,data,info,arrays,extra):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,y_target=data['y_target'],**arrays);tmp.replace(path)
    atomic_json(path.with_suffix('.json'),info|{'status':'COMPLETE','input_hashes':{k:protocol.array_sha256(v) for k,v in data.items()},
        'backbone_weights_frozen':True,'backbone_parameter_finetuning':False,'model_seed':0,'Shared_run':False,
        'target_labels_used_for_fit':False,'canonical_context_rows':len(data['y_source']),
        'additional_history_reference_rows_used':False,'benchmark_parameters_already_frozen':True}|extra)


def banks(track,m,f,r,est):
    from run_tabpfn import model,fit_predict
    from run_feature_views import projections
    path=OUT/'ensemble_banks'/track/str(m['task_id'])/f'{f}_{r}.npz'
    if path.exists() and path.with_suffix('.json').exists():return
    data,info,base=frozen_input(track,m,f,r);X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
    tick=time.perf_counter();native=model('regression','cuda:0');native.n_estimators=6
    native_raw=fit_predict(native,X,y,Q,'regression',None)
    qt=QuantileTransformer(n_quantiles=min(64,len(y)),output_distribution='normal',random_state=0,subsample=None)
    native_quantile=fit_predict(native,qt.fit_transform(X),y,qt.transform(Q),'regression',None)
    feature=np.stack([base if 'identity' in name else fit_predict(est,A,y,T,'regression',None) for name,A,T in projections(X,Q)])
    ibank=[];iqrs=[];hashes=[]
    for i in range(6):
        ids=np.arange(len(y)) if i==0 else np.random.default_rng(1009*i).choice(len(y),size=len(y),replace=True)
        est.fit(X[ids],y[ids]);values=est.predict(Q,output_type='main',quantiles=[.15,.85])
        pred=np.asarray(values['mean'],dtype=float).reshape(-1)
        qs=values['quantiles'];spread=np.maximum(np.asarray(qs[1])-np.asarray(qs[0]),0.)
        if i==0:assert np.allclose(pred,base,rtol=1e-6,atol=1e-6),(track,m['task_id'],f,r)
        ibank.append(base if i==0 else pred);iqrs.append(np.asarray(spread).reshape(-1));hashes.append(protocol.array_sha256(ids))
    ibank=np.stack(ibank);iqrs=np.stack(iqrs);assert feature.shape==ibank.shape==iqrs.shape==(6,len(t))
    predictions={'Base':base,'Native6_raw':native_raw,'Native6_source_quantiles':native_quantile,
        'FeatureMean6_raw':feature.mean(0),'Median6_raw':np.median(feature,axis=0),
        'TrimmedMean6_raw':np.sort(feature,axis=0)[1:-1].mean(0),
        'IQR6_keep1_raw_adapted':ibank[int(np.argmin(iqrs.mean(1)))]}
    assert all(np.isfinite(p).all() for p in [feature,ibank,iqrs,*predictions.values()])
    persist(path,data,info,{'feature_bank':feature,'iqr_bank':ibank,'iqr':iqrs,
        'source_mean':np.asarray(float(y.mean())),'source_scale':np.asarray(max(float(y.std()),1e-12)),
        **{'prediction__'+k:v for k,v in predictions.items()}},
        {'methods':{k:metrics(t,p,'regression',info['scale']) for k,p in predictions.items()},'context_count':6,
         'bootstrap_indices_sha256':hashes,'IQR_bank_contains_Base':True,'IQR_original_400_to_40_evaluated':False,
         'adaptation_annotation':'† for source/context preprocessing and six-candidate IQR budget adaptation',
         'seconds':time.perf_counter()-tick})
    print('PFN_BANK_DONE',track,m['dataset'],f,r,flush=True)


def exclusive_reference(function):
    @functools.wraps(function)
    def wrapped(track,m,f,r,kind):
        lock=OUT/'reference_locks'/kind/track/str(m['task_id'])/f'{f}_{r}.lock'
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            try:return function(track,m,f,r,kind)
            finally:fcntl.flock(handle,fcntl.LOCK_UN)
    return wrapped


@exclusive_reference
def reference(track,m,f,r,kind):
    from run_tabpfn import model,fit_predict,AutoPostHocEnsemblePredictor,TabPFNBaseModelSource,TaskType
    from tabpfn_extensions.rf_pfn.SklearnBasedRandomForestTabPFN import RandomForestTabPFNRegressor
    path=OUT/'references'/kind/track/str(m['task_id'])/f'{f}_{r}.npz'
    if path.exists() and path.with_suffix('.json').exists():return
    data,info,base=frozen_input(track,m,f,r);X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
    tick=time.perf_counter();extra={}
    if kind=='PHE6':
        configs=[(f'seed_{i}',model('regression','cuda:0',seed=i)) for i in range(6)]
        est=AutoPostHocEnsemblePredictor(preset='default',max_time=30,task_type=TaskType.REGRESSION,
            ges_scoring_string='mse',device='cuda',bm_random_state=0,ges_random_state=0,
            tabpfn_base_model_source=TabPFNBaseModelSource.CUSTOM,custom_tabpfn_models=configs,
            max_models=6,n_repeats=1,holdout_fraction=.25,ges_n_iterations=25,ignore_pretraining_limits=True)
        pred=fit_predict(est,X,y,Q,'regression',None)
        extra={'reference_implementation':'tabpfn-extensions 0.0.4 official PHE',
            'adaptation_annotation':'†‡: controlled six-seed same-checkpoint portfolio; source holdout selection',
            'candidate_configurations':6,'native_estimators_per_configuration':1,'source_holdout_fraction':.25,
            'GES_iterations':25,'validation_time_limit_seconds':30,
            'same_as_unrestricted_default_PHE':False,'logical_ensemble_count':6,'budget_group':'six-configuration portfolio'}
    else:
        est=RandomForestTabPFNRegressor(tabpfn=model('regression','cuda:0'),n_jobs=1,n_estimators=6,
            random_state=0,adaptive_tree=True,max_depth=2,min_samples_split=100,min_samples_leaf=20,
            max_predict_time=-1)
        with prediction_counter() as counter:pred=fit_predict(est,X,y,Q,'regression',None)
        extra={'reference_implementation':'tabpfn-extensions 0.0.4 official RF-PFN regressor',
            'adaptation_annotation':'†‡: six trees, depth two, source-only adaptive pruning',
            'logical_tree_count':6,'max_depth':2,'min_samples_split':100,'min_samples_leaf':20,'adaptive_tree':True,
            'six_trees_do_not_imply_six_TFM_calls':True,'budget_group':'six trees, variable node contexts',
            'measured_call_counts':counter}
    assert np.isfinite(pred).all()
    persist(path,data,info,{'prediction':pred,'prediction__Base':base},extra|{
        'metrics':metrics(t,pred,'regression',info['scale']),'seconds':time.perf_counter()-tick})
    print('PFN_REFERENCE_DONE',kind,track,m['dataset'],f,r,flush=True)


if __name__=='__main__':
    import torch
    from run_tabpfn import model
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['banks','PHE6','RF_PFN6']);p.add_argument('--tracks',nargs='+',default=['five','seven'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--reverse',action='store_true',help='Reverse RF job order; per-condition locks prevent duplicate evaluation')
    a=p.parse_args();torch.set_num_threads(1);warnings.filterwarnings('ignore');logging.disable(logging.INFO)
    work=[j for j in frozen_jobs() if j[0] in a.tracks][a.shard::a.shards]
    if a.reverse:
        assert a.phase=='RF_PFN6'
        work.reverse()
    if a.limit:work=work[:a.limit]
    est=model('regression','cuda:0')
    for track,m,f,r in work:
        if a.phase=='banks':banks(track,m,f,r,est)
        else:reference(track,m,f,r,a.phase)
    print('PFN_ENSEMBLE_WORKER_FINISHED',a.phase,a.shard,flush=True)
