"""Frozen native, IQR-selection and official RF-PFN regression comparisons.

Canonical source inputs match Base, with no new history/reference samples.
The RF-PFN six-tree control records its additional node/validation calls.
"""
import argparse
import functools
import gc
import json
import logging
import random
import time
import traceback
import warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from run_tabpfn import model,fit_predict
from run_canonical_regression import fingerprint
from run_tabicl import metrics


def work():
    jobs=[]
    for track in ['five','seven_calibrated_v2']:
        families=protocol.FAMILIES
        if track.startswith('seven'):
            from seven_shifts import FAMILIES
            families=FAMILIES
        conditions=[('original' if track=='five' else 'covariate',0)]+[(f,r) for f in families for r in [20,50,80,100]]
        jobs.extend((m,f,r) for m in specifications(track) if m['task']=='regression' for f,r in conditions)
    return jobs


def quantile_prediction(est,X,y,Q):
    if np.std(y)<1e-12 or X.shape[1]==0 or np.all(np.ptp(X,axis=0)==0):
        return np.full(len(Q),float(np.mean(y))),np.full(len(Q),float(np.quantile(y,.85)-np.quantile(y,.15)))
    est.fit(X,y)
    output=est.predict(Q,output_type='main',quantiles=[.15,.85])
    q=np.asarray(output['quantiles'])
    if q.shape==(2,len(Q)):q=q.T
    assert q.shape==(len(Q),2),q.shape
    return np.asarray(output['mean']).reshape(-1),np.maximum(q[:,1]-q[:,0],0.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--component',choices=['pfn','rf','both'],default='both')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    from tabpfn import TabPFNRegressor
    counter={};phase=['setup']
    original_predict=TabPFNRegressor.predict
    @functools.wraps(original_predict)
    def counted_predict(self,X,*a,**kw):
        key=phase[0];counter[key]=counter.get(key,0)+int(self.n_estimators)
        return original_predict(self,X,*a,**kw)
    TabPFNRegressor.predict=counted_predict
    from tabpfn_extensions.rf_pfn.SklearnBasedRandomForestTabPFN import RandomForestTabPFNRegressor
    from tabpfn_extensions.rf_pfn.SklearnBasedDecisionTreeTabPFN import DecisionTreeTabPFNBase
    original_tree_predict=DecisionTreeTabPFNBase.predict_
    @functools.wraps(original_tree_predict)
    def counted_tree_predict(self,X,y=None,check_input=True):
        previous=phase[0];phase[0]='rf_source_validation' if y is not None else 'rf_query_nodes'
        try:return original_tree_predict(self,X,y=y,check_input=check_input)
        finally:phase[0]=previous
    DecisionTreeTabPFNBase.predict_=counted_tree_predict
    models={};failures=0;jobs=work()[args.shard::args.shards]
    if args.limit:jobs=jobs[:args.limit]
    for meta,family,rho in jobs:
        components=['pfn','rf'] if args.component=='both' else [args.component]
        for component in components:
            out=RUN/'results/regression_expansion'/component/meta['track']/'regression'/str(meta['task_id'])/f'{family}_{rho}.npz'
            if out.with_suffix('.json').exists():continue
            try:
                data,info=load_condition(meta,family,rho)
                X,y,Q=(data[k] for k in ['X_source','y_source','X_target'])
                old=RUN/'results/tabpfn'/meta['track']/'regression'/str(meta['task_id'])/out.name
                with np.load(old) as f:truth=f['y_target'];base=f['prediction__tabpfn_v3_base_check']
                assert np.array_equal(truth,data['y_target'])
                counter.clear();tick=time.perf_counter();predictions={};arrays={};extra={}
                if 'single' not in models:models['single']=model('regression','cuda:0')
                if component=='pfn':
                    if 'native' not in models:
                        models['native']=model('regression','cuda:0');models['native'].n_estimators=6
                    phase[0]='native6';start=time.perf_counter()
                    native=fit_predict(models['native'],X,y,Q,'regression',None)
                    extra['native_seconds']=time.perf_counter()-start
                    bank=[];iqrs=[];indices=[];phase[0]='ordinary_bootstrap6';start=time.perf_counter()
                    for i in range(6):
                        rows=np.random.default_rng(1009*(i+1)).choice(len(y),len(y),replace=True)
                        models['single'].random_state=0
                        pred,iqr=quantile_prediction(models['single'],X[rows],y[rows],Q)
                        bank.append(pred);iqrs.append(iqr);indices.append(fingerprint(rows))
                    bank,iqrs=np.stack(bank),np.stack(iqrs)
                    selected=int(np.argmin(iqrs.mean(axis=1)))
                    predictions={'Native_PFN6':native,'IQR6_keep1_TabPFN_adapted':bank[selected]}
                    arrays.update(bank=bank,iqr=iqrs)
                    extra.update(bootstrap_indices_sha256=indices,selected_IQR_member=selected,
                        bootstrap_seconds=time.perf_counter()-start,IQR_quantiles=[.15,.85],
                        IQR_role='Six-candidate IQR selection adaptation; not original 400-candidate OligoICP experiment',
                        native_candidate_count=6,IQR_candidate_count=6,IQR_retained_count=1)
                else:
                    random.seed(0);np.random.seed(0);torch.manual_seed(0)
                    forest=RandomForestTabPFNRegressor(tabpfn=models['single'],n_jobs=1,
                        n_estimators=6,max_depth=2,adaptive_tree=True,random_state=0,max_predict_time=-1)
                    phase[0]='rf_fit';start=time.perf_counter();forest.fit(X,y)
                    extra['rf_fit_seconds']=time.perf_counter()-start
                    phase[0]='rf_predict';start=time.perf_counter()
                    pred=np.asarray(forest.predict(Q)).reshape(-1)
                    extra.update(rf_predict_seconds=time.perf_counter()-start,tree_count=6,max_depth=2,
                        adaptive_tree=True,rf_role='Official RF-PFN algorithm with fixed six-tree, depth-two budget; node calls counted separately',
                        tree_node_counts=[int(t._tree.tree_.node_count) for t in forest.estimators_])
                    predictions={'RF_PFN6_depth2_adaptive':pred}
                    del forest
                scale=max(float(truth.std()),1e-12) if meta['track'].startswith('seven') else info['scale']
                scores={}
                for name,pred in predictions.items():
                    assert pred.shape==truth.shape and np.isfinite(pred).all()
                    arrays['prediction__'+name]=pred;scores[name]=metrics(truth,pred,'regression',scale)
                arrays.update(y_target=truth,base=base,source_y_mean=np.asarray(float(y.mean())))
                out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,**arrays);tmp.replace(out)
                atomic_json(out.with_suffix('.json'),info | {'status':'COMPLETE','seed':0,
                    'component':component,'backbone':'tabpfn','family':family,'rho':rho,
                    'rows_per_context':len(y),'source_access':'Canonical Base context only; no history/reference augmentation',
                    'source_X_sha256':fingerprint(X),'source_y_sha256':fingerprint(y),
                    'query_X_sha256':fingerprint(Q),'base_reused_from':str(old.relative_to(RUN)),
                    'target_labels_used_for_prediction_or_selection':False,'backbone_weights_frozen':True,
                    'score_ordering_bounds_used':False,'Shared_run':False,
                    'seconds':time.perf_counter()-tick,'backbone_forward_units':counter.copy(),
                    'methods':scores}|extra)
                print('EXPANSION_DONE',component,meta['track'],meta['task_id'],family,rho,counter,flush=True)
                gc.collect();torch.cuda.empty_cache()
            except Exception as exc:
                failures+=1
                atomic_json(RUN/'results/inference_failures'/f"reg_expand_{component}_{meta['track']}_{meta['task_id']}_{family}_{rho}.json",
                    {'error':repr(exc),'traceback':traceback.format_exc()})
                print('EXPANSION_FAILED',component,repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
