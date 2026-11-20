"""Frozen TabICL strategy comparisons on a new internal-only benchmark.

Benchmark difficulty was calibrated with development labels; neither this
script nor its results are a confirmatory replication of the paper protocol.
Existing method policies remain frozen, and every observed outcome is retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import warnings

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from prepare_data import ROOT,RUN,atomic_json,protocol
# Keep the installed TabICL package used by baseline calibration.
from run_tabicl import Predictor,metrics,normalize,preserving_bootstrap
from internal_seven_fixed_strength import OUT,SELECTED,make_condition,cells

MAIN=['Base','Native6','Random6_adapted','DENSE6_adapted','GES6_adapted',
      'Convex6_adapted','Loglinear6_adapted','FeatureNNLS6_adapted','DivideConquer_adapted']


def policies(fold,data_id):
    result={}
    for method in ['Native','GES','Convex','Loglinear']:
        p=RUN/'results/robust_six_combiners_stable/seven_reimplemented/classification'/f'fold_{fold}_{method}.json'
        d=json.loads(p.read_text());assert data_id in d['test_data_ids'] and data_id not in d['train_data_ids']
        result[method]=d
    for method in ['Random','DENSE']:
        p=RUN/'results/moderate_random_dense_subspace50/combiners/seven_reimplemented'/f'fold_{fold}_{method}.json'
        d=json.loads(p.read_text());assert data_id in d['test_data_ids'] and data_id not in d['calibration_data_ids']
        result[method]=d
    p=RUN/'results/combiners/seven_reimplemented/classification'/f'fold_{fold}_feature_nnls.json'
    d=json.loads(p.read_text());assert data_id in d['test_data_ids'] and data_id not in d['train_data_ids']
    result['FeatureNNLS']=d
    return result


def moderate_bank(raw,adapt,pattern):
    if pattern==0:return raw
    if pattern==1:return np.concatenate([raw[:5],adapt[1:2]])
    if pattern==2:return np.concatenate([raw[:5],adapt[3:4]])
    if pattern==3:return np.concatenate([raw[:4],adapt[1:2],adapt[3:4]])
    if pattern==4:return np.concatenate([raw[:5],adapt[4:5]])
    if pattern==5:return np.concatenate([raw[:4],adapt[3:5]])
    raise ValueError(pattern)


def run(shard,shards,limit):
    import torch,tabicl
    from seven_shifts import classification_clean
    from robust_six_views import construct,density_and_proxy
    from fit_robust_six import predict as frozen_predict,dense_outputs
    from fit_moderate_random_dense import outputs
    from dense_full_context import candidates as dense_candidates
    from run_feature_views import projections
    from fit_feature_nnls import predict as feature_predict
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    frozen=json.loads(SELECTED.read_text());assert frozen['status']=='FROZEN'
    model=Predictor('classification','cuda:0');clean_cache={};policy_cache={}
    jobs=cells()[shard::shards]
    if limit:jobs=jobs[:limit]
    for meta,family,rho in jobs:
        out=OUT/'ensembles'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        tick=time.perf_counter()
        if meta['task_id'] not in clean_cache:clean_cache[meta['task_id']]=classification_clean(meta)
        data,info=make_condition(meta,family,rho,frozen,clean_cache[meta['task_id']])
        X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
        base_path=OUT/'base'/frozen['candidate']/str(meta['task_id'])/out.name
        with np.load(base_path) as z:base=z['prediction'];assert np.array_equal(t,z['y_target'])
        bm=json.loads(base_path.with_suffix('.json').read_text())
        hashes={k:protocol.array_sha256(data[k]) for k in ['X_source','y_source','X_target','y_target']}
        if bm['input_hashes'] is not None:assert hashes==bm['input_hashes']
        else:
            old=RUN/'cache/seven_reimplemented/classification'/str(meta['task_id'])/out.name
            with np.load(old) as z:
                for k in hashes:assert np.array_equal(data[k],z[k])
        if meta['task_id'] not in policy_cache:
            policy_cache[meta['task_id']]=policies(int(meta['outer_fold']),int(meta['data_id']))
        ps=policy_cache[meta['task_id']]
        native,_=model(X,y,Q,members=6,n_classes=2)
        PX,py=data['X_pool'],data['y_pool'];raw=[]
        for i in range(6):
            ids=preserving_bootstrap(py,np.random.default_rng(1009*(i+1)),size=len(y))
            pred,_=model(PX[ids],py[ids],Q,n_classes=2);raw.append(pred)
        raw=np.stack(raw)
        views,diagnostics=construct(X,y,Q,history=(data['X_history'],data['y_history']),
                                  recent=(data['X_reference'],data['y_reference']),force_views=True)
        adapt=[base];densities=[];proxy=[]
        for i,(_,A,b,T) in enumerate(views):
            density,reliability,_,_=density_and_proxy(A,b,T)
            densities.append(density);proxy.append(float(reliability*reliability*density.mean()))
            if i:
                ids=preserving_bootstrap(b,np.random.default_rng(1009*(i+1)),size=len(y))
                pred,_=model(A[ids],b[ids],T,n_classes=2,random_state=i);adapt.append(pred)
        adapt=np.stack(adapt);density=np.stack(densities)
        c={'meta':{'view_diagnostics':diagnostics},'base':base,'native_raw':native,
           'native_adapt':native,'bank_raw':raw,'bank_adapt':adapt,
           'dense_raw':dense_candidates(raw,X,y,Q,PX,py),'dense_adapt':dense_outputs(adapt,density),
           'native_base_backoff':True}
        # The existing Native policy selects raw in all five folds. Guard any
        # future incompatible policy instead of silently using raw predictions.
        assert ps['Native']['selected_mode']==0
        preds={'Base':base,'Native6':native}
        for method in ['Random','DENSE']:
            d=ps[method];bank=moderate_bank(raw,adapt,d['selected_pattern'])
            preds[f'{method}6_adapted']=outputs(bank)[d['selected_rule']]
        for method in ['GES','Convex','Loglinear']:
            d=ps[method];preds[f'{method}6_adapted']=frozen_predict(c,method,d['selected_mode'],d['model'])
        feature=[];feature_names=[]
        for name,A,T in projections(X,Q):
            pred=base if 'identity' in name else model(A,y,T,n_classes=2)[0]
            feature.append(pred);feature_names.append(name)
        feature=np.stack(feature);d=ps['FeatureNNLS']
        preds['FeatureNNLS6_adapted']=feature_predict({'bank':feature},np.asarray(d['weights']),'classification',d['temperature'])
        count=min(6,max(1,int(np.ceil(X.shape[1]/500))))
        if count==1:preds['DivideConquer_adapted']=base
        else:
            bank=[]
            for i in range(count):
                cols=np.random.default_rng(5003+i).choice(X.shape[1],min(X.shape[1],500),replace=False)
                bank.append(model(X[:,cols],y,Q[:,cols],n_classes=2)[0])
            preds['DivideConquer_adapted']=normalize(np.mean(bank,axis=0))
        preds['Random6_raw_diagnostic']=normalize(raw.mean(0))
        preds['FeatureMean6_raw_diagnostic']=normalize(feature.mean(0))
        assert all(np.isfinite(p).all() and p.shape==(len(t),2) for p in preds.values())
        observed={name:metrics(t,p,'classification',1.) for name,p in preds.items()}
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,y_target=t,raw_bank=raw,adapted_bank=adapt,feature_bank=feature,
                            **{'prediction__'+name:p for name,p in preds.items()});tmp.replace(out)
        atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','methods':observed,
           'input_hashes':hashes,'source_pool_rows':len(py),'rows_per_context':len(y),'logical_bank_members':6,
           'all_method_policies_frozen_from':'existing seven_reimplemented dataset-OOF policies',
           'method_parameters_refit_on_new_benchmark':False,'evaluated_dataset_labels_used_for_method_fit':False,
           'benchmark_itself_calibrated_using_development_labels':True,'Shared_run':False,
           'TabICL_library':tabicl.__file__,'feature_roles':feature_names,'adaptation_roles':[v[0] for v in views],
           'view_diagnostics':diagnostics,'seconds':time.perf_counter()-tick,
           'cached_multiple_method_banks_computed_not_six_total_calls_for_entire_experiment':True})
        print('ENSEMBLES_DONE',meta['task_id'],family,rho,flush=True)
    print('WORKER_FINISHED',shard,flush=True)


def report():
    original=json.loads((RUN/'results/final_ensemble_summary.json').read_text())['records']
    classification=json.loads((RUN/'results/classification_adaptation_summary.json').read_text())['primary']
    moderate=json.loads((RUN/'results/random_dense_current_summary.json').read_text())['current']
    lookup={'Base':next(r for r in original if r['task']=='classification' and r['backbone']=='tabicl' and r['label']=='Base')}
    rows=[];complete_datasets=0;by_method={name:[] for name in MAIN}
    for folder in sorted((OUT/'ensembles').iterdir()):
        if not folder.is_dir():continue
        paths=list(folder.glob('*.json'))
        if len(paths)!=28:continue
        complete_datasets+=1
        for p in paths:
            d=json.loads(p.read_text())
            for name in MAIN:by_method[name].append(d['methods'][name])
    names={'Native6':'Native','Random6_adapted':'Random','DENSE6_adapted':'DENSE','GES6_adapted':'GES',
           'Convex6_adapted':'Convex','Loglinear6_adapted':'Loglinear'}
    for name in MAIN:
        if name=='Base':old=lookup['Base'];clean=old['clean']['metrics'];five=old['five']['metrics'];seven_old=old['seven']['metrics']
        elif name in ['Random6_adapted','DENSE6_adapted']:
            old=next(r for r in moderate if r['method']==names[name]);clean=old['clean'];five=old['five'];seven_old=old['seven']
        elif name in names:
            old=next(r for r in classification if r['method']==names[name]);clean=old['clean'];five=old['five'];seven_old=old['seven']
        else:
            old=next(r for r in original if r['task']=='classification' and r['backbone']=='tabicl'
                     and (r['method']=='TabFM_feature_views6_NNLS_adapted' if name=='FeatureNNLS6_adapted' else r['method']=='divide_conquer_feature_adapted'))
            clean=old['clean']['metrics'];five=old['five']['metrics'];seven_old=old['seven']['metrics']
        values={k:float(np.mean([d[k] for d in by_method[name]])) for k in ['accuracy','auc']} if complete_datasets else None
        rows.append({'method':name,'Clean_frozen':clean,'five_frozen':five,'seven_internal':values,'seven_previous':seven_old})
    result={'status':'COMPLETE' if complete_datasets==35 else 'RUNNING','complete_datasets':complete_datasets,
            'expected_datasets':35,'condition_files':len(list((OUT/'ensembles').glob('*/*.json'))),
            'records':rows,'benchmark':json.loads(SELECTED.read_text()),'internal_only':True,
            'existing_Clean_five_modified':False,'Shared_run':False,'Markdown_written':False}
    atomic_json(OUT/'latest_ensemble_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',action='store_true');p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.report:report()
    else:run(a.shard,a.shards,a.limit)
