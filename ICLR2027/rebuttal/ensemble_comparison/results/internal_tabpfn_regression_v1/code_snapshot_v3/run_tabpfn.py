"""Official PHE/RF-PFN and explicitly unadapted components; no backbone training."""
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
RUN=Path(__file__).resolve().parents[1]
ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(RUN/'external/tabpfn_extensions_0_0_4'))
import numpy as np
import torch
from sklearn.cluster import KMeans
from tabpfn import TabPFNClassifier,TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.pfn_phe import (
    AutoPostHocEnsemblePredictor,TabPFNBaseModelSource,TaskType)
from tabpfn_extensions.rf_pfn.SklearnBasedRandomForestTabPFN import RandomForestTabPFNClassifier
from prepare_data import atomic_json,protocol
from run_tabicl import normalize,metrics,preserving_bootstrap

CLASS_CKPT=ROOT.parent/'PFN-Research/checkpoints/tabpfn3/tabpfn-v3-classifier-v3_default.ckpt'
REG_CKPT=RUN/'checkpoints/tabpfn-v3-regressor-v3_default.ckpt'
CONFIGS=[(.5,False),(.75,False),(1.,False),(.5,True),(.75,True),(1.,True)]


def model(task,device,seed=0,temperature=.9,balance=False):
    kwargs=dict(n_estimators=1,random_state=seed,device=device,
                model_path=str(CLASS_CKPT if task=='classification' else REG_CKPT),
                ignore_pretraining_limits=True,n_preprocessing_jobs=1)
    if task=='classification':kwargs.update(softmax_temperature=temperature,balance_probabilities=balance)
    return (TabPFNClassifier if task=='classification' else TabPFNRegressor)(**kwargs)


def aligned(est,X,task,classes):
    if task=='regression':return np.asarray(est.predict(X),dtype=float).reshape(-1)
    raw=est.predict_proba(X)
    output=np.zeros((len(X),classes))
    output[:,np.asarray(est.classes_,dtype=int)]=raw
    return normalize(output)


def fit_predict(est,X,y,Q,task,classes):
    if X.shape[1]==0 or np.all(np.ptp(np.asarray(X,dtype=np.float64),axis=0)==0):
        if task=='regression':return np.full(len(Q),float(np.mean(y)))
        prior=np.bincount(np.asarray(y,dtype=int),minlength=classes)/len(y)
        return normalize(np.broadcast_to(prior,(len(Q),classes)).copy())
    if task=='classification' and len(np.unique(y))==1:
        p=np.zeros((len(Q),classes));p[:,int(y[0])]=1.;return normalize(p)
    if task=='regression' and float(np.std(y))<1e-12:return np.full(len(Q),float(np.mean(y)))
    est.fit(X,y)
    return aligned(est,Q,task,classes)


def micp_no_capfn(est,X,y,Q,classes,budget=256,count_override=None):
    """MICP Eq.2, gamma=1, source-standardized distances; no trained CaPFN."""
    count=min(6,max(1,int(np.ceil(len(y)/budget)))) if count_override is None else count_override
    loc=X.mean(0);scale=np.maximum(X.std(0),1e-6)
    Z=(X-loc)/scale;query=(Q-loc)/scale
    router=KMeans(n_clusters=count,n_init=10,random_state=0).fit(Z)
    assignment=router.predict(query);out=np.zeros((len(Q),classes));supports=[]
    for k,center in enumerate(router.cluster_centers_):
        cluster=np.flatnonzero(router.labels_==k)
        nearest=np.argsort(np.sum((Z-center)**2,axis=1),kind='stable')
        ids=(np.random.default_rng(k).choice(cluster,budget,replace=False)
             if len(cluster)>budget else nearest[:min(budget,len(y))])
        # Preserve observed source classes, an explicit tabular protocol adaptation.
        for label in np.unique(y):
            if not np.any(y[ids]==label):
                candidate=next(i for i in nearest if y[i]==label)
                for position in range(len(ids)-1,-1,-1):
                    if np.sum(y[ids]==y[ids[position]])>1:
                        ids[position]=candidate;break
        supports.append(ids.tolist());selected=np.flatnonzero(assignment==k)
        if not len(selected):continue
        out[selected]=fit_predict(est,X[ids],y[ids],Q[selected],'classification',classes)
    return normalize(out),{'prompters':count,'gamma':1.,'support_budget':budget,
                           'support_indices':supports,'routing':assignment.tolist(),
                           'geometry':'source-standardized Euclidean, sklearn KMeans',
                           'original_CaPFN_evaluated':False}


def cell(task,data,meta,family,rho,out,device,rf):
    X0,y0,Q,truth=(data[x] for x in ['X_source','y_source','X_target','y_target'])
    if str(meta.get('track','')).startswith('seven') or meta.get('condition_materialized',False):X,y=X0,y0
    else:
        changed=protocol.manipulate_source_context(task,X0,y0,family=family,rho=rho/100,
            seed=1729+rho,shift_scale=6. if task=='classification' else 10.,label_scale=1.)
        X,y=changed.X_context,changed.y_context
    pool_X=data['X_pool'] if 'X_pool' in data else X
    pool_y=data['y_pool'] if 'y_pool' in data else y
    classes=meta.get('n_classes',2 if str(meta.get('track','')).startswith('seven') else len(np.unique(y0))) if task=='classification' else None
    estimates={};times={};notes=[]
    base=model(task,device)
    torch.cuda.synchronize();start=time.perf_counter()
    estimates['tabpfn_v3_base_check']=fit_predict(base,X,y,Q,task,classes)
    torch.cuda.synchronize();times['tabpfn_v3_base_check']=time.perf_counter()-start

    if meta.get('include_native_ensemble_baselines',False):
        for count in [6,8]:
            native=model(task,device);native.n_estimators=count
            torch.cuda.synchronize();start=time.perf_counter()
            estimates[f'tabpfn_native_{count}']=fit_predict(native,X,y,Q,task,classes)
            torch.cuda.synchronize();times[f'tabpfn_native_{count}']=time.perf_counter()-start

    # This is the bootstrap component ONLY, not full trained BETA.
    bank=[];start=time.perf_counter()
    for i in range(6):
        rng=np.random.default_rng(1009*(i+1))
        rows=(preserving_bootstrap(pool_y,rng,size=len(y)) if task=='classification'
              else rng.choice(len(pool_y),len(y),replace=True))
        base.random_state=0
        bank.append(fit_predict(base,pool_X[rows],pool_y[rows],Q,task,classes))
    torch.cuda.synchronize()
    estimates['tabpfn_bootstrap_mean_6']=np.mean(bank,axis=0)
    times['tabpfn_bootstrap_mean_6']=time.perf_counter()-start
    if task=='classification':
        estimates['BETA_bootstrap_component_only_no_adaptation']=estimates['tabpfn_bootstrap_mean_6']
        times['BETA_bootstrap_component_only_no_adaptation']=times['tabpfn_bootstrap_mean_6']
        # At N_source == context budget, every MICP prompter has the same support.
        # The frozen, no-CaPFN component therefore reduces to Base in this track.
        if len(pool_y)<=256:
            estimates['MixturePFN_MICP_no_CaPFN']=estimates['tabpfn_v3_base_check']
            times['MixturePFN_MICP_no_CaPFN']=times['tabpfn_v3_base_check']
            notes.append('MICP support budget 256 >= source size; equivalent full-source contexts; Base reused')
        else:
            torch.cuda.synchronize();start=time.perf_counter()
            estimates['MixturePFN_MICP_no_CaPFN'],routing=micp_no_capfn(base,pool_X,pool_y,Q,classes)
            torch.cuda.synchronize();times['MixturePFN_MICP_no_CaPFN']=time.perf_counter()-start
            notes.append('MICP gamma=1 with 256-row support on shared allowed source pool; no adapter training')

    configs=[(f'config_{i}',model(task,device,i,t,b)) for i,(t,b) in enumerate(CONFIGS)]
    typ=(TaskType.REGRESSION if task=='regression' else TaskType.BINARY if classes==2 else TaskType.MULTICLASS)
    phe=AutoPostHocEnsemblePredictor(preset='default',max_time=30,task_type=typ,
        ges_scoring_string='log_loss' if task=='classification' else 'mse',
        device='cuda',bm_random_state=0,ges_random_state=0,
        tabpfn_base_model_source=TabPFNBaseModelSource.CUSTOM,custom_tabpfn_models=configs,
        max_models=6,n_repeats=1,holdout_fraction=.25,ges_n_iterations=25,ignore_pretraining_limits=True)
    torch.cuda.synchronize();start=time.perf_counter()
    constant=(len(np.unique(y))==1)
    singleton=(task=='classification' and not constant and int(np.unique(y,return_counts=True)[1].min())<2)
    if singleton:
        notes.append('PHE singleton-safe source holdout: reserve each observed class in training; disjoint validation may omit rare classes; complete-label log loss')
    if constant:
        estimates['PHE_official_custom6_native_source_calibration']=estimates['tabpfn_v3_base_check']
        notes.append('Single unique source response: analytical constant prediction; PHE fitting skipped')
    elif task=='classification':
        phe.fit(X,y)
        raw=phe.predict_proba(Q)
        values=np.zeros((len(Q),classes));values[:,np.asarray(phe._label_encoder.classes_,dtype=int)]=raw
        estimates['PHE_official_custom6_native_source_calibration']=normalize(values)
    else:
        phe.fit(X,y)
        estimates['PHE_official_custom6_native_source_calibration']=np.asarray(phe.predict(Q)).reshape(-1)
    torch.cuda.synchronize()
    times['PHE_official_custom6_native_source_calibration']=time.perf_counter()-start
    notes.append('Official tabpfn-extensions 0.0.4 PHE; custom six-configuration controlled portfolio, one holdout repeat, 25 GES iterations, 30-second validation limit')

    if task=='classification' and rf:
        forest=RandomForestTabPFNClassifier(tabpfn=model(task,device),n_jobs=1,
            adaptive_tree=False,n_estimators=6,random_state=0,max_predict_time=1e9)
        torch.cuda.synchronize();start=time.perf_counter()
        estimates['RF_PFN_official6_no_adaptive_tree']=fit_predict(forest,X,y,Q,task,classes)
        torch.cuda.synchronize();times['RF_PFN_official6_no_adaptive_tree']=time.perf_counter()-start
    result={}
    arrays={}
    evaluation_scale=(max(float(np.std(truth)),1e-12) if task=='regression' and str(meta.get('track','')).startswith('seven') else meta['scale'])
    for name,pred in estimates.items():
        if not np.all(np.isfinite(pred)):raise ValueError('nonfinite '+name)
        result[name]={**metrics(truth,pred,task,evaluation_scale),'seconds':times[name]}
        arrays['prediction__'+name]=pred
    arrays['y_target']=truth
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(out)
    atomic_json(out.with_suffix('.json'),{**meta,'family':family,'rho':rho,'seed':0,
        'status':'COMPLETE','methods':result,'notes':notes,
        'adaptation_training':False,'target_labels_used_for_fit':False,'bootstrap_revision':2,
        'micp_routing':routing if task=='classification' and len(pool_y)>256 else None,
        'PHE_status':'ANALYTICAL_CONSTANT' if constant else 'COMPLETE_SINGLETON_SAFE_HOLDOUT' if singleton else 'COMPLETE',
        'PHE_configurations_after_silo_pruning':0 if constant else len(phe._ens_model._estimators),
        'PHE_nonzero_members':0 if constant else int(np.count_nonzero(phe._ens_model.weights_)),
        'PHE_calibration':'native source holdout, not original cross-dataset protocol',
        'BETA_full_method_evaluated':False,'MixturePFN_CaPFN_evaluated':False})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--rf',action='store_true')
    parser.add_argument('--task',choices=['both','classification','regression'],default='both')
    args=parser.parse_args()
    torch.set_num_threads(1)
    logging.getLogger('AutoPostHocEnsemble').setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)
    logging.disable(logging.INFO)
    warnings.filterwarnings('ignore',category=FutureWarning)
    warnings.filterwarnings('ignore',message='The y_pred values do not sum to one')
    jobs=[]
    for task,suite in [('classification',99),('regression',353)]:
        if args.task not in ['both',task]:continue
        raw=json.loads((RUN/f'cache/suite_{suite}.json').read_text())['study']['tasks']['task_id']
        jobs.extend((task,int(t)) for t in raw)
    jobs=jobs[args.shard::args.shards]
    if args.limit:jobs=jobs[:args.limit]
    failures=0
    for task,task_id in jobs:
        if task=='regression':
            waited=0
            while not REG_CKPT.exists() and waited<1800:time.sleep(5);waited+=5
            if not REG_CKPT.exists():raise FileNotFoundError(REG_CKPT)
        p=RUN/'cache/five'/task/f'{task_id}.npz'
        meta=json.loads(p.with_suffix('.json').read_text())
        conditions=[('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        if args.smoke:conditions=conditions[:1]
        with np.load(p) as data:
            for family,rho in conditions:
                out=RUN/'results/tabpfn/five'/task/str(task_id)/f'{family}_{rho}.npz'
                if out.exists() and out.with_suffix('.json').exists():continue
                print('START',task,task_id,family,rho,flush=True)
                try:
                    cell(task,data,meta,family,rho,out,'cuda:0',args.rf)
                    print('DONE',task,task_id,family,rho,flush=True)
                except Exception as exc:
                    failures+=1
                    row={'task':task,'task_id':task_id,'family':family,'rho':rho,
                         'error':repr(exc),'traceback':traceback.format_exc()}
                    atomic_json(RUN/'results/inference_failures'/f'tabpfn_{task}_{task_id}_{family}_{rho}.json',row)
                    print('FAILED',json.dumps(row),flush=True)
                torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
