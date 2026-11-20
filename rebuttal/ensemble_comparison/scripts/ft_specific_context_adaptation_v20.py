"""Specific-only context adaptation on frozen FT/Shared/development benchmark.

Contexts may grow to2048 already-used FT TRAIN rows; this is explicitly a
different inference budget from frozen Shared. Also retain <=256-row views.
All native ensemble sizes load the SAME task's fine-tuned tensor delta.
No query labels enter any context construction or model call.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif,f_regression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from tabicl import TabICLClassifier,TabICLRegressor
import ft_specific_refine_v19 as refine
import finetune_tabace_main_v1 as main
import finetune_source_tabace_v5 as source
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,CHECKPOINTS

OUT=RUN/'results/ft_specific_context_adaptation_v20'
NAMES=['native8_reference256','native4_reference1024','native1_reference2048','local_reference256',
       'transport_reference1024','ridge_denoised1024','trees_denoised1024','recent_observed_native4']


class FTNative(Predictor):
    def __init__(self,job):
        super().__init__(job['task'],'cuda:0');self.job=job
        path=refine.FT/refine.rel(job)/'weights.pt'
        self.state=torch.load(path,map_location='cpu',weights_only=False)['state_dict'];self.weight_sha256=main.sha(path)
        self.loaded=[]

    def __call__(self,X,y,Q,*,members=1,**kwargs):
        if members not in self.models:
            cls=TabICLClassifier if self.task=='classification' else TabICLRegressor
            kind='classifier' if self.task=='classification' else 'regressor'
            model=cls(n_estimators=members,device='cuda:0',model_path=str(CHECKPOINTS/f'tabicl-{kind}-v2-20260212.ckpt'),
                allow_auto_download=False,n_jobs=1,batch_size=1,random_state=0)
            model._resolve_device();model._load_model()
            missing,unexpected=model.model_.load_state_dict(self.state,strict=False)
            assert not unexpected and all(not k.startswith('icl_predictor.') or '.tf_icl.blocks.' in k for k in missing)
            model._load_model=lambda:None;model._weights_cached_for_rebuttal=True
            self.models[members]=model;self.loaded.append(members)
        return super().__call__(X,y,Q,members=members,**kwargs)


def balanced_rows(y,n,seed=0):
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(y==v) for v in np.unique(y)]
    counts=np.full(len(groups),n//len(groups));counts[:n%len(groups)]+=1
    rows=np.concatenate([rng.choice(g,int(c),replace=len(g)<c) for g,c in zip(groups,counts)])
    rng.shuffle(rows);return rows


def prepare(job):
    pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
    with np.load(refine.FT/refine.rel(job)/'split.npz') as z:split={k:z[k] for k in z.files}
    X,y,ids=main.flat(pool);pos=split['train'];A=X[pos].astype(np.float64);b=y[pos].copy()
    assert not np.intersect1d(ids[pos],pool['query_ids']).size
    assert not np.intersect1d(pos,np.r_[split['val'],split['cal']]).size
    score=(f_classif(A,b)[0] if job['task']=='classification' else f_regression(A,b)[0])
    score=np.nan_to_num(score,nan=0.,posinf=1e6,neginf=0.)
    columns=np.argsort(score,kind='stable')[-min(64,A.shape[1]):]
    center=np.median(A[:,columns],axis=0);scale=np.maximum(np.subtract(*np.percentile(A[:,columns],[75,25],axis=0)),1e-6)
    importance=np.sqrt(np.maximum(score[columns],0.)/max(float(np.max(score)),1e-8));importance=np.clip(importance,.1,1.)
    Z=np.clip((A[:,columns]-center)/scale,-10,10)*importance
    env=dict(pool=pool,split=split,A=A,b=b,ids=ids[pos],columns=columns,center=center,scale=scale,importance=importance,Z=Z,
        task=job['task'],nc=int(y.max())+1 if job['task']=='classification' else None)
    if job['task']=='regression':
        T=np.clip((A[:,columns]-center)/scale,-10,10)
        ridge=RidgeCV(alphas=[.1,1.,10.,100.],cv=3).fit(T,b)
        forest=ExtraTreesRegressor(n_estimators=64,min_samples_leaf=5,max_features=.8,n_jobs=1,random_state=0).fit(T,b)
        env['ridge_labels']=ridge.predict(T);env['trees_labels']=forest.predict(T)
    return env


class CachedPredictor:
    def __init__(self,job):
        self.predictor=FTNative(job);self.root=OUT/refine.rel(job)/'prediction_cache';self.memory={}

    def __call__(self,X,y,Q,members,nc):
        key=hashlib.sha256(('|'.join([protocol.array_sha256(v) for v in [X,y,Q]]+[str(members),self.predictor.weight_sha256])).encode()).hexdigest()
        if key in self.memory:
            p,meta=self.memory[key];return p,meta|dict(cache_hit=True,new_compute_seconds=0.)
        path=self.root/f'{key}.npz'
        if path.exists() and path.with_suffix('.json').exists():
            with np.load(path) as z:p=z['prediction']
            meta=json.loads(path.with_suffix('.json').read_text());self.memory[key]=(p,meta)
            return p,meta|dict(cache_hit=True,new_compute_seconds=0.)
        actual=min(2048,len(y));X=X[:actual];y=y[:actual]
        torch.cuda.synchronize();start=time.perf_counter();attempts=[]
        while True:
            try:p,_=self.predictor(X,y,Q,members=members,n_classes=nc,random_state=0);break
            except torch.cuda.OutOfMemoryError:
                attempts.append(dict(rows=len(y),error='CUDA OOM'));torch.cuda.empty_cache()
                if len(y)<=256:raise
                X=X[:max(256,len(y)//2)];y=y[:len(X)]
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        assert np.isfinite(p).all()
        meta=dict(context_rows=len(y),native_members=members,logical_TFM_passes=members,compute_seconds=elapsed,
            new_compute_seconds=elapsed,cache_hit=False,OOM_attempts=attempts,FT_checkpoint_sha256=self.predictor.weight_sha256,
            same_finetuned_weights_for_all_native_sizes=True)
        main.scaling.npwrite(path,prediction=p);atomic_json(path.with_suffix('.json'),meta)
        self.memory[key]=(p,meta);return p,meta


def local_prediction(env,predictor,Q):
    T=np.clip((Q[:,env['columns']]-env['center'])/env['scale'],-10,10)*env['importance']
    clusters=min(8,len(Q),max(1,len(np.unique(T,axis=0))))
    labels=KMeans(n_clusters=clusters,n_init=3,random_state=0).fit_predict(T)
    out=np.empty((len(Q),env['nc'])) if env['task']=='classification' else np.empty(len(Q));cost=[]
    for cluster in np.unique(labels):
        take=np.flatnonzero(labels==cluster);point=T[take].mean(0)
        distance=np.mean((env['Z']-point)**2,axis=1);rank=np.argsort(distance,kind='stable')
        chosen=list(rank[:min(256,len(rank))])
        if env['task']=='classification':
            for label in np.unique(env['b']):
                if np.any(env['b'][chosen]==label):continue
                candidate=next(int(i) for i in rank if env['b'][i]==label)
                for j in range(len(chosen)-1,-1,-1):
                    if np.sum(env['b'][chosen]==env['b'][chosen[j]])>1:chosen[j]=candidate;break
        chosen=np.asarray(chosen,dtype=int)
        p,meta=predictor(env['A'][chosen],env['b'][chosen],Q[take],1,env['nc']);out[take]=p;cost.append(meta)
    return out,dict(context_rows=max(m['context_rows'] for m in cost),native_members=1,logical_TFM_passes=len(cost),
        compute_seconds=sum(m['compute_seconds'] for m in cost),new_compute_seconds=sum(m['new_compute_seconds'] for m in cost),
        query_clusters=len(cost),supervised_metric_FT_TRAIN_only=True)


def candidates(env,predictor,*,X,y,Q,family,track):
    """Receives observed context and unlabeled queries, never query labels/rho."""
    A,b=env['A'],env['b'];n=len(b);task=env['task'];out=[];cost=[]
    small=(balanced_rows(b,min(256,n)) if task=='classification' else np.arange(min(256,n)))
    medium=(balanced_rows(b,min(1024,n)) if task=='classification' else np.arange(min(1024,n)))
    large=(balanced_rows(b,min(2048,n)) if task=='classification' else np.arange(min(2048,n)))
    def add(C,d,V,m):
        p,record=predictor(C,d,V,m,env['nc']);out.append(p);cost.append(record)
    add(A[small],b[small],Q,8)
    add(A[medium],b[medium],Q,4)
    add(A[large],b[large],Q,1)
    p,record=local_prediction(env,predictor,Q);out.append(p);cost.append(record)
    if task=='classification' and track=='seven' and family in ['domain','support','temporal']:
        center=np.median(A,axis=0);scale=np.maximum(np.subtract(*np.percentile(A,[75,25],axis=0)),1e-6)
        target_center=np.median(Q,axis=0);target_scale=np.maximum(np.subtract(*np.percentile(Q,[75,25],axis=0)),1e-6)
        aligned=(A[medium]-center)/scale*target_scale+target_center
        add(aligned,b[medium],Q,1)
    else:out.append(out[2]);cost.append(cost[2]|dict(alias='native1_reference2048',new_compute_seconds=0.))
    if task=='regression':
        add(A[medium],env['ridge_labels'][medium],Q,1)
        add(A[medium],env['trees_labels'][medium],Q,1)
    else:
        for _ in range(2):out.append(out[2]);cost.append(cost[2]|dict(alias='native1_reference2048',new_compute_seconds=0.))
    if task=='classification' and track=='seven' and family in ['concept','temporal']:
        recent_X,recent_y=X[-min(64,len(y)):],y[-min(64,len(y)):]
        rows=balanced_rows(recent_y,256);add(recent_X[rows],recent_y[rows],Q,4)
    else:out.append(out[1]);cost.append(cost[1]|dict(alias='native4_reference1024',new_compute_seconds=0.))
    assert len(out)==len(NAMES)
    return np.stack(out),cost


def get_case(job,block,family,rho,role,env,predictor):
    root=OUT/refine.rel(job)/('calibration' if role=='validation' else 'query')
    if role=='query':root=root/block
    path=root/f'{family}_{rho}.npz'
    base=refine.frozen_case(job,block,family,rho,role)
    if path.exists() and path.with_suffix('.json').exists():
        with np.load(path) as z:bank=z['bank'];truth=z['y']
        assert np.array_equal(truth,base['y'])
        record=json.loads(path.with_suffix('.json').read_text())
        return base|dict(bank=bank,names=record['names'])
    if role=='query':data,info=source.bank(job['task'],block,job['meta'],family,rho)
    else:
        data=source.materialize(job,env['pool'],np.r_[env['split']['train'],env['split']['val']],env['split']['cal'],family,rho)
        info=base['meta']
    assert {k:protocol.array_sha256(v) for k,v in data.items()}==base['meta']['input_hashes']
    new,cost=candidates(env,predictor,X=data['X_source'],y=data['y_source'],Q=data['X_target'],family=family,track=job['track'])
    bank=np.concatenate([base['bank'],new],axis=0);names=base['names']+NAMES
    main.scaling.npwrite(path,bank=bank,y=base['y'])
    atomic_json(path.with_suffix('.json'),dict(names=names,new_candidates=cost,Shared_prediction_sha256=protocol.array_sha256(base['shared']),
        original_input_hashes=base['meta']['input_hashes'],FT_checkpoint_sha256=predictor.predictor.weight_sha256,
        candidate_context_cap=2048,Shared_context_cap=256,rows_budget_changed=True,reference_labels_only_existing_FT_TRAIN=True,
        query_labels_used_for_context_construction=False,constructor_strength_blind=True,seed=0))
    return base|dict(bank=bank,names=names)


def select(job,env,cells):
    # Deduplicate actual prediction aliases before regularized simplex fitting.
    keys={};keep=[]
    for j in range(len(cells[0]['bank'])):
        signature=tuple(protocol.array_sha256(c['bank'][j]) for c in cells)
        if j<2 or signature not in keys:keys[signature]=j;keep.append(j)
    compact=[c|dict(bank=c['bank'][keep],names=[c['names'][j] for j in keep]) for c in cells]
    p,log=refine.select(job,env['pool'],env['split'],compact)
    if p['kind']=='single':p['index']=keep[p['index']]
    else:
        weights=np.zeros(len(cells[0]['bank']));weights[keep]=p['weights'];p['weights']=weights.tolist()
    log['deduplicated_indices']=keep;return p,log


def run(job):
    dest=OUT/refine.rel(job)
    if (dest/'completion.json').exists():return
    env=prepare(job);predictor=CachedPredictor(job);cells=[]
    for family in main.families(job['track']):
        for rho in [50,100]:cells.append(get_case(job,job['track'],family,rho,'validation',env,predictor))
    policies={};trials={}
    for family in main.families(job['track']):policies[family],trials[family]=select(job,env,[c for c in cells if c['meta']['family']==family])
    atomic_json(dest/'policies.json',dict(status='FROZEN_BEFORE_QUERY_SCORING',Specific=policies,trials=trials,
        branch_names=cells[0]['names'],Shared_unchanged=True,FT_unchanged=True,benchmark_unchanged=True,
        reference_original_ids_sha256=protocol.array_sha256(env['ids']),context_cap=2048,
        row_budget_differs_from_Shared=True,final_query_labels_used=False,calibration_grouped_by_original_row=True,seed=0))
    def loader(j,b,f,r,role):return get_case(j,b,f,r,role,env,predictor)
    refine.evaluate(job,policies,OUT,loader)
    atomic_json(dest/'completion.json',dict(status='COMPLETE',FT_checkpoint_sha256=predictor.predictor.weight_sha256,
        native_sizes_loaded_with_same_FT_weights=predictor.predictor.loaded,original_clean_Specific_frozen=True,
        expanded_context_cap=2048,Shared_frozen=True))
    del predictor,env;gc.collect();torch.cuda.empty_cache()


def worker(shard,shards,limit=0,task=None):
    torch.set_num_threads(1);warnings.filterwarnings('ignore');jobs=main.jobs()[shard::shards]
    if task:jobs=[j for j in jobs if j['task']==task]
    if limit:jobs=jobs[:limit]
    for job in jobs:
        try:
            print('SPECIFIC_CONTEXT_START',job['task'],job['track'],job['meta']['task_id'],flush=True)
            run(job);print('SPECIFIC_CONTEXT_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True);atomic_json(OUT/refine.rel(job)/'error.json',dict(error=error))
            gc.collect();torch.cuda.empty_cache()


def queue_worker(shard):
    import fcntl
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    jobs=main.jobs();offset=shard*len(jobs)//8;jobs=jobs[offset:]+jobs[:offset]
    for job in jobs:
        dest=OUT/refine.rel(job);dest.mkdir(parents=True,exist_ok=True)
        with (dest/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (dest/'completion.json').exists():continue
            try:
                print('QUEUE_START',job['task'],job['track'],job['meta']['task_id'],flush=True)
                run(job);print('QUEUE_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
                (dest/'error.json').unlink(missing_ok=True)
            except Exception:
                error=traceback.format_exc();print(error,flush=True);atomic_json(dest/'error.json',dict(error=error))
                gc.collect();torch.cuda.empty_cache()


def launch():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_NEW_SPECIFIC_CONTEXT_INFERENCE',seed=0,FT=str(refine.FT),Shared=str(refine.SHARED),
        paper_gap_targets=refine.TARGETS,Shared_and_benchmark_frozen=True,Clean='reuse prior Specific',
        candidate_roles=NAMES,context_cap=2048,Shared_context_cap=256,context_budget_differs=True,
        all_native_sizes_use_same_FT_delta=True,extra_label_access='none beyond prior FT TRAIN reference rows',
        supervised_context_adaptation='feature relevance for local retrieval; regression Ridge/ExtraTrees context-label denoising',
        auxiliary_models_do_not_predict_final_query_targets=True,construction_receives_unlabeled_queries_only=True,
        policy_selection='original-row grouped source calibration CV; source-prior-weighted Acc for classification, normalized MSE for regression',
        final_query_labels_used_for_selection=False,all_attempts_retained=True,Markdown_written=False))
    children=[]
    for gpu in range(8):
        log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
            env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(dict(gpu=gpu,pid=p.pid))
    atomic_json(OUT/'workers.json',children);print(json.dumps(children),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','queue','launch','report']);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0);p.add_argument('--task',default=None);a=p.parse_args()
    if a.phase=='launch':launch()
    elif a.phase=='report':refine.report(OUT)
    elif a.phase=='queue':queue_worker(a.shard)
    else:worker(a.shard,a.shards,a.limit,a.task)
