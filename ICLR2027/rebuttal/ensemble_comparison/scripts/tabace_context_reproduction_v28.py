"""Paired Base/Shared/Specific reproduction at four context row budgets.

Raw constructors use exactly Base's observed source pool. Specific contains all
Shared candidates. Query labels never enter constructors. Source-validation
fusion and explicitly tagged supervised adaptations are separate stages.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import benchmark256_v22 as frozen
import tabicl_context_scaling_v23 as scaling
import validation_recovery_v26 as repaired
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,normalize,metrics
import shared_reference as shared
import specific_reference as specific

OUT=RUN/'results/tabace_context_reproduction_v28'
BUDGETS=[256,1024,4096,8192]
_GEOMETRY_CACHE={}
_REPAIR_CACHE={}


def plan():
    path=OUT/'plan.json'
    if path.exists():return
    targets={
        'classification':{
            'clean':dict(Base=[.804,.901],Shared=[.804,.901],Specific=[.804,.900]),
            'five':dict(Base=[.494,.676],Shared=[.575,.720],Specific=[.610,.752]),
            'seven':dict(Base=[.763,.826],Shared=[.772,.831],Specific=[.805,.854])},
        'regression':{
            'clean':dict(Base=[.739,.476],Shared=[.738,.478],Specific=[.738,.476]),
            'five':dict(Base=[-1.077,1.270],Shared=[.363,.783],Specific=[.485,.708]),
            'seven':dict(Base=[.273,.730],Shared=[.467,.659],Specific=[.655,.535])}}
    atomic_json(OUT/'paper_targets.json',targets)
    atomic_json(path,dict(status='DECLARED_BEFORE_NEW_MEMBER_INFERENCE',seed=0,n_estimators_per_member=1,
        budgets=BUDGETS,classification_datasets=dict(clean=72,five=72,seven=35),regression_datasets=dict(clean=35,five=35,seven=33),
        targets='paper_targets.json; check all oriented Base->Shared, Base->Specific, Shared->Specific differences',
        strict_ranking='Specific>Shared>Base for accuracy,AUC,R2; reverse for nRMSE',
        task='complete all methods, not Base-only numerical calibration',
        previous_goal_turn='PROGRESS: audited Base inputs and observed source-validation transfer limitations; Shared/Specific not yet complete',
        query_input_providers=dict(classification='frozen v22 at256; audited v23 above256',regression_clean='v22/v23',
            regression_five='immutable frozen_five_query_v26 repair probability0.00064',
            regression_seven='v22/v23 frozen fixed-strength protocol; small-growth Base calibration remains unresolved'),
        regression_seven_Base_calibration_still_incomplete=True,
        raw_shared_branches=5,initial_specific_extra_branches=5,Specific_contains_all_Shared=True,
        raw_all_methods_observed_source_information_identical=True,
        per_branch_rows='at most budget; actual row counts and aliases recorded',
        Shared_family_blind=True,Specific_family_identifier_visible=True,constructors_strength_blind=True,
        fusion_selection='held-out TRAIN labels only; no new query label fitting',
        extra_clean_label_adaptations='authorized but separate tagged stage; never silently mixed into raw reconstruction',
        reconstruction_not_bit_exact_archived_TabACE=True,development_benchmark=True,original_paper_replication=False,
        semisynthetic_regression_blocks=['clean','five'],Markdown_written=False))


def projection(Z,V):
    if Z.shape[1]<=64:return Z,V
    rng=np.random.default_rng(0)
    matrix=rng.normal(size=(Z.shape[1],32)).astype(np.float32)/np.sqrt(32.)
    return np.asarray(Z@matrix,dtype=np.float32),np.asarray(V@matrix,dtype=np.float32)


def safe_repair_labels(Z,y,threshold):
    key=(protocol.array_sha256(Z),protocol.array_sha256(y))
    if key not in _REPAIR_CACHE:
        A,_=projection(np.asarray(Z,dtype=np.float32),np.asarray(Z[:1],dtype=np.float32))
        count=min(11,len(y));nn=NearestNeighbors(n_neighbors=count,n_jobs=1).fit(A)
        ids=nn.kneighbors(A,return_distance=False)
        neighbors=np.stack([row[row!=i][:min(10,len(y)-1)] for i,row in enumerate(ids)])
        classes=np.unique(y);votes=np.stack([(y[neighbors]==c).mean(1) for c in classes],axis=1)
        if len(_REPAIR_CACHE)>=16:_REPAIR_CACHE.pop(next(iter(_REPAIR_CACHE)))
        _REPAIR_CACHE[key]=(classes,votes)
    classes,votes=_REPAIR_CACHE[key]
    result=y.copy();mask=votes.max(1)>=threshold;result[mask]=classes[votes[mask].argmax(1)]
    for label in classes:
        if not np.any(result==label):result[np.flatnonzero(y==label)[0]]=label
    return result


specific.repair_labels=safe_repair_labels


def shared_contexts(X,y,Q,budget,task,base):
    if task=='regression':return shared.contexts(X,y,Q,X,y,budget,task,base=base)
    key=(protocol.array_sha256(X),protocol.array_sha256(y))
    if key not in _GEOMETRY_CACHE:
        Z,V,center,scale=shared.geometry(X,Q)
        matrix=None
        if Z.shape[1]>64:matrix=np.random.default_rng(0).normal(size=(Z.shape[1],32)).astype(np.float32)/np.sqrt(32.)
        A=Z.astype(np.float32) if matrix is None else np.asarray(Z.astype(np.float32)@matrix,dtype=np.float32)
        nn=NearestNeighbors(n_neighbors=min(11,len(y)),n_jobs=1).fit(A)
        ids=nn.kneighbors(A,return_distance=False)
        neighbors=np.stack([row[row!=i][:min(10,len(y)-1)] for i,row in enumerate(ids)])
        consensus=np.max(np.stack([(y[neighbors]==label).mean(1) for label in np.unique(y)],axis=1),axis=1)
        order=specific.importance(X,y.astype(float))
        if len(_GEOMETRY_CACHE)>=4:_GEOMETRY_CACHE.pop(next(iter(_GEOMETRY_CACHE)))
        _GEOMETRY_CACHE[key]=(A,center,scale,matrix,consensus,order)
    A,center,scale,matrix,consensus,order=_GEOMETRY_CACHE[key]
    V=((Q-center)/scale).astype(np.float32);T=V if matrix is None else np.asarray(V@matrix,dtype=np.float32)
    anchors=np.quantile(T,[.1,.3,.5,.7,.9],axis=0)
    anchor_distance=pairwise_distances(A,anchors,metric='sqeuclidean').min(1)
    d=pairwise_distances(A,T,metric='sqeuclidean');k=min(10,len(T))
    target_distance=np.partition(d,k-1,axis=1)[:,:k].mean(1)
    outlying=(np.minimum(np.abs(A),10.)**2).mean(1)
    disagreement=1-consensus
    views=[]
    for name,score,fraction in [('target_neighbor',anchor_distance,.8),('representative',target_distance+.35*outlying,.7),
                                 ('locally_stable',disagreement+.10*target_distance,.8),('low_conflict',disagreement+.20*outlying,.75)]:
        rows=shared.class_subset(y,score,fraction,budget);views.append((name,X[rows],y[rows],Q))
    columns=np.asarray([j for j in range(X.shape[1]) if j!=int(order[0])])
    if not len(columns):columns=np.asarray([0])
    views.insert(3,('feature_subspace',X[:,columns],y,Q[:,columns]));return views


def specific_contexts(X,y,Q,budget,task,block,family,base):
    h=max(1,3*len(y)//4);h=min(h,len(y)-1)
    return specific.specific_contexts(X,y,Q,X,y,budget,task,family=family,
        track='seven_reimplemented' if block=='seven' else 'five',
        history=(X[:h],y[:h]),reference=(X[h:],y[h:]),base=base)


def query_input(task,block,meta,family,rho,budget,pool):
    core,info=scaling.bank(task,block,meta,family,rho)
    if budget==256:
        path=frozen.OUT/'input_banks'/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
        with np.load(path) as z:base=z['Base_prediction']
        return core,info['scale'],base,str(path),False
    if task=='regression' and block=='five':
        rate=json.loads((repaired.OUT/'frozen_five_query/parameters.json').read_text())['repair_probability']
        env=repaired.environment(pool,meta,block,family,rho,np.arange(len(pool['y'])))
        X,y,_=repaired.context(env,rate,budget);data=core|dict(X_source=X,y_source=y)
        path=repaired.OUT/'frozen_five_query/predictions'/str(budget)/str(meta['task_id'])/f'{family}_{rho}.npz'
    else:
        data,_=scaling.materialize(task,block,meta,family,rho,budget,pool)
        path=scaling.OUT/'predictions'/str(budget)/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
    record=json.loads(path.with_suffix('.json').read_text())
    hashes={k:protocol.array_sha256(v) for k,v in data.items()};assert hashes==record['input_hashes']
    with np.load(path) as z:base=z['prediction']
    return data,info['scale'],base,str(path),task=='regression' and block=='seven'


def case_grid(block):
    from seven_shifts import FAMILIES
    return [('original',0)] if block=='clean' else [(f,r) for f in (protocol.FAMILIES if block=='five' else FAMILIES) for r in [20,50,80,100]]


def jobs():
    rows=[]
    for task in ['classification','regression']:
        for track in ['five','seven']:
            for meta in frozen.cohort(task,track):rows.append((task,track,meta))
    return rows


def run_case(predictor,data,base,task,block,meta,family,rho,budget,scale,out,origin,provisional=False):
    X,y,Q=(data[k] for k in ['X_source','y_source','X_target']);nc=base.shape[1] if task=='classification' else None
    original_hashes={k:protocol.array_sha256(v) for k,v in data.items()}
    start=time.monotonic();common=shared_contexts(X,y,Q,len(y),task,base)
    extra=specific_contexts(X,y,Q,len(y),task,block,family,base)
    all_views=common+extra;assert len(common)==5 and len(extra)==5
    construction=time.monotonic()-start
    sourcehash=(protocol.array_sha256(X),protocol.array_sha256(y),protocol.array_sha256(Q));cached={sourcehash:base}
    predictions=[];roles=[];rows=[];times=[];aliases=[];hashes=[]
    for name,A,b,T in all_views:
        assert len(b)<=len(y),(name,len(b),len(y))
        key=tuple(protocol.array_sha256(np.asarray(v)) for v in [A,b,T])
        alias=key in cached;started=time.monotonic()
        if alias:p=cached[key]
        else:p,_=predictor(A,b,T,members=1,random_state=0,n_classes=nc);cached[key]=p
        assert np.isfinite(p).all();predictions.append(p);roles.append(name);rows.append(len(b));times.append(time.monotonic()-started);aliases.append(alias);hashes.append(key)
    bank=np.stack(predictions);assert np.array_equal(bank[:5],np.asarray(predictions[:5]))
    assert original_hashes=={k:protocol.array_sha256(v) for k,v in data.items()},'Constructor mutated paired benchmark inputs'
    diagnostic_shared=bank[:5].mean(0);diagnostic_specific=bank.mean(0)
    scaling.npwrite(out,base=base,members=bank,y_target=data['y_target'],source_mean=np.asarray(float(y.mean())),
                    source_scale=np.asarray(max(float(y.std()),1e-8)))
    atomic_json(out.with_suffix('.json'),dict(status='MEMBER_BANK_COMPLETE_FUSION_PENDING',task=task,block=block,task_id=meta['task_id'],
        dataset=meta['dataset'],family=family,rho=rho,requested_context=budget,actual_context=len(y),seed=0,
        shared_roles=roles[:5],specific_additional_roles=roles[5:],Specific_contains_all_Shared=True,
        context_rows=rows,aliases=aliases,member_input_hashes=hashes,member_seconds=times,construction_seconds=construction,
        source_input_hashes={k:protocol.array_sha256(v) for k,v in data.items()},Base_origin=origin,
        Base_metrics=metrics(data['y_target'],base,task,scale),diagnostic_mean_metrics=dict(
            Shared=metrics(data['y_target'],diagnostic_shared,task,scale),Specific=metrics(data['y_target'],diagnostic_specific,task,scale)),
        scale=scale,observed_source_information_matched=True,extra_clean_labels_used=False,
        Shared_family_blind=True,Specific_family_visible=True,constructors_receive_strength=False,
        query_labels_used_by_constructors=False,geometry_projection='fixedseed32D only for context ranking when p>64',
        semisynthetic=task=='regression' and block in ['clean','five'],
        provisional_Base_calibration=provisional,development_only=True,original_paper_replication=False))


def worker(shard,shards,budgets):
    import torch,warnings
    warnings.filterwarnings('ignore');torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    predictors={}
    for task,track,meta in jobs()[shard::shards]:
        try:
            pool=scaling.prepare(task,track,meta)
            if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
            blocks=['clean','five'] if track=='five' else ['seven']
            for budget in budgets:
                for block in blocks:
                    for family,rho in case_grid(block):
                        out=OUT/'banks/query'/str(budget)/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                        if out.exists() and out.with_suffix('.json').exists():continue
                        data,scale,base,origin,provisional=query_input(task,block,meta,family,rho,budget,pool)
                        run_case(predictors[task],data,base,task,block,meta,family,rho,budget,scale,out,origin,provisional)
                    print('QUERY_BANK_BLOCK_COMPLETE',budget,task,block,meta['dataset'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'query_{task}_{track}_{meta["task_id"]}.json',dict(error=error))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--budgets',default='256');a=p.parse_args()
    plan() if a.phase=='plan' else worker(a.shard,a.shards,[int(n) for n in a.budgets.split(',')])
