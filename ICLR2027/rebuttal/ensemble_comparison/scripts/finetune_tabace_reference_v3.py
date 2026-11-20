"""Supervision-matched DEVELOPMENT integration using clean FT TRAIN references.

Distinct from the original observed-context-only TabACE reconstruction. All
reference labels were already accessible to the actual backbone trainer. An
FT+Reference1 control separates reference access from multicontext fusion.
Original v1/v2 comparisons and all scores remain preserved.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import torch
import finetune_tabace_main_v1 as main
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import metrics

OUT=RUN/'results/finetune_tabace_reference_v3'
ROOTS=dict(classification=RUN/'results/finetune_tabace_main_v2',regression=RUN/'results/finetune_tabace_main_v1')
_original_tensors=main.fusion.training_tensors


def root(job):return OUT/'jobs'/job['task']/job['track']/str(job['meta']['task_id'])


def train_root(job):return ROOTS[job['task']]/'jobs'/job['task']/job['track']/str(job['meta']['task_id'])


def tensors(cells,m):
    data=_original_tensors(cells,m)
    if cells[0]['task']=='classification':
        factors=[]
        for c in cells:
            y=c['y'];labels,counts=np.unique(y,return_counts=True)
            lookup={int(k):len(y)/(len(labels)*count) for k,count in zip(labels,counts)}
            factors.extend(lookup[int(k)] for k in y)
        data['weights']*=torch.as_tensor(factors,dtype=torch.float64)
    return data


def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'plan.json'
    if p.exists():return
    atomic_json(p,dict(status='DECLARED_BEFORE_NEW_REFERENCE_CONTEXT_PREDICTIONS',seed=0,context_cap=256,
        benchmark_sha256=main.sha(main.frozen.CONFIG),benchmark_unchanged=True,
        goal='compare actual Base-to-FT and FT-to-combination increments; evaluate whether latter matches or exceeds former',
        target_is_development_objective_not_forced_test_ranking=True,
        methods=['FT','FT+Reference1','FT+Shared_ref','FT+Specific_ref'],
        roots={k:str(v) for k,v in ROOTS.items()},same_finetuned_checkpoint_all_methods=True,
        reference_access='only pre-corruption clean official TRAIN labels from the exact rows already accessible to backbone fine-tuning',
        reference_pool_cap=1024,reference_query_final_overlap=0,reference_calibration_overlap=0,
        added_final_query_label_access=False,clean_TRAIN_reference_access_is_explicit_adaptation=True,
        Shared='same five registered construction roles on the clean FT TRAIN reference pool, family blind',
        Specific='contains the five Shared reference roles and five original family-aware observed-context roles',
        specific_shared_inheritance=True,strength_blind_constructors=True,
        reference_control='one class-preserving/random clean TRAIN context with at most256 rows, same checkpoint',
        fusion='unchanged original v28 gate/weight form; reference-label scale for regression; source-only calibration',
        classification_calibration_loss='class-balanced NLL',fixed_gain=1.,fixed_regularization=.01,
        reference_and_original_branch_prediction_evidence='kept separately; no unfinetuned Shared/Specific rerun',
        original_observed_only_v1_v2_all_results_preserved=True,development_only=True,original_paper_replication=False,
        regression_Clean_five_semisynthetic=True,classification_five_query_feature_perturbation=True,
        time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),script_sha256=main.sha(__file__),Markdown_written=False))


def references(job,pool,split):
    X,y,ids=main.flat(pool);chosen=split['train']
    if job['task']=='classification':
        # A round-robin class preserving prefix, plus unique remaining TRAIN.
        groups=[list(chosen[y[chosen]==c]) for c in np.unique(y[chosen])]
        ordered=[]
        while len(ordered)<min(1024,len(chosen)):
            for g in groups:
                if g and len(ordered)<min(1024,len(chosen)):ordered.append(g.pop(0))
        chosen=np.asarray(ordered,dtype=int)
    else:chosen=chosen[:1024]
    assert not np.intersect1d(ids[chosen],pool['query_ids']).size
    assert not np.intersect1d(chosen,np.r_[split['val'],split['cal']]).size
    return X[chosen],y[chosen],ids[chosen]


def load_original(job,block,family,rho,role):
    base=train_root(job)
    p=base/'calibration'/f'{family}_{rho}.npz' if role=='validation' else base/'query'/block/f'{family}_{rho}.npz'
    with np.load(p) as z:a={k:z[k] for k in z.files}
    return a,json.loads(p.with_suffix('.json').read_text())


def make(job,predictor,data,block,family,rho,role,scale,ref):
    dest=root(job)/('calibration' if role=='validation' else 'query')
    if role=='query':dest=dest/block
    path=dest/f'{family}_{rho}.npz'
    if path.exists() and path.with_suffix('.json').exists():
        with np.load(path) as z:a={k:z[k] for k in z.files}
        return a,json.loads(path.with_suffix('.json').read_text())
    original,old=load_original(job,block,family,rho,role)
    assert old['input_hashes']=={k:protocol.array_sha256(v) for k,v in data.items()}
    X,y,Q=(data[k] for k in ['X_source','y_source','X_target'])
    A,b,ids=ref;task=job['task'];nc=int(b.max())+1 if task=='classification' else None
    cache={};seconds={};member_rows={}
    def predict(name,C,d,V):
        key=tuple(protocol.array_sha256(v) for v in (C,d,V));member_rows[name]=len(d)
        assert len(d)<=256
        if key in cache:seconds[name]=0.;return cache[key]
        torch.cuda.synchronize();start=time.perf_counter()
        p,_=predictor(C,d,V,n_classes=nc,members=1,random_state=0)
        torch.cuda.synchronize();seconds[name]=time.perf_counter()-start
        assert np.isfinite(p).all();cache[key]=p;return p
    # Exact same labeled source pool is available to both controls.
    reference=predict('Reference1',A[:256],b[:256],Q)
    start=time.perf_counter()
    views=main.contexts.shared.contexts(A,b,Q,A[:256],b[:256],256,task,base=original['base'])
    construction=time.perf_counter()-start
    new=[];names=[]
    for i,(name,C,d,V) in enumerate(views):
        key=f'{i}:{name}';names.append(key);new.append(predict(key,C,d,V))
    assert len(new)==5
    members=np.concatenate([np.stack(new),original['bank'][5:]],axis=0)
    assert len(members)==10
    arrays=dict(base=original['base'],reference=reference,bank=members,y=data['y_target'],
        mu=np.array(float(b.mean())),sigma=np.array(max(float(b.std()),1e-8)))
    main.scaling.npwrite(path,**arrays)
    old_extra=sum(old['inference_seconds'][n] for n in old['members'][5:])
    shared_seconds=old['FT_inference_seconds']+sum(seconds[n] for n in names)
    record=old|dict(reference_adaptation=True,reference_pool_rows=len(b),reference_original_ids_sha256=protocol.array_sha256(ids),
        reference_labels_already_accessible_to_weight_trainer=True,new_member_names=names,new_member_rows=member_rows,
        new_member_inference_seconds=seconds,new_reference_construction_seconds=construction,
        Reference1_inference_seconds=seconds['Reference1'],FT_Shared_inference_seconds=shared_seconds,
        FT_Specific_inference_seconds=shared_seconds+old_extra,original_observed_specific_extras_reused=True)
    atomic_json(path.with_suffix('.json'),record)
    return arrays,record


def job_run(job):
    dest=root(job);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completion.json').exists():return
    assert (train_root(job)/'completion.json').exists(),str(train_root(job))
    main.OUT=ROOTS[job['task']]
    pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
    with np.load(train_root(job)/'split.npz') as z:split={k:z[k] for k in z.files}
    ref=references(job,pool,split)
    predictor=main.UpdatedPredictor(job)
    calibration_cells=[]
    for family,rho in [('original',0)]+[(f,r) for f in main.families(job['track']) for r in (50,100)]:
        block='clean' if rho==0 else job['track']
        data=main.materialize(job,pool,np.r_[split['train'],split['val']],split['cal'],family,rho)
        scale=1. if job['task']=='classification' else max(float(data['y_target'].std()),1e-8)
        arrays,record=make(job,predictor,data,block,family,rho,'validation',scale,ref)
        calibration_cells.append(main.cell(arrays,record))
    start=time.perf_counter()
    shared=main.fusion.fit(calibration_cells,job['task'],5,1.,.01)
    specific={}
    for family in ['original']+main.families(job['track']):
        specific[family]=main.fusion.fit([c for c in calibration_cells if c['meta']['family']==family],job['task'],10,1.,.01)
    atomic_json(dest/'policies.json',dict(status='FROZEN_BEFORE_NEW_QUERY_SCORING',Shared=shared,Specific=specific,
        source_calibration_only=True,reference_adaptation=True,fitting_seconds=time.perf_counter()-start,final_query_labels_used=False))
    rows=[]
    for block in (['clean','five'] if job['track']=='five' else ['seven']):
        for family,rho in main.contexts.case_grid(block):
            data,info=main.scaling.bank(job['task'],block,job['meta'],family,rho)
            arrays,record=make(job,predictor,data,block,family,rho,'query',info['scale'],ref)
            c=main.cell(arrays,record)
            predictions={'FT':c['base'],'FT+Reference1':arrays['reference'],
                'FT+Shared_ref':main.fusion.predict(c,shared),'FT+Specific_ref':main.fusion.predict(c,specific[family])}
            for name,p in predictions.items():
                rows.append(dict(task=job['task'],block=block,task_id=job['meta']['task_id'],data_id=job['meta']['data_id'],
                    dataset=job['meta']['dataset'],family=family,rho=rho,method=name,
                    **metrics(data['y_target'],p,job['task'],info['scale']),reference_adaptation=True))
    pd.DataFrame(rows).to_csv(dest/'query_detail.csv',index=False)
    atomic_json(dest/'completion.json',dict(status='COMPLETE',conditions=len(rows)//4,methods=list(predictions),
        reference_original_ids_sha256=protocol.array_sha256(ref[2]),same_updated_checkpoint_as_parent=True,
        additional_query_labels_used=False,reference_pool_already_FT_TRAIN=True))
    del predictor;gc.collect();torch.cuda.empty_cache()


def worker(shard,shards,limit=0):
    torch.set_num_threads(1);warnings.filterwarnings('ignore');main.fusion.training_tensors=tensors
    selected=main.jobs()[shard::shards]
    if limit:selected=selected[:limit]
    for job in selected:
        print('REFERENCE_START',job['task'],job['track'],job['meta']['task_id'],flush=True)
        try:
            job_run(job);print('REFERENCE_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(root(job)/'error.json',dict(job=job,error=error))
            gc.collect();torch.cuda.empty_cache()


def report():
    files=sorted((OUT/'jobs').glob('*/*/*/query_detail.csv'))
    if not files:return
    detail=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    base=[]
    for _,r in detail[detail['method']=='FT'].iterrows():
        p=main.frozen.OUT/'input_banks'/r['task']/r['block']/str(r['task_id'])/f"{r['family']}_{r['rho']}.json"
        d=json.loads(p.read_text());base.append({k:r[k] for k in ['task','block','task_id','data_id','dataset','family','rho']}
            |dict(method='Base (cached, paired)')|d['metrics'])
    detail=pd.concat([pd.DataFrame(base),detail],ignore_index=True);detail.to_csv(OUT/'query_detail.csv',index=False)
    rows=[];deltas=[]
    for (task,block),g in detail.groupby(['task','block']):
        fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        expected=len(main.frozen.cohort(task,block))
        for method,h in g.groupby('method'):
            rows.append(dict(task=task,block=block,method=method,datasets=h['task_id'].nunique(),expected_datasets=expected,
                complete=h['task_id'].nunique()==expected,**{f:float(h[f].mean()) for f in fields}))
        for metric in fields:
            table=g.groupby(['task_id','method'])[metric].mean().unstack('method')
            before=table['FT']-table['Base (cached, paired)']
            for method in ['FT+Reference1','FT+Shared_ref','FT+Specific_ref']:
                after=table[method]-table['FT']
                direction=-1 if metric=='nrmse' else 1
                deltas.append(dict(task=task,block=block,metric=metric,method=method,datasets=len(table),
                    Base_to_FT_oriented_gain=float(direction*before.mean()),FT_to_combination_oriented_gain=float(direction*after.mean()),
                    combination_increment_at_least_FT_increment=bool(direction*after.mean()>=direction*before.mean())))
    pd.DataFrame(rows).to_csv(OUT/'averages.csv',index=False)
    pd.DataFrame(deltas).to_csv(OUT/'increment_comparison.csv',index=False)
    complete=len(list((OUT/'jobs').glob('*/*/*/completion.json')))
    errors=[p for p in (OUT/'jobs').glob('*/*/*/error.json') if not (p.parent/'completion.json').exists()]
    atomic_json(OUT/'progress.json',dict(completed_jobs=complete,expected_jobs=len(main.jobs()),unresolved_errors=len(errors),full_complete=complete==len(main.jobs())))
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print('REFERENCE_PROGRESS',complete,'/',len(main.jobs()),flush=True)


def launch_after_v2():
    plan();pending={int(w['gpu']):int(w['pid']) for w in json.loads((ROOTS['classification']/'workers.json').read_text())}
    children=[];(OUT/'logs').mkdir(exist_ok=True)
    while pending:
        for gpu,oldpid in list(pending.items()):
            try:os.kill(oldpid,0);continue
            except ProcessLookupError:pass
            log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
            p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
                env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();children.append(dict(gpu=gpu,pid=p.pid));atomic_json(OUT/'workers.json',children)
            print('REFERENCE_WORKER_STARTED',gpu,p.pid,flush=True);del pending[gpu]
        if pending:time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report','launch_after_v2'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='worker':worker(a.shard,a.shards,a.limit)
    elif a.phase=='report':report()
    else:launch_after_v2()
