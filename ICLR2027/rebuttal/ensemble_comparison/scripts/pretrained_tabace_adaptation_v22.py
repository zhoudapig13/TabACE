"""Adapt pretrained TabACE contexts/fusion using the same FT TRAIN reference pool.

The Base, all final benchmark conditions, and previously measured FT rows are
frozen. Reference contexts use clean TRAIN labels; no final query labels fit or
select policies. The original observed-only reconstruction is kept in v21.
"""
import argparse
import fcntl
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
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
import pandas as pd
import torch
import finetune_tabace_main_v1 as main
import finetune_source_tabace_v5 as source
import finetune_tabace_reference_v3 as refs
import finetune_tabace_hierarchical_v4 as hierarchy
import ft_specific_refine_v19 as mixtures
import ft_specific_context_adaptation_v20 as expanded
import pretrained_tabace_rerun_v21 as raw
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,CHECKPOINTS,metrics

OUT=RUN/'results/pretrained_tabace_adaptation_v22'


class Cache:
    def __init__(self,job):
        self.model=Predictor(job['task'],'cuda:0');self.root=OUT/raw.relative(job)/'prediction_cache';self.memory={}
        kind='classifier' if job['task']=='classification' else 'regressor'
        self.sha=main.sha(CHECKPOINTS/f'tabicl-{kind}-v2-20260212.ckpt')

    def __call__(self,X,y,Q,members,nc):
        key=hashlib.sha256(('|'.join([protocol.array_sha256(a) for a in [X,y,Q]]+[str(members),self.sha])).encode()).hexdigest()
        if key in self.memory:
            p,meta=self.memory[key];return p,meta|dict(cache_hit=True,new_compute_seconds=0.)
        path=self.root/f'{key}.npz'
        if path.exists() and path.with_suffix('.json').exists():
            with np.load(path) as z:p=z['prediction']
            meta=json.loads(path.with_suffix('.json').read_text());self.memory[key]=(p,meta)
            return p,meta|dict(cache_hit=True,new_compute_seconds=0.)
        X,y=X[:2048],y[:2048];attempts=[];torch.cuda.synchronize();start=time.perf_counter()
        while True:
            try:p,_=self.model(X,y,Q,members=members,n_classes=nc,random_state=0);break
            except torch.cuda.OutOfMemoryError:
                attempts.append(len(y));torch.cuda.empty_cache()
                if len(y)<=256:raise
                X,y=X[:max(256,len(y)//2)],y[:max(256,len(y)//2)]
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        assert np.isfinite(p).all()
        meta=dict(context_rows=len(y),native_members=members,logical_TFM_passes=members,compute_seconds=elapsed,
            new_compute_seconds=elapsed,cache_hit=False,OOM_rows=attempts,pretrained_checkpoint_sha256=self.sha,FT_delta_loaded=False)
        main.scaling.npwrite(path,prediction=p);atomic_json(path.with_suffix('.json'),meta);self.memory[key]=(p,meta)
        return p,meta


def case(job,block,family,rho,role,env,predictor):
    rel=raw.relative(job);sub=Path('calibration') if role=='validation' else Path('query')/block
    original=raw.OUT/rel/sub/f'{family}_{rho}.npz';old=json.loads(original.with_suffix('.json').read_text())
    with np.load(original) as z:arrays={k:z[k] for k in z.files}
    observed=main.cell(arrays,old);path=OUT/rel/sub/f'{family}_{rho}.npz'
    if path.exists() and path.with_suffix('.json').exists():
        with np.load(path) as z:reference=z['reference'];extra=z['extra']
        meta=json.loads(path.with_suffix('.json').read_text())
    else:
        if role=='query':data,_=source.bank(job['task'],block,job['meta'],family,rho)
        else:data=source.materialize(job,env['pool'],np.r_[env['split']['train'],env['split']['val']],env['split']['cal'],family,rho)
        assert old['input_hashes']=={k:protocol.array_sha256(v) for k,v in data.items()}
        A,b,ids=env['reference'];Q=data['X_target']
        views=main.contexts.shared.contexts(A,b,Q,A[:256],b[:256],256,job['task'],base=arrays['base'])
        assert len(views)==5
        views[0]=('canonical_reference',A[:256],b[:256],Q)
        predictions=[];cost=[];names=[]
        for name,C,d,V in views:
            assert len(d)<=256
            p,m=predictor(C,d,V,1,env['nc']);predictions.append(p);cost.append(m);names.append(name)
        reference=np.stack(predictions)
        extra,extra_cost=expanded.candidates(env,predictor,X=data['X_source'],y=data['y_source'],Q=Q,family=family,track=job['track'])
        main.scaling.npwrite(path,reference=reference,extra=extra)
        meta=dict(reference_roles=names,extra_roles=expanded.NAMES,reference_cost=cost,extra_cost=extra_cost,
            source_input_hashes=old['input_hashes'],source_reference_ids_sha256=protocol.array_sha256(ids),
            Shared_context_cap=256,Specific_context_cap=2048,pretrained_weights_only=True,
            extra_clean_reference_labels='same official TRAIN rows accessible to frozen FT trainer',
            final_query_labels_used_by_constructors=False,reference_query_and_calibration_overlap=0)
        atomic_json(path.with_suffix('.json'),meta)
    reference_cell=observed|dict(bank=reference,mu=float(env['reference'][1].mean()),sigma=max(float(env['reference'][1].std()),1e-8))
    return dict(observed=observed,reference=reference_cell,extra=extra,cost=meta)


def subset(c,mask):return hierarchy.subset(c,mask)


def select_shared(cases,job,env,held):
    groups={};train={};valid={}
    for kind in ['observed','reference']:
        groups[kind]=[c[kind] for c in cases];train[kind]=[];valid[kind]=[]
        for c in groups[kind]:
            ids=hierarchy.row_map(job,env['split'],env['pool'],c['meta']['family'],c['meta']['rho'])
            mask=np.isin(ids,held);train[kind].append(subset(c,~mask));valid[kind].append(subset(c,mask))
    baseline=dict(kind='base');best=dict(bank='observed',gate='entropy',reg=None,policy=baseline,loss=hierarchy.score(valid['observed'],baseline));trials=[]
    gates=['entropy','entropy_or_disagreement'] if job['task']=='classification' else ['entropy']
    for kind in ['observed','reference']:
        for gate in gates:
            for reg in [.01,.1]:
                p=hierarchy.fit(train[kind],job['task'],5,gate,reg)
                loss=hierarchy.score(valid[kind],p);trials.append(dict(bank=kind,gate=gate,reg=reg,validation_loss=loss))
                if loss<best['loss']-1e-5:best=dict(bank=kind,gate=gate,reg=reg,policy=p,loss=loss)
    full=baseline if best['reg'] is None else hierarchy.fit(groups[best['bank']],job['task'],5,best['gate'],best['reg'])
    return full,best,trials


def special_cell(bundle,kind,shared):
    c=bundle['observed'];sp=main.fusion.predict(bundle[kind],shared)
    bank=np.concatenate([sp[None],c['base'][None],bundle['reference']['bank'],c['bank'],bundle['extra']],axis=0)
    names=['Shared','Base']+['reference:'+str(j) for j in range(5)]+['observed:'+str(j) for j in range(10)]+expanded.NAMES
    return c|dict(bank=bank,names=names,shared=sp)


def select_specific(job,env,cells,held):
    train=[];valid=[]
    for c in cells:
        ids=hierarchy.row_map(job,env['split'],env['pool'],c['meta']['family'],c['meta']['rho']);mask=np.isin(ids,held)
        train.append(mixtures.subset(c,~mask));valid.append(mixtures.subset(c,mask))
    _,labels,_=main.flat(env['pool']);prior=np.bincount(labels[:256].astype(int),minlength=env['nc'])+1. if job['task']=='classification' else np.ones(1)
    prior=prior/prior.sum();evaluation='source_prior' if job['task']=='classification' else 'natural';candidates=[]
    for j in range(len(cells[0]['bank'])):
        p=dict(kind='single',index=j);candidates.append((mixtures.utility(valid,p,evaluation,prior),p))
    for mode in (['balanced','source_prior'] if job['task']=='classification' else ['natural']):
        for reg in [.01,.1]:
            p=mixtures.fit(train,mode,prior,reg);candidates.append((mixtures.utility(valid,p,evaluation,prior),p))
    value,p=max(candidates,key=lambda a:a[0]);full=p if p['kind']=='single' else mixtures.fit(cells,p['mode'],prior,p['regularization'])
    # The fitted full-calibration policy is reconstructed after the final Shared
    # refit below; this function's training/selection never uses held-out labels.
    return full,dict(selected_validation_utility=value,trials=[dict(utility=v,policy=p) for v,p in candidates])


def run(job):
    dest=OUT/raw.relative(job)
    if (dest/'completion.json').exists():return
    env=expanded.prepare(job);env['reference']=refs.references(job,env['pool'],env['split']);predictor=Cache(job)
    bundles=[]
    for family,rho in [('original',0)]+[(f,r) for f in main.families(job['track']) for r in [50,100]]:
        block='clean' if rho==0 else job['track'];bundles.append(case(job,block,family,rho,'validation',env,predictor))
    held=hierarchy.calibration_split(job,env['pool'],env['split'])
    shared,best,shared_trials=select_shared(bundles,job,env,held)
    # Candidate Shared predictions for Specific selection use the 75%-fit policy,
    # never the Shared policy refit on the 25% validation labels.
    fit_cells=[special_cell(c,best['bank'],best['policy']) for c in bundles]
    full_cells=[special_cell(c,best['bank'],shared) for c in bundles]
    policies={};logs={}
    for family in ['original']+main.families(job['track']):
        fitted,log=select_specific(job,env,[c for c in fit_cells if c['meta']['family']==family],held)
        if fitted['kind']=='convex':
            group=[c for c in full_cells if c['meta']['family']==family]
            _,labels,_=main.flat(env['pool']);prior=np.bincount(labels[:256].astype(int),minlength=env['nc'])+1. if job['task']=='classification' else np.ones(1)
            policies[family]=mixtures.fit(group,fitted['mode'],prior/prior.sum(),fitted['regularization'])
        else:policies[family]=fitted
        logs[family]=log
    atomic_json(dest/'policies.json',dict(status='FROZEN_BEFORE_FINAL_QUERY_SCORING',Shared=shared,Specific=policies,
        selected_bank=best['bank'],Shared_trials=shared_trials,Specific_trials=logs,selection_original_positions=held.tolist(),
        additional_clean_TRAIN_reference_labels=True,held_validation_Shared_predictions_not_fit_on_held_labels=True,
        final_query_labels_for_policy_fit_or_selection=False,pretrained_TFM_weights_only=True,seed=0))
    rows=[]
    for block in (['clean','five'] if job['track']=='five' else ['seven']):
        for family,rho in main.contexts.case_grid(block):
            bundle=case(job,block,family,rho,'query',env,predictor);c=special_cell(bundle,best['bank'],shared)
            values=dict(Base=c['base'],Shared=c['shared'],Specific=mixtures.predict(c,policies[family]),Reference1=bundle['reference']['bank'][0])
            path=dest/'predictions'/block/f'{family}_{rho}.npz';main.scaling.npwrite(path,**values,y=c['y'])
            for name,p in values.items():
                rows.append(dict(task=job['task'],block=block,task_id=job['meta']['task_id'],data_id=job['meta']['data_id'],dataset=job['meta']['dataset'],
                    family=family,rho=rho,method=name,**metrics(c['y'],p,job['task'],c['meta']['scale'])))
    pd.DataFrame(rows).to_csv(dest/'query_detail.csv',index=False)
    atomic_json(dest/'completion.json',dict(status='COMPLETE',pretrained_weights_only=True,Shared_context_cap=256,Specific_context_cap=2048,
        reference_labels_same_pool_as_FT_training=True,final_benchmark_unchanged=True,seed=0))
    del env,predictor;gc.collect();torch.cuda.empty_cache()


def worker(shard):
    torch.set_num_threads(1);warnings.filterwarnings('ignore');main.fusion.training_tensors=hierarchy.tensors
    jobs=main.jobs();offset=shard*len(jobs)//8;jobs=jobs[offset:]+jobs[:offset]
    for job in jobs:
        dest=OUT/raw.relative(job);dest.mkdir(parents=True,exist_ok=True)
        with (dest/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (dest/'completion.json').exists():continue
            print('PRETRAINED_ADAPT_START',job['task'],job['track'],job['meta']['task_id'],flush=True)
            try:
                run(job);(dest/'error.json').unlink(missing_ok=True);print('PRETRAINED_ADAPT_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
            except Exception:
                error=traceback.format_exc();print(error,flush=True);atomic_json(dest/'error.json',dict(error=error));gc.collect();torch.cuda.empty_cache()


def report():
    paths=list(OUT.glob('jobs/*/*/*/query_detail.csv'))
    if not paths:return
    d=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);d.to_csv(OUT/'query_detail.csv',index=False)
    rows=[]
    for (task,block,method),g in d.groupby(['task','block','method']):
        fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        rows.append(dict(task=task,block=block,method=method,datasets=g.task_id.nunique(),complete=g.task_id.nunique()==len(main.frozen.cohort(task,block)),
            source='pretrained adapted rerun',**{f:float(g[f].mean()) for f in fields}))
    averages=pd.DataFrame(rows);averages.to_csv(OUT/'averages.csv',index=False)
    oldft=pd.read_csv(raw.FT_REPORT);oldft=oldft[oldft.method=='FT'].copy();oldft['source']='frozen FT'
    combined=pd.read_csv(raw.FT_SPECIFIC_REPORT);combined['method']=combined.method.map({'FT+Shared_frozen':'FT+Shared','FT+Specific_adapted':'FT+Specific'});combined['source']='frozen FT adapted'
    long=pd.concat([averages[averages.method!='Reference1'],oldft,combined],ignore_index=True);long.to_csv(OUT/'replacement_comparison_long.csv',index=False)
    table=[];gaps=[]
    for task in ['classification','regression']:
        fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        for method in ['Base','Shared','Specific','FT','FT+Shared','FT+Specific']:
            g=long[(long.task==task)&(long.method==method)];row=dict(task=task,method=method)
            for block in ['clean','five','seven']:
                h=g[g.block==block]
                if len(h):
                    h=h.iloc[0];row[block]=' / '.join(f'{h[f]:.4f}' for f in fields)
                    if method in raw.TARGETS[task][block]:
                        for j,f in enumerate(fields):gaps.append(dict(task=task,block=block,method=method,metric=f,value=h[f],
                            paper=raw.TARGETS[task][block][method][j],difference=h[f]-raw.TARGETS[task][block][method][j]))
            table.append(row)
    pd.DataFrame(table).to_csv(OUT/'replacement_comparison.csv',index=False);pd.DataFrame(gaps).to_csv(OUT/'paper_reference_gaps.csv',index=False)
    if len(paths)==175:
        latest=RUN/'results/base_tabace_ft_comparison_latest.csv';temp=latest.with_suffix('.tmp.csv')
        pd.DataFrame(table).to_csv(temp,index=False);temp.replace(latest)
        atomic_json(latest.with_suffix('.json'),dict(status='MEASURED_ADAPTED_PRETRAINED_AND_FROZEN_FT',source=str(OUT),seed=0,
            all_rows_measured=True,FT_rows_unchanged=True,Base_original_weights=True,Shared_Specific_context_and_policy_adaptation=True,
            clean_TRAIN_reference_supervision=True,Shared_context_cap=256,Specific_context_cap=2048,FT_Specific_context_cap=2048,Markdown_written=False))
    print(pd.DataFrame(table).to_string(index=False),flush=True);print('COMPLETED',len(paths),'/175',flush=True)


def launch():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_PRETRAINED_ADAPTATION',seed=0,benchmark_and_Base_frozen=True,
        user_authorized_context_and_fusion_adaptation_and_scores_above_paper=True,
        Shared='same five roles with TRAIN reference adaptation, original entropy/disagreement gated fusion, source held-out selection',
        Specific='inherits Shared, adds raw observed roles and native/local/transport/denoised TRAIN reference experts; source-selected expert/convex fusion',
        Shared_reference_pool=1024,Shared_context_cap=256,Specific_context_cap=2048,
        extra_reference_labels='same clean official TRAIN rows already available to FT trainer',
        no_backbone_weight_updates=True,policy_fit='75% source calibration',policy_selection='25% source calibration grouped original rows',
        final_query_labels_used_for_selection=False,constructor_receives_unlabeled_queries_only=True,
        original_raw_reconstruction_preserved=str(raw.OUT),frozen_FT_report_sha256=main.sha(raw.FT_REPORT),
        frozen_FT_Specific_report_sha256=main.sha(raw.FT_SPECIFIC_REPORT),
        development_adaptation_not_exact_original_paper_replication=True,Markdown_written=False))
    workers=[]
    for gpu in range(8):
        log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu)],env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();workers.append(dict(gpu=gpu,pid=p.pid))
    atomic_json(OUT/'workers.json',workers);print(workers,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker','report']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.phase=='launch':launch()
    elif a.phase=='report':report()
    else:worker(a.shard)
