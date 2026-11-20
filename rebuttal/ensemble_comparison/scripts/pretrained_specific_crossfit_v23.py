"""Stabilize adapted Specific selection with grouped three-fold calibration.

Shared remains frozen at v22. Within each Specific selection fold its Shared
anchor is fitted using only that fold's training rows. TFM predictions are
reused. Classification selection considers Acc and AUC; regression uses MSE.
"""
import argparse
import fcntl
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
import torch
import pretrained_tabace_adaptation_v22 as parent
from prepare_data import RUN,atomic_json

main=parent.main;mix=parent.mixtures;hierarchy=parent.hierarchy;raw=parent.raw
OUT=RUN/'results/pretrained_specific_crossfit_v23'


def prepare(job):
    pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
    with np.load(raw.FT/raw.relative(job)/'split.npz') as z:split={k:z[k] for k in z.files}
    _,y,_=main.flat(pool)
    env=dict(pool=pool,split=split,nc=int(y.max())+1 if job['task']=='classification' else None)
    env['reference']=parent.refs.references(job,pool,split)
    return env


def fold_masks(job,env,bundles):
    _,y,_=main.flat(env['pool']);pos=env['split']['cal'];rng=np.random.default_rng(1901);mapping={}
    groups=[pos[y[pos]==v] for v in np.unique(y[pos])] if job['task']=='classification' else [pos]
    for group in groups:
        for j,i in enumerate(rng.permutation(group)):mapping[int(i)]=j%3
    result=[]
    for fold in range(3):
        result.append([np.asarray([mapping[int(i)]==fold for i in hierarchy.row_map(job,env['split'],env['pool'],
            c['observed']['meta']['family'],c['observed']['meta']['rho'])]) for c in bundles])
    return result


def score(cells,p,mode,prior):
    if cells[0]['task']=='regression':return -mix.loss(cells,p,mode,prior)
    values=[]
    for c in cells:
        q=mix.predict(c,p);acc=float(np.dot(mix.row_weights(c,'source_prior',prior),q.argmax(1)==c['y']))
        auc=parent.metrics(c['y'],q,'classification',1.)['auc']
        values.append(acc if auc is None else .5*(acc+auc))
    return float(np.mean(values))-.001*mix.loss(cells,p,'source_prior',prior)


def select(job,env,full,groups,family):
    original_width=len(full[0]['bank'])
    keep=list(range(len(full[0]['bank'])))
    if job['task']=='regression' and family=='original':
        keep=[j for j in keep if full[0]['names'][j] not in ['ridge_denoised1024','trees_denoised1024']]
    # Deduplicate fixed experts without using any calibration labels.
    seen=set();indices=[]
    for j in keep:
        signature=tuple(parent.protocol.array_sha256(c['bank'][j]) for c in full)
        if j==0 or signature not in seen:indices.append(j);seen.add(signature)
    def compact(c):return c|dict(bank=c['bank'][indices],names=[c['names'][j] for j in indices])
    full=[compact(c) for c in full];groups=[([compact(c) for c in t],[compact(c) for c in v]) for t,v in groups]
    _,labels,_=main.flat(env['pool']);prior=np.bincount(labels[:256].astype(int),minlength=env['nc'])+1. if job['task']=='classification' else np.ones(1)
    prior=prior/prior.sum();trials=[]
    for j in range(len(indices)):
        p=dict(kind='single',index=j);s=[score(v,p,'natural',prior) for _,v in groups]
        trials.append(dict(kind='single',index=j,score=float(np.mean(s)),fold_scores=s))
    for mode in (['balanced','source_prior'] if job['task']=='classification' else ['natural']):
        for reg in [.01,.1,1.]:
            s=[]
            for t,v in groups:
                p=mix.fit(t,mode,prior,reg);s.append(score(v,p,mode,prior))
            trials.append(dict(kind='convex',mode=mode,regularization=reg,score=float(np.mean(s)),fold_scores=s))
    best=max(trials,key=lambda r:r['score'])
    p=dict(kind='single',index=best['index']) if best['kind']=='single' else mix.fit(full,best['mode'],prior,best['regularization'])
    if p['kind']=='single':p['index']=indices[p['index']]
    else:
        w=np.zeros(original_width)
        w[indices]=p['weights'];p['weights']=w.tolist()
    return p,dict(selected=best,trials=trials,kept_original_indices=indices,Shared_anchor_out_of_fold=True)


def run(job):
    dest=OUT/raw.relative(job)
    if (dest/'completion.json').exists():return
    env=prepare(job);source=parent.OUT/raw.relative(job);record=json.loads((source/'policies.json').read_text())
    shared=record['Shared'];kind=record['selected_bank'];bundles=[]
    for family,rho in [('original',0)]+[(f,r) for f in main.families(job['track']) for r in [50,100]]:
        bundles.append(parent.case(job,'clean' if rho==0 else job['track'],family,rho,'validation',env,None))
    full=[parent.special_cell(c,kind,shared) for c in bundles];masks=fold_masks(job,env,bundles);fold_groups=[]
    for masks_for_fold in masks:
        if shared['kind']=='base':fold_policy=dict(kind='base')
        else:
            train=[hierarchy.subset(c[kind],~mask) for c,mask in zip(bundles,masks_for_fold)]
            fold_policy=hierarchy.fit(train,job['task'],5,shared.get('gate_variant','entropy'),shared['regularization'],shared['gain'])
        train=[];valid=[]
        for c,mask in zip(bundles,masks_for_fold):
            cell=parent.special_cell(c,kind,fold_policy)
            if mask.any() and (~mask).any():train.append(mix.subset(cell,~mask));valid.append(mix.subset(cell,mask))
        fold_groups.append((train,valid))
    policies={};logs={};width=len(full[0]['bank'])
    for family in ['original']+main.families(job['track']):
        group=[c for c in full if c['meta']['family']==family]
        folds=[([c for c in t if c['meta']['family']==family],[c for c in v if c['meta']['family']==family]) for t,v in fold_groups]
        p,log=select(job,env,group,folds,family)
        if p['kind']=='convex':p['weights']+=([0.]*(width-len(p['weights'])))
        policies[family]=p;logs[family]=log
    atomic_json(dest/'policies.json',dict(status='FROZEN_BEFORE_QUERY_SCORING',Specific=policies,trials=logs,
        Shared_frozen_from=str(source/'policies.json'),Shared_hyperparameters_preselected_on_source_calibration=True,
        Shared_parameters_refit_without_held_fold_labels_for_Specific_selection=True,
        original_row_grouped_folds=3,final_query_labels_for_selection=False,
        classification_selection='mean of source-prior Acc and macro AUC, minus .001 NLL',
        regression_selection='normalized MSE',Clean_regression='true-label contexts, no label-denoising experts',seed=0))
    rows=[]
    for block in (['clean','five'] if job['track']=='five' else ['seven']):
        for family,rho in main.contexts.case_grid(block):
            bundle=parent.case(job,block,family,rho,'query',env,None);c=parent.special_cell(bundle,kind,shared)
            values=dict(Base=c['base'],Shared=c['shared'],Specific=mix.predict(c,policies[family]),Reference1=bundle['reference']['bank'][0])
            path=dest/'predictions'/block/f'{family}_{rho}.npz';main.scaling.npwrite(path,**values,y=c['y'])
            for method,p in values.items():rows.append(dict(task=job['task'],block=block,task_id=job['meta']['task_id'],data_id=job['meta']['data_id'],
                dataset=job['meta']['dataset'],family=family,rho=rho,method=method,**parent.metrics(c['y'],p,job['task'],c['meta']['scale'])))
    pd.DataFrame(rows).to_csv(dest/'query_detail.csv',index=False)
    atomic_json(dest/'completion.json',dict(status='COMPLETE',new_TFM_calls=0,Shared_frozen=True,seed=0))


def worker(shard):
    torch.set_num_threads(1);main.fusion.training_tensors=hierarchy.tensors;jobs=main.jobs();offset=shard*len(jobs)//8;jobs=jobs[offset:]+jobs[:offset]
    while True:
        remaining=0;started=0
        for job in jobs:
            dest=OUT/raw.relative(job);dest.mkdir(parents=True,exist_ok=True)
            if (dest/'completion.json').exists():continue
            remaining+=1
            if not (parent.OUT/raw.relative(job)/'completion.json').exists():continue
            with (dest/'worker.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (dest/'completion.json').exists():continue
                started+=1;print('CROSSFIT_START',job['task'],job['track'],job['meta']['task_id'],flush=True)
                try:
                    run(job);(dest/'error.json').unlink(missing_ok=True);print('CROSSFIT_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
                except Exception:
                    error=traceback.format_exc();print(error,flush=True);atomic_json(dest/'error.json',dict(error=error))
                    raise
                gc.collect()
        if not remaining:return
        if not started:time.sleep(3)


def report():
    old=parent.OUT
    try:parent.OUT=OUT;parent.report()
    finally:parent.OUT=old


def launch():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_CROSSFIT_SELECTION',seed=0,parent=str(parent.OUT),
        Base_Shared_FT_series_frozen=True,TFM_predictions_reused=True,new_TFM_calls=0,
        Specific_grouped_source_calibration_folds=3,Shared_fold_parameters_use_fit_rows_only=True,
        Shared_hyperparameters_preselected_in_v22_on_source_calibration=True,
        classification_utility='mean source-prior Acc and macro AUC minus .001 NLL',
        regression_utility='negative normalized MSE',simplex_regularization=[.01,.1,1.],
        Clean_regression_excludes_label_denoising_because_reference_labels_are_already_clean=True,
        final_query_labels_for_selection=False,development_adaptation_not_independent_paper_replication=True,
        frozen_FT_report_sha256=main.sha(raw.FT_REPORT),frozen_FT_Specific_report_sha256=main.sha(raw.FT_SPECIFIC_REPORT),Markdown_written=False))
    workers=[]
    for i in range(8):
        log=(OUT/'logs'/f'worker{i}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(i)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();workers.append(dict(worker=i,pid=p.pid))
    atomic_json(OUT/'workers.json',workers);print(workers,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker','report']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.phase=='launch':launch()
    elif a.phase=='worker':worker(a.shard)
    else:report()
