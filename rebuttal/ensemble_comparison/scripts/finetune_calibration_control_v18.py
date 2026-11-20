"""Calibration-information-matched FT controls; no new contexts or TFM calls."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import finetune_tabace_main_v1 as main
import finetune_source_tabace_v5 as source
import finetune_tabace_hierarchical_v4 as hierarchy
from prepare_data import RUN,atomic_json
from run_tabicl import metrics,normalize

OUT=RUN/'results/finetune_calibration_control_v18'
SELECTED=RUN/'results/finetune_final_selected_v13'


def rel(job):return Path('jobs')/job['task']/job['track']/str(job['meta']['task_id'])


def prediction(c,policy):
    if policy['kind']=='identity':return c['base']
    p=np.asarray(policy['parameters'])
    if c['task']=='classification':
        logits=np.log(np.clip(c['base'],1e-12,1.))*np.exp(-p[0])+p[1:]
        return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
    return policy['mu']+policy['sigma']*(p[0]*(c['base']-policy['mu'])/policy['sigma']+p[1])


def weights(c):
    if c['task']=='classification':
        y=c['y'];labels,counts=np.unique(y,return_counts=True);lookup={int(k):len(y)/(len(labels)*n) for k,n in zip(labels,counts)}
        return np.asarray([lookup[int(k)] for k in y])/len(y)
    return np.full(len(c['y']),1/len(c['y'])/max(c['meta']['scale']**2,1e-12))


def score(cells,policy):
    values=[]
    for c in cells:
        p=prediction(c,policy)
        loss=-np.log(np.clip(p[np.arange(len(c['y'])),c['y'].astype(int)],1e-12,1.)) if c['task']=='classification' else (p-c['y'])**2
        values.append(float(np.dot(weights(c),loss)))
    return float(np.mean(values))


def fit(cells,task,reg,mu,sd):
    count=cells[0]['base'].shape[1] if task=='classification' else 1
    initial=np.zeros(count+1) if task=='classification' else np.array([1.,0.])
    def objective(p):
        value=0.;gradient=np.zeros_like(p)
        for c in cells:
            w=weights(c);truth=c['y']
            if task=='classification':
                logp=np.log(np.clip(c['base'],1e-12,1.));scaled=logp*np.exp(-p[0]);z=scaled+p[1:]
                q=np.exp(z-logsumexp(z,axis=1,keepdims=True));delta=q.copy();delta[np.arange(len(truth)),truth.astype(int)]-=1
                value+=float(np.dot(w,-np.log(np.clip(q[np.arange(len(truth)),truth.astype(int)],1e-12,1.))))
                gradient[0]+=float(np.sum(w[:,None]*delta*(-scaled)));gradient[1:]+=(w[:,None]*delta).sum(0)
            else:
                x=(c['base']-mu)/sd;pred=mu+sd*(p[0]*x+p[1]);error=pred-truth
                value+=float(np.dot(w,error**2));gradient+=2*np.array([np.dot(w*error,sd*x),np.dot(w*error,np.full(len(x),sd))])
        value/=len(cells);gradient/=len(cells)
        penalty=p.copy() if task=='classification' else p-np.array([1.,0.])
        value+=reg*np.mean(penalty**2);gradient+=2*reg*penalty/len(p)
        return value,gradient
    bounds=[(-2,2)]+[(-5,5)]*count if task=='classification' else [(-4,4),(-5,5)]
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=150,ftol=1e-10))
    assert np.isfinite(result.x).all()
    return dict(kind='logit_temperature_bias' if task=='classification' else 'affine',parameters=result.x.tolist(),reg=reg,
        mu=mu,sigma=sd,fit_success=bool(result.success),iterations=int(result.nit))


def choose(train,held,all_cells,task,mu,sd):
    identity=dict(kind='identity');best=(score(held,identity),None,identity);trials=[]
    for reg in [.001,.01,.1]:
        p=fit(train,task,reg,mu,sd);value=score(held,p);trials.append(dict(reg=reg,held_TRAIN_loss=value))
        if value<best[0]-1e-6:best=(value,reg,p)
    return (identity if best[1] is None else fit(all_cells,task,best[1],mu,sd)),trials


def run(job):
    dest=OUT/rel(job)
    if (dest/'completion.json').exists():return
    parent=SELECTED/rel(job);pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
    with np.load(parent/'split.npz') as z:split={k:z[k] for k in z.files}
    _,clean_y,_=main.flat(pool);mu=float(clean_y[split['train']].mean());sd=max(float(clean_y[split['train']].std()),1e-8)
    held_ids=hierarchy.calibration_split(job,pool,split);cells=[];train=[];held=[]
    for path in sorted((parent/'calibration').glob('*.npz')):
        c=hierarchy.read_cell(path);ids=hierarchy.row_map(job,split,pool,c['meta']['family'],c['meta']['rho']);mask=np.isin(ids,held_ids)
        cells.append(c);train.append(hierarchy.subset(c,~mask));held.append(hierarchy.subset(c,mask))
    start=time.perf_counter();global_policy,trials=choose(train,held,cells,job['task'],mu,sd);families={};family_trials={}
    for family in ['original']+main.families(job['track']):
        families[family],family_trials[family]=choose([c for c in train if c['meta']['family']==family],
            [c for c in held if c['meta']['family']==family],[c for c in cells if c['meta']['family']==family],job['task'],mu,sd)
    atomic_json(dest/'policies.json',dict(status='FROZEN_BEFORE_QUERY_SCORING',global_policy=global_policy,family_policies=families,
        global_trials=trials,family_trials=family_trials,fitting_seconds=time.perf_counter()-start,
        source_calibration_information_identical_to_TabACE=True,additional_TFM_forward_calls=0,
        no_alternative_contexts=True,final_query_labels_used_for_policy_selection=False,
        held_original_row_positions=held_ids.tolist(),resampled_duplicates_grouped=True))
    rows=[]
    for block in (['clean','five'] if job['track']=='five' else ['seven']):
        for family,rho in main.contexts.case_grid(block):
            c=hierarchy.read_cell(parent/'query'/block/f'{family}_{rho}.npz')
            for name,p in [('FT+Calibration',global_policy),('FT+FamilyCalibration',families[family])]:
                values=metrics(c['y'],prediction(c,p),job['task'],c['meta']['scale'])
                rows.append(dict(task=job['task'],block=block,task_id=job['meta']['task_id'],data_id=job['meta']['data_id'],
                    dataset=job['meta']['dataset'],family=family,rho=rho,method=name,**values))
    pd.DataFrame(rows).to_csv(dest/'query_detail.csv',index=False)
    atomic_json(dest/'completion.json',dict(status='COMPLETE',methods=['FT+Calibration','FT+FamilyCalibration'],extra_TFM_calls=0))


def worker(shard,shards):
    for job in main.jobs()[shard::shards]:
        try:run(job);print('CALIBRATION_CONTROL_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True);atomic_json(OUT/rel(job)/'error.json',dict(job=job,error=error))
        gc.collect()


def launch():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_CALIBRATION_CONTROL_FITS',source_calibration_labels_matched=True,
        classification_control='temperature plus class bias',regression_control='affine calibration in clean TRAIN units',
        regularization_grid=[.001,.01,.1],held_original_row_disjoint=True,selection_on_final_queries=False,
        all_selected_backbone_checkpoints_unchanged=True,extra_TFM_calls=0,seed=0,Markdown_written=False))
    while not (SELECTED/'completion.json').exists():time.sleep(2)
    children=[]
    for shard in range(8):
        log=(OUT/'logs'/f'worker{shard}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(shard),'--shards','8'],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(dict(worker=shard,pid=p.pid));atomic_json(OUT/'workers.json',children)
        print('CALIBRATION_CONTROL_WORKER_STARTED',shard,p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','launch']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='worker':worker(a.shard,a.shards)
    else:launch()
