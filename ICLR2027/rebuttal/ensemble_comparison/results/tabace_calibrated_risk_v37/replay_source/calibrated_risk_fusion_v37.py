"""Regression reliability calibration with grouped out-of-fold TRAIN risks.

Adds the dataset-level reliability signal missing from global-only priors.
Four deterministic TRAIN-row folds prevent a query's own label from entering
its reliability estimates. All choices are frozen before original query labels
are loaded. The TFM banks, checkpoints, Base predictions and context counts
are immutable. This is an explicitly adapted reconstruction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import benchmark256_v22 as frozen
import tabace_context_reproduction_v28 as reference
from prepare_data import RUN,atomic_json
from run_tabicl import metrics

ROOT=RUN/'results/tabace_calibrated_risk_v37'
PROFILES=[dict(shrinkage=k,power=b,agreement=g) for k in [0.,32.,128.] for b in [.5,1.,2.,4.] for g in [0.,.25,1.,4.]]


def load(source,role,budget,block):
    cohort={str(m['task_id']):m for m in frozen.cohort('regression',block)};cells=[]
    for path in sorted((source/'banks'/role/str(budget)/'regression'/block).glob('*/*.npz')):
        if '.tmp.' in path.name:continue
        meta=json.loads(path.with_suffix('.json').read_text());meta['data_id']=cohort[str(meta['task_id'])]['data_id']
        with np.load(path) as z:c=dict(base=z['base'],bank=z['members'],y=z['y_target'])
        assert c['bank'].shape[0]==21
        cells.append(c|dict(meta=meta,path=path,key=str(path.relative_to(source/'banks'/role/str(budget)))))
    expected=len(cohort)*(1 if block=='clean' else 20 if block=='five' else 28)
    assert len(cells)==expected,(source,role,budget,block,len(cells),expected)
    return cells


def row_folds(meta,track,n):
    path=reference.OUT/'validation_splits/regression'/track/f'{meta["task_id"]}.npz'
    with np.load(path) as z:ids=z['validation_original_ids']
    assert len(ids)==n and len(np.unique(ids))==n
    return np.asarray([int.from_bytes(hashlib.sha256(f'0:{meta["data_id"]}:{int(i)}'.encode()).digest()[:8],'big')%4 for i in ids],dtype=int)


def prepare(cells,m,block):
    tasks=sorted({str(c['meta']['task_id']) for c in cells});groups={task:i for i,task in enumerate(tasks)}
    matrices=[];truth=[];group=[];fold=[];starts=[];lengths=[];y_by_task={};f_by_task={};offset=0
    for c in cells:
        task=str(c['meta']['task_id']);n=len(c['y']);starts.append(offset);lengths.append(n);offset+=n
        if task not in y_by_task:
            y_by_task[task]=c['y'];f_by_task[task]=row_folds(c['meta'],'seven' if block=='seven' else 'five',n)
        assert np.array_equal(y_by_task[task],c['y']),'Regression calibration rows changed across conditions'
        matrices.append(np.column_stack([c['base'],c['bank'][:m].T]));truth.append(c['y'])
        group.append(np.full(n,groups[task],dtype=int));fold.append(f_by_task[task])
    P=np.concatenate(matrices);Y=np.concatenate(truth);G=np.concatenate(group);F=np.concatenate(fold)
    risks=np.zeros((5,len(tasks),m+1));sigmas=np.ones((5,len(tasks)));counts=np.zeros((5,len(tasks)))
    for split in range(5):
        for task,j in groups.items():
            selected=(G==j)&((F!=split) if split<4 else True)
            rows=(f_by_task[task]!=split) if split<4 else np.ones(len(y_by_task[task]),dtype=bool)
            assert rows.sum()>0
            scale=float(np.std(y_by_task[task][rows]));scale=scale if scale>1e-6 else 1.
            risks[split,j]=np.mean((P[selected]-Y[selected,None])**2,axis=0)/(scale*scale)
            sigmas[split,j]=scale;counts[split,j]=int(rows.sum())
    starts=np.asarray(starts);lengths=np.asarray(lengths)
    sst=np.asarray([np.sum((c['y']-c['y'].mean())**2) for c in cells])
    return dict(P=P,Y=Y,G=G,F=F,risks=risks,sigmas=sigmas,counts=counts,tasks=tasks,
        prior=np.median(risks,axis=1),starts=starts,lengths=lengths,sst=sst,
        scales=np.asarray([c['meta']['scale'] for c in cells]))


def combine(P,risk,prior,count,scale,profile):
    weight=count/(count+profile['shrinkage'])
    mixed=weight[:,None]*risk+(1-weight[:,None])*prior
    logw=-profile['power']*np.log(np.clip(mixed,0,1e12)+.05)
    z=np.exp(logw-logw.max(1,keepdims=True));z/=z.sum(1,keepdims=True)
    if profile['agreement']:
        center=(z*P).sum(1)
        logw-=profile['agreement']*np.abs(P-center[:,None])/scale[:,None]
        z=np.exp(logw-logw.max(1,keepdims=True));z/=z.sum(1,keepdims=True)
    return (z*P).sum(1)


def score(data,prediction):
    sse=np.add.reduceat((prediction-data['Y'])**2,data['starts'])
    r2=np.where(data['sst']>0,1-sse/np.maximum(data['sst'],1e-300),np.where(sse==0,1.,0.))
    nrmse=np.sqrt(sse/data['lengths'])/data['scales']
    result=dict(r2=float(r2.mean()),nrmse=float(nrmse.mean()))
    return result,.5*(result['r2']-result['nrmse'])


def fit(cells,m,block,fallback_predictions):
    d=prepare(cells,m,block);fallback=np.concatenate(fallback_predictions)
    baseline,value=score(d,fallback);best=None;best_value=value;best_oof=fallback;trials=[]
    for profile in PROFILES:
        prediction=combine(d['P'],d['risks'][d['F'],d['G']],d['prior'][d['F']],
            d['counts'][d['F'],d['G']],d['sigmas'][d['F'],d['G']],profile)
        scores,utility=score(d,prediction);trials.append(profile|dict(source_grouped_OOF_scores=scores,utility=utility))
        if utility>best_value+1e-10:best=profile;best_value=utility;best_oof=prediction
    policy=None
    if best is not None:
        policy=dict(kind='calibrated_risk_fusion',m=m,profile=best,
            group_stats={task:dict(risk=d['risks'][4,j].tolist(),prior=d['prior'][4].tolist(),
                count=float(d['counts'][4,j]),scale=float(d['sigmas'][4,j])) for j,task in enumerate(d['tasks'])},
            calibration='Four grouped TRAIN-row folds; own fold excluded from all reliability statistics',
            original_query_labels_used=False,clean_TRAIN_calibration_supervision=True)
    oof={c['key']:best_oof[start:start+n] for c,start,n in zip(cells,d['starts'],d['lengths'])}
    return policy,oof,dict(fallback_scores=baseline,trials=trials,chosen_profile=best)


def predict(c,policy):
    if policy is None:return c['base']
    s=policy['group_stats'][str(c['meta']['task_id'])];n=len(c['base']);m=policy['m']
    P=np.column_stack([c['base'],c['bank'][:m].T])
    return combine(P,np.broadcast_to(s['risk'],P.shape),np.broadcast_to(s['prior'],P.shape),
        np.full(n,s['count']),np.full(n,s['scale']),policy['profile'])


def run(source,out,budget,block):
    cells=load(source,'validation',budget,block)
    shared,shared_oof,details=fit(cells,10,block,[c['base'] for c in cells])
    policies=out/'policies'/str(budget)/'regression'/block
    atomic_json(policies/'Shared.json',dict(status='FROZEN_BEFORE_QUERY',model=shared,selection=details,seed=0))
    specific={}
    for family in sorted({c['meta']['family'] for c in cells}):
        selected=[c for c in cells if c['meta']['family']==family]
        model,_,detail=fit(selected,21,block,[shared_oof[c['key']] for c in selected])
        specific[family]=model
        atomic_json(policies/f'Specific_{family}.json',dict(status='FROZEN_BEFORE_QUERY',model=model,
            fallback='Shared' if model is None else None,selection=detail,seed=0))
    # Original query labels are loaded only after every policy is frozen.
    query=load(source,'query',budget,block);rows=[]
    for c in query:
        sh=predict(c,shared);sp=sh if specific[c['meta']['family']] is None else predict(c,specific[c['meta']['family']])
        path=out/'predictions'/str(budget)/'regression'/block/str(c['meta']['task_id'])/c['path'].name
        reference.scaling.npwrite(path,Base=c['base'],Shared=sh,Specific=sp,y_target=c['y'])
        predictions=dict(Base=c['base'],Shared=sh,Specific=sp)
        meta=c['meta'];summary={k:meta[k] for k in ['task','block','task_id','dataset','family','rho','requested_context','actual_context','scale']}
        summary.update(input_hashes=meta['source_input_hashes'],methods={k:metrics(c['y'],p,'regression',meta['scale']) for k,p in predictions.items()},
            provisional_Base_calibration=meta['provisional_Base_calibration'],seed=0,
            source_bank=str(c['path']),clean_TRAIN_calibration_supervision=True,
            original_query_labels_used_for_fit=False,calibration_row_group_folds=4,
            adaptation_version='v37',original_paper_replication=False,development_only=True,
            available_observed_pool_rows=meta.get('available_observed_pool_rows',meta['actual_context']))
        atomic_json(path.with_suffix('.json'),summary)
        for name,values in summary['methods'].items():rows.append({k:v for k,v in summary.items() if k not in ['methods','input_hashes']}|dict(method=name)|values)
    pd.DataFrame(rows).to_csv(policies/'query_detail.csv',index=False)
    atomic_json(policies/'completion.json',dict(status='COMPLETE',conditions=len(query),four_grouped_TRAIN_folds=True,
        original_query_labels_used_for_fit=False,adaptation_marker='dagger',TFM_banks_unchanged=True))
    print('CALIBRATED_RISK_COMPLETE',source.name,budget,block,len(query),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--budget',type=int,required=True);p.add_argument('--block',choices=['clean','five','seven'],required=True);a=p.parse_args()
    run(a.source,a.out,a.budget,a.block)
