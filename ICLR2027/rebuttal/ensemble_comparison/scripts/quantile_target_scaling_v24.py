"""Explicit source-only target-transform adaptation of the v24 fixed environments.

No weight finetuning, no query-label fitting, no score-range selection. This is
reported as an adapted method, never silently substituted for Native-1.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import tabicl_regression_scaling_v24 as env
from prepare_data import atomic_json,protocol

OUT=env.OUT/'quantile_target_adaptation'


def materialize_environment(track,meta,block,family,rho,pool):
    base,info=env.old.bank('regression',block,meta,family,rho)
    if block=='seven':
        A,b,ids,_=env.seven_environment(pool,family,rho,base['X_target'])
        order=env.stratified_order(pool['y'][ids],meta['task_id'],0)
    else:
        from tabicl_semisynthetic_five_v22 import corrupt_data
        clean=base|dict(X_source=pool['X'].copy(),y_source=pool['y'].copy())
        changed=clean if block=='clean' else corrupt_data(clean,family,rho,env.old.frozen.configuration()['regression_five']['families'][family]['parameters'])
        A,b=changed['X_source'],changed['y_source'];ids=np.arange(len(b))
        order=env.stratified_order(pool['y'],meta['task_id'],0)
    actual={k:protocol.array_sha256(v) for k,v in dict(X=A,y=b,ids=ids,order=order).items()}
    frozen=json.loads((env.OUT/'environments'/block/str(meta['task_id'])/f'{family}_{rho}.json').read_text())
    assert actual==frozen['hashes']
    return base,info,A,b,ids,order


def plan():
    if (OUT/'plan.json').exists():return
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_ADAPTED_PREDICTIONS',seed=0,n_estimators=1,
        adaptation='source-only Gaussian quantile target transform, inverse-transformed predictions',
        n_quantiles='min(1000,actual context size)',random_state=0,query_labels_for_fit=False,
        finetuning=False,underlying_environment='unchanged v24 fixed full source environment',
        original_paper_replication=False,development_only=True,all_trials_reported=True,
        max_configurations=1,no_numeric_score_acceptance_bounds=True,script_sha256=env.old.sha(__file__)))


def worker(shard,shards,only=None,pilot=False):
    import warnings,torch
    from run_tabicl import Predictor,metrics
    from seven_shifts import FAMILIES
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0')
    jobs=env.jobs()
    if only:jobs=[(k,m) for k,m in jobs if str(m['task_id']) in only.split(',')]
    for track,meta in jobs[shard::shards]:
        try:
            pool=env.old.prepare('regression',track,meta)
            grid=([('clean','original',0)]+[('five',f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]) if track=='five' else [('seven',f,r) for f in FAMILIES for r in [20,50,80,100]]
            if pilot:grid=[(k,f,r) for k,f,r in grid if k=='clean' or (r==100 and f in ['boundary','mixed','label','selection','temporal'])]
            for block,family,rho in grid:
                base,info,A,b,ids,order=materialize_environment(track,meta,block,family,rho,pool)
                for budget in env.BUDGETS:
                    out=OUT/'predictions'/str(budget)/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                    if out.exists() and out.with_suffix('.json').exists():continue
                    idx=order[:budget];X=A[idx];y=b[idx];start=time.monotonic()
                    mapper=QuantileTransformer(n_quantiles=min(1000,len(y)),output_distribution='normal',random_state=0)
                    targets=mapper.fit_transform(y[:,None]).ravel()
                    transformed,_=predictor(X,targets,base['X_target'],members=1,random_state=0)
                    pred=mapper.inverse_transform(transformed[:,None]).ravel();assert np.isfinite(pred).all()
                    hashes={k:protocol.array_sha256(v) for k,v in (base|dict(X_source=X,y_source=y)).items()}
                    rawpath=env.OUT/'predictions'/str(budget)/block/str(meta['task_id'])/out.name
                    raw=json.loads(rawpath.with_suffix('.json').read_text());assert raw['input_hashes']==hashes
                    env.old.npwrite(out,prediction=pred,y_target=base['y_target'],transformed_prediction=transformed,
                        source_environment_positions=idx,transformed_source_y=targets,
                        fitted_quantiles=mapper.quantiles_,fitted_references=mapper.references_)
                    atomic_json(out.with_suffix('.json'),dict(task='regression',block=block,task_id=meta['task_id'],dataset=meta['dataset'],family=family,rho=rho,
                        requested_context=budget,actual_context=len(y),seed=0,n_estimators=1,input_hashes=hashes,scale=info['scale'],
                        metrics=metrics(base['y_target'],pred,'regression',info['scale']),raw_metrics=raw['metrics'],
                        source_only_target_adaptation=True,query_labels_used_for_fit=False,finetuning=False,
                        elapsed_seconds=time.monotonic()-start,development_only=True))
            print('ADAPTED_DATASET_COMPLETE',track,meta['dataset'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{track}_{meta["task_id"]}.json',dict(error=error))


def report():
    rows=[]
    for p in (OUT/'predictions').glob('*/*/*/*.json'):
        if p.name.endswith('.tmp.json'):continue
        rec=json.loads(p.read_text());rows.append({k:v for k,v in rec.items() if k not in ['metrics','raw_metrics','input_hashes']}|rec['metrics'])
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(OUT/'detail.csv',index=False)
    per=df.groupby(['requested_context','block','task_id','dataset'],as_index=False).agg(conditions=('family','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    per['complete']=per.conditions==per.block.map(dict(clean=1,five=20,seven=28));per.to_csv(OUT/'per_dataset.csv',index=False)
    avg=per[per.complete].groupby(['requested_context','block'],as_index=False).agg(datasets=('task_id','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    avg.to_csv(OUT/'averages.csv',index=False);print(avg.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--only');p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='worker':worker(a.shard,a.shards,a.only,a.pilot)
    else:report()
