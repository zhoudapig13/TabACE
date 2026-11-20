"""One frozen seven-shift probe selected on held-out TRAIN monotonicity.

The desired small-gain upper bound was NOT met on validation. This probe tests
transfer of the least-gain monotone candidate without silently accepting it as
the final benchmark. Original v22 core and prior query results remain immutable.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import validation_recovery_v26 as v
from prepare_data import atomic_json,protocol

OUT=v.OUT/'frozen_seven_monotonic_query'


def freeze():
    file=OUT/'parameters.json'
    assert not file.exists(),'Never refreeze this configuration from query feedback'
    table=pd.read_csv(v.OUT/'validation_averages.csv')
    ref=pd.read_csv(v.OUT/'validation_base256.csv').set_index('block').loc['seven']
    candidates=[]
    for rate,rows in table[table.block=='seven'].groupby('repair_probability'):
        rows=rows.sort_values('requested_context')
        if len(rows)!=3 or not (rows.datasets==33).all():continue
        r2=np.r_[ref.r2,rows.r2.values];rmse=np.r_[ref.nrmse,rows.nrmse.values]
        if np.all(np.diff(r2)>0) and np.all(np.diff(rmse)<=0):
            candidates.append(dict(rate=float(rate),validation_r2=r2.tolist(),validation_nrmse=rmse.tolist(),final_gain=float(r2[-1]-r2[0])))
    assert candidates,'No source-validation monotone candidate'
    selected=min(candidates,key=lambda r:r['final_gain'])
    atomic_json(file,dict(status='FROZEN_BEFORE_NEW_QUERY_PREDICTIONS',repair_probability=selected['rate'],
        selection='Least final R2 gain among source-validation monotone candidates',
        validation=selected,all_monotone_candidates=candidates,
        validation_small_gain_bound_met=.005<=selected['final_gain']<=.08,
        validation_csv_sha256=v.prior.original.sha(v.OUT/'validation_averages.csv'),
        original_query_labels_used_for_selection=False,development_only=True,independent_test_claim=False,
        corrupted256_core_unchanged=True,clean_and_classification_unchanged=True,
        previous_query_scores_have_been_viewed=True,query_result_will_not_automatically_replace_main_protocol=True,
        seed=0,n_estimators=1,nominal_strengths=[.2,.5,.8,1.],context_budgets=v.BUDGETS))
    print('FROZEN_SEVEN_MONOTONIC',selected,flush=True)


def worker(shard,shards):
    import warnings,torch
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    settings=json.loads((OUT/'parameters.json').read_text());rate=settings['repair_probability']
    predictor=Predictor('regression','cuda:0');failures=0
    jobs=[meta for block,meta in v.prior.jobs() if block=='seven']
    for meta in jobs[shard::shards]:
        try:
            pool=v.prior.original.prepare('regression','seven',meta)
            for family,rho in v.grid('seven'):
                env=v.environment(pool,meta,'seven',family,rho,np.arange(len(pool['y'])))
                core,info=v.prior.original.bank('regression','seven',meta,family,rho)
                for budget in v.BUDGETS:
                    out=OUT/'predictions'/str(budget)/str(meta['task_id'])/f'{family}_{rho}.npz'
                    if out.exists() and out.with_suffix('.json').exists():continue
                    X,y,mask=v.context(env,rate,budget);data=core|dict(X_source=X,y_source=y)
                    assert np.array_equal(X[:256],core['X_source']) and np.array_equal(y[:256],core['y_source'])
                    start=time.monotonic()
                    if budget==256:
                        origin=v.prior.original.frozen.OUT/'input_banks/regression/seven'/str(meta['task_id'])/out.name
                        with np.load(origin) as z:p=z['Base_prediction']
                    else:p,_=predictor(X,y,data['X_target'],members=1,random_state=0);origin=None
                    assert np.isfinite(p).all()
                    v.prior.original.npwrite(out,prediction=p,y_target=data['y_target'])
                    atomic_json(out.with_suffix('.json'),dict(task_id=meta['task_id'],dataset=meta['dataset'],
                        task='regression',block='seven',family=family,rho=rho,requested_context=budget,actual_context=len(y),
                        clean_extension_rows=int(mask.sum()),repair_probability=rate,scale=info['scale'],
                        metrics=metrics(data['y_target'],p,'regression',info['scale']),
                        input_hashes={k:protocol.array_sha256(a) for k,a in data.items()},
                        original_core256_preserved=True,seed=0,n_estimators=1,checkpoint_finetuned=False,
                        extra_clean_TRAIN_labels_used=bool(mask.sum()),elapsed_seconds=time.monotonic()-start,
                        reused_original_prediction=None if origin is None else str(origin),
                        settings_sha256=v.prior.original.sha(OUT/'parameters.json'),development_only=True))
            print('SEVEN_MONOTONIC_QUERY_DATASET_COMPLETE',meta['dataset'],flush=True)
        except Exception:
            failures+=1;error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{meta["task_id"]}.json',dict(error=error))
    if failures:raise SystemExit(1)


def report():
    rows=[]
    for budget in v.BUDGETS:
        files=[p for p in (OUT/'predictions'/str(budget)).glob('*/*.json') if '.tmp.' not in p.name]
        if len(files)!=924:continue
        records=[json.loads(p.read_text()) for p in files]
        rows.append(dict(context=budget,datasets=33,conditions=924,
            r2=float(np.mean([r['metrics']['r2'] for r in records])),
            nrmse=float(np.mean([r['metrics']['nrmse'] for r in records])),
            actual_rows_mean=float(np.mean([r['actual_context'] for r in records])),
            clean_extension_rows_mean=float(np.mean([r['clean_extension_rows'] for r in records]))))
    df=pd.DataFrame(rows);df.to_csv(OUT/'averages.csv',index=False)
    monotone=len(rows)==4 and np.all(np.diff(df.r2)>0) and np.all(np.diff(df.nrmse)<=0)
    small=monotone and .005<=float(df.r2.iloc[-1]-df.r2.iloc[0])<=.08
    atomic_json(OUT/'status.json',dict(status='SMALL_MONOTONE_QUERY_CURVE' if small else 'TARGET_NOT_MET_OR_INCOMPLETE',
        complete_contexts=len(rows),query_monotone=bool(monotone),small_gain=bool(small),rows=rows,
        does_not_automatically_replace_main_protocol=True,development_only=True))
    print(df.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','worker','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    worker(a.shard,a.shards) if a.phase=='worker' else globals()[a.phase]()
