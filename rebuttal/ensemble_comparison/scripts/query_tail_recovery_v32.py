"""One query evaluation of the frozen source-only extension-tail protocol."""
import argparse
import json
import os
import time
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import tail_recovery_v32 as experiment
from prepare_data import atomic_json,protocol


def worker(shard,shards):
    import warnings,torch
    from run_tabicl import Predictor,metrics
    v=experiment.source.v;root=experiment.OUT
    settings=json.loads((root/'parameters.json').read_text());settings_hash=v.prior.original.sha(root/'parameters.json')
    assert json.loads((root/'validation_audit.json').read_text())['status']=='PASS'
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0');failures=0
    jobs=[m for block,m in v.prior.jobs() if block=='seven']
    for meta in jobs[shard::shards]:
        try:
            pool=v.prior.original.prepare('regression','seven',meta)
            for family,rho in v.grid('seven'):
                env=v.environment(pool,meta,'seven',family,rho,np.arange(len(pool['y'])))
                core,info=v.prior.original.bank('regression','seven',meta,family,rho)
                policy=settings['policies'].get(family,dict(kind='base',cap=0.))
                for budget in [256,1024,4096,8192]:
                    path=root/'query'/str(budget)/str(meta['task_id'])/f'{family}_{rho}.npz'
                    if path.exists() and path.with_suffix('.json').exists():continue
                    X,y=experiment.context(env,policy['kind'],policy['cap'],budget);data=core|dict(X_source=X,y_source=y)
                    assert np.array_equal(X[:256],core['X_source']) and np.array_equal(y[:256],core['y_source'])
                    started=time.monotonic()
                    if budget==256:
                        old=v.prior.original.frozen.OUT/'input_banks/regression/seven'/str(meta['task_id'])/path.name
                        with np.load(old) as z:p=z['Base_prediction']
                    else:p,_=predictor(X,y,data['X_target'],members=1,random_state=0)
                    assert np.isfinite(p).all();v.prior.original.npwrite(path,prediction=p,y_target=data['y_target'])
                    atomic_json(path.with_suffix('.json'),dict(task='regression',block='seven',task_id=meta['task_id'],
                        dataset=meta['dataset'],family=family,rho=rho,requested_context=budget,actual_context=len(y),
                        policy=policy,scale=info['scale'],metrics=metrics(data['y_target'],p,'regression',info['scale']),
                        input_hashes={k:protocol.array_sha256(a) for k,a in data.items()},settings_sha256=settings_hash,
                        original_corrupted256_core_preserved=True,checkpoint_finetuned=False,seed=0,n_estimators=1,
                        extra_clean_TRAIN_labels_used=False,observed_source_extension_clipping=True,
                        development_only=True,original_paper_replication=False,elapsed_seconds=time.monotonic()-started))
            print('TAIL_QUERY_DATASET_COMPLETE',meta['dataset'],flush=True)
        except Exception:
            failures+=1;error=traceback.format_exc();print(error,flush=True)
            atomic_json(root/'errors'/f'query_{meta["task_id"]}.json',dict(error=error))
    if failures:raise SystemExit(1)


def report():
    rows=[]
    for budget in [256,1024,4096,8192]:
        paths=[p for p in (experiment.OUT/'query'/str(budget)).glob('*/*.json') if '.tmp.' not in p.name]
        if len(paths)!=924:continue
        records=[json.loads(p.read_text()) for p in paths]
        rows.append(dict(context=budget,datasets=33,conditions=924,
            r2=float(np.mean([r['metrics']['r2'] for r in records])),
            nrmse=float(np.mean([r['metrics']['nrmse'] for r in records])),
            actual_rows_mean=float(np.mean([r['actual_context'] for r in records]))))
    frame=pd.DataFrame(rows);frame.to_csv(experiment.OUT/'query_averages.csv',index=False)
    monotone=len(rows)==4 and bool(np.all(np.diff(frame.r2)>0) and np.all(np.diff(frame.nrmse)<=0))
    gain=float(frame.r2.iloc[-1]-frame.r2.iloc[0]) if len(rows)==4 else None
    small=bool(monotone and .005<=gain<=.08)
    atomic_json(experiment.OUT/'query_status.json',dict(status='SMALL_MONOTONE_QUERY_CURVE' if small else 'TARGET_NOT_MET_OR_INCOMPLETE',
        complete_budgets=len(rows),query_monotone=monotone,final_R2_gain=gain,small_gain=small,
        does_not_automatically_replace_main_protocol=True,new_development_protocol=True))
    print(frame.to_string(index=False),flush=True)


def launch():
    from continue_tabace_v28 import launch as spawn,save,RUN
    path=experiment.OUT/'query_execution.json';assert not path.exists()
    state=dict(status='RUNNING',pid=os.getpid(),workers=[]);children=[]
    for shard,gpu in enumerate([0,5]):
        log=RUN/'logs'/f'tail_recovery_v32_query_gpu{gpu}.log'
        proc=spawn('query_tail_recovery_v32.py',['worker','--shard',str(shard),'--shards','2'],log,gpu)
        row=dict(shard=shard,gpu=gpu,pid=proc.pid,log=str(log));children.append((proc,row));state['workers'].append(row)
    while True:
        for proc,row in children:
            if proc.poll() is not None:row['exit_code']=proc.returncode
        save(path,state)
        if all(p.poll() is not None for p,_ in children):break
        time.sleep(10)
    report();state['status']='QUERY_FINISHED_REQUIRES_FULL_AUDIT';save(path,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','report','launch']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    worker(a.shard,a.shards) if a.phase=='worker' else globals()[a.phase]()
