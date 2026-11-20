"""Corruption-matched actual FT on the frozen source-only classification suite."""
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import finetune_tabace_main_v1 as main
import finetune_tabace_main_v2 as balanced
import finetune_source_benchmark_v5 as benchmark
from prepare_data import RUN,atomic_json

OUT=RUN/'results/finetune_source_tabace_v5'
_jobs=main.jobs
_bank=main.scaling.bank
_materialize=main.materialize


def jobs():return [j for j in _jobs() if j['task']=='classification' and j['track']=='five']


def bank(task,block,meta,family='original',rho=0):
    if task=='classification' and block=='five':
        cfg=json.loads((benchmark.OUT/'frozen.json').read_text())
        path=__import__('pathlib').Path(cfg['input_bank'])/str(meta['task_id'])/f'{family}_{rho}.npz'
        with np.load(path) as z:data={k:z[k] for k in ['X_source','y_source','X_target','y_target']}
        return data,json.loads(path.with_suffix('.json').read_text())
    return _bank(task,block,meta,family,rho)


def materialize(job,pool,positions,query_positions,family,rho):
    if job['task']=='classification' and job['track']=='five' and rho:
        clean=_materialize(job,pool,positions,query_positions,'original',0)
        cfg=json.loads((benchmark.OUT/'frozen.json').read_text())
        X,y,_=benchmark.corruptor.corrupt(clean['X_source'],clean['y_source'],family=family,rho=rho/100.,**cfg['parameters'])
        return clean|dict(X_source=X,y_source=y)
    return _materialize(job,pool,positions,query_positions,family,rho)


def setup():
    balanced.setup();main.OUT=OUT;main.jobs=jobs;main.scaling.bank=bank;main.materialize=materialize


def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return
    cfg=json.loads((benchmark.OUT/'frozen.json').read_text())
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_SOURCE_ONLY_FT_UPDATES',seed=0,context_cap=256,
        datasets=72,source_context_only=True,query_features_and_labels_unchanged=True,
        benchmark_frozen=cfg,benchmark_sha256=main.sha(benchmark.OUT/'frozen.json'),
        actual_weight_updates=True,steps=128,learning_rate=3e-6,additional_clean_official_TRAIN_labels=True,
        queries_for_training_selection_calibration='disjoint official TRAIN rows; final query excluded',
        calibration_and_checkpoint_query_rows_disjoint=True,nominal_strengths=[.2,.5,.8,1.],
        training_mixture='20% clean,80% exact frozen source-only corruptions',
        selection='40% clean balanced NLL +60% corrupted balanced NLL, source validation only',
        methods=['FT','FT+Shared','FT+Specific'],unfinetuned_Shared_Specific_not_run=True,
        earlier_query_feature_benchmark_results_preserved=True,development_only=True,not_archived_paper_replication=True,
        script_sha256=main.sha(__file__),time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),Markdown_written=False))


def report():
    paths=list((OUT/'jobs').glob('*/*/*/query_detail.csv'))
    if not paths:return
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);base=[]
    for _,r in df[df.method=='FT'].iterrows():
        _,record=bank(r['task'],r['block'],dict(task_id=r['task_id']),r['family'],int(r['rho']))
        base.append({k:r[k] for k in ['task','block','task_id','data_id','dataset','family','rho']}|dict(method='Base (paired source protocol)')|record['metrics'])
    df=pd.concat([pd.DataFrame(base),df],ignore_index=True);df.to_csv(OUT/'query_detail.csv',index=False)
    rows=[]
    for (block,method),g in df.groupby(['block','method']):
        rows.append(dict(task='classification',block=block,method=method,datasets=g.task_id.nunique(),complete=g.task_id.nunique()==72,
            accuracy=float(g.accuracy.mean()),auc=float(g.auc.mean())))
    pd.DataFrame(rows).to_csv(OUT/'averages.csv',index=False)
    done=len(list((OUT/'jobs').glob('*/*/*/completion.json')))
    atomic_json(OUT/'progress.json',dict(completed_jobs=done,expected_jobs=72,complete=done==72))
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print('SOURCE_FT_PROGRESS',done,'/72',flush=True)


def launch_after_base():
    waiting=json.loads((benchmark.OUT/'workers.json').read_text())
    while waiting:
        rest=[]
        for w in waiting:
            try:os.kill(w['pid'],0);rest.append(w)
            except ProcessLookupError:pass
        waiting=rest
        if waiting:time.sleep(2)
    if not (benchmark.OUT/'frozen.json').exists():benchmark.select()
    plan();children=[];(OUT/'logs').mkdir(exist_ok=True)
    for gpu in range(8):
        log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        process=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
            env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(dict(gpu=gpu,pid=process.pid));atomic_json(OUT/'workers.json',children)
        print('SOURCE_FT_WORKER_STARTED',gpu,process.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report','launch_after_base']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    setup()
    if a.phase=='plan':plan()
    elif a.phase=='worker':main.worker(a.shard,a.shards,steps=128)
    elif a.phase=='report':report()
    else:launch_after_base()
