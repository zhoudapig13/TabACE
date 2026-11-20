"""Evaluate the validated five-corruption block while seven-shift work continues."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import family_recovery_v31 as family
import query_family_recovery_v31 as query
from prepare_data import atomic_json

ROOT=family.OUT/'frozen_five_query'


def freeze():
    path=ROOT/'parameters.json';assert not path.exists(),'Do not refreeze from query feedback'
    selected=json.loads((family.OUT/'selection.json').read_text())['best']['five']
    assert selected['all_constraints_met']
    assert json.loads((family.source.OUT/'validation_audit.json').read_text())['status']=='PASS'
    atomic_json(path,dict(status='FROZEN_BEFORE_QUERY_EVALUATION',fractions={'five':selected['fractions']},
        selection_sha256=family.sha(family.OUT/'selection.json'),source_validation_goal_met=True,
        validated_source_means=selected['validation_means'],seed=0,n_estimators=1,
        development_only=True,original_paper_replication=False,extra_pre_corruption_TRAIN_information=True,
        original_corrupted256_core_preserved=True,classification_clean_and_seven_unchanged=True))


def worker(shard,shards):
    rows=[(block,meta) for block,meta in family.source.v.prior.jobs() if block=='five']
    family.source.v.prior.jobs=lambda:rows
    family.OUT=ROOT
    query.worker(shard,shards)


def report():
    rows=[]
    for budget in [256,1024,4096,8192]:
        paths=[p for p in (ROOT/'query'/str(budget)/'five').glob('*/*.json') if '.tmp.' not in p.name]
        if len(paths)!=700:continue
        records=[json.loads(p.read_text()) for p in paths]
        rows.append(dict(context=budget,datasets=35,conditions=700,
            r2=float(np.mean([r['metrics']['r2'] for r in records])),
            nrmse=float(np.mean([r['metrics']['nrmse'] for r in records])),
            actual_rows_mean=float(np.mean([r['actual_context'] for r in records]))))
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'query_averages.csv',index=False)
    monotone=len(rows)==4 and bool(np.all(np.diff(frame.r2)>0) and np.all(np.diff(frame.nrmse)<=0))
    gain=float(frame.r2.iloc[-1]-frame.r2.iloc[0]) if len(rows)==4 else None
    small=bool(monotone and .005<=gain<=.08)
    atomic_json(ROOT/'query_status.json',dict(status='SMALL_MONOTONE_QUERY_CURVE' if small else 'TARGET_NOT_MET_OR_INCOMPLETE',
        complete_budgets=len(rows),query_monotone=monotone,final_R2_gain=gain,small_gain=small,
        new_development_protocol=True,does_not_automatically_replace_main_protocol=True))
    print(frame.to_string(index=False),flush=True)


def launch():
    from continue_tabace_v28 import launch as spawn,save,RUN
    import time
    path=ROOT/'execution.json';assert not path.exists(),'Avoid duplicate frozen query evaluation'
    state=dict(status='RUNNING',pid=os.getpid(),workers=[]);children=[]
    for shard,gpu in enumerate([0,5]):
        log=RUN/'logs'/f'five_family_v31_query_gpu{gpu}.log'
        proc=spawn(Path(__file__).name,['worker','--shard',str(shard),'--shards','2'],log,gpu)
        row=dict(shard=shard,gpu=gpu,pid=proc.pid,log=str(log));children.append((proc,row));state['workers'].append(row)
    while True:
        for proc,row in children:
            if proc.poll() is not None:row['exit_code']=proc.returncode
        save(path,state)
        if all(p.poll() is not None for p,_ in children):break
        time.sleep(10)
    report();state['status']='QUERY_FINISHED_REQUIRES_FULL_AUDIT';save(path,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','worker','report','launch']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    worker(a.shard,a.shards) if a.phase=='worker' else globals()[a.phase]()
