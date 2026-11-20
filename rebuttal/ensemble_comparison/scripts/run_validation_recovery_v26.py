"""Bounded held-out-TRAIN calibration followed by one frozen query evaluation."""
import json
import os
from pathlib import Path
import subprocess
import time
import validation_recovery_v26 as v
from prepare_data import atomic_json


def status(phase,**extra):
    row=dict(phase=phase,**extra);atomic_json(v.OUT/'controller_status.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def active(pid):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        return b'validation_recovery_v26.py' in cmd
    except FileNotFoundError:return False


def run_phase(phase,*extra):
    log=v.OUT/f'controller_{phase}.log'
    cmd=[str(v.RUN/'.venv/bin/python'),'-u',str(Path(v.__file__)),phase,*extra]
    with log.open('a') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{phase} failed: {log}')


def workers(phase,label,ratefile=None):
    children=[];execution=[]
    for gpu in range(8):
        log=v.RUN/'logs'/f'validation_recovery_v26_{label}_gpu{gpu}.log'
        cmd=[str(v.RUN/'.venv/bin/python'),'-u',str(Path(v.__file__)),phase,'--shard',str(gpu),'--shards','8']
        if ratefile:cmd+=['--rate-file',str(ratefile)]
        f=log.open('a')
        p=subprocess.Popen(cmd,env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT)
        children.append((p,f));execution.append(dict(gpu=gpu,pid=p.pid,log=str(log)))
    atomic_json(v.OUT/f'{label}_execution.json',execution)
    for p,f in children:
        code=p.wait();f.close()
        if code:raise RuntimeError(f'{phase} worker exited {code}')
    if list((v.OUT/'errors').glob('*.json')):raise RuntimeError('Worker reported error; inspect errors before continuing')


def assert_validation_complete():
    data=json.loads((v.OUT/'validation_selection.json').read_text())
    for block in ['five','seven']:
        tried=[r for r in data['all_trials'] if r['block']==block]
        if len(tried)!=len(v.rates()):raise RuntimeError(f'Incomplete validation for {block}: {len(tried)}/{len(v.rates())}')
    return data


def audit(stage):
    cmd=[str(v.RUN/'.venv/bin/python'),'-u',str(v.RUN/'scripts/audit_validation_recovery_v26.py'),stage]
    log=v.OUT/f'controller_{stage}_audit.log'
    with log.open('a') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'Audit failed: {log}')


def main():
    v.plan();status('WAITING_INITIAL_VALIDATION')
    execution=json.loads((v.OUT/'initial_execution.json').read_text())
    while any(active(r['pid']) for r in execution):time.sleep(5)
    if list((v.OUT/'errors').glob('*.json')):raise RuntimeError('Initial worker error')
    run_phase('validation_report');data=assert_validation_complete()
    for iteration in range(2):
        if all(data['best'][b]['validation_goal_met'] for b in ['five','seven']):break
        status('REFINING_TRAIN_VALIDATION',iteration=iteration+1,best=data['best'])
        previous=v.rates();run_phase('refine');current=v.rates()
        added=[x for x in current if x not in previous]
        if not added:break
        ratefile=v.OUT/f'controller_round{iteration+1}_rates.json';atomic_json(ratefile,dict(rates=added))
        workers('validation_worker',f'refine{iteration+1}',ratefile)
        run_phase('validation_report');data=assert_validation_complete()
    status('AUDITING_TRAIN_VALIDATION',best=data['best']);audit('validation')
    if not (v.OUT/'frozen_parameters.json').exists():run_phase('freeze')
    frozen=json.loads((v.OUT/'frozen_parameters.json').read_text())
    status('RUNNING_ONCE_FROZEN_QUERY_EVALUATION',configuration=frozen)
    workers('query_worker','frozen_query')
    run_phase('query_report');status('AUDITING_QUERY_RESULTS');audit('query')
    status('COMPLETE',configuration=frozen,
        query_completion=json.loads((v.OUT/'query_completion.json').read_text()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('FAILED',error=str(error));raise
