"""Observe live training, launch each audit once, and preserve the full goal.

This process never fits adapters, restarts jobs, changes scores, modifies Base,
or declares the user goal complete. Failed audits remain explicit failures.
"""
import json
import os
import subprocess
import time
from pathlib import Path
from prepare_data import RUN,atomic_json

STAGES={'classification':'tabace_augmented_classifier_v54','regression':'tabace_bagged_specific_v56'}
ROOT=RUN/'results/tabace_hybrid_audits_v57'


def launch(key,script,args,state,children):
    path=RUN/'logs'/f'hybrid_v57_{key}.log';stream=path.open('a')
    p=subprocess.Popen([str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts'/script),*args],stdout=stream,stderr=subprocess.STDOUT,
        env=os.environ|{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},start_new_session=True)
    row=dict(key=key,pid=p.pid,script=script,arguments=args,log=str(path));children[key]=(p,row);state['tasks'].append(row)


def groups_complete(stage,task,blocks,budget):
    return all((RUN/'results'/stage/'policies'/str(budget)/task/block/'completion.json').exists() for block in blocks)


def full_complete(stage):
    return all(groups_complete(stage,t,['clean','five','seven'],b) for b in [256,1024,4096,8192] for t in ['classification','regression'])


def summarize(state):
    current=RUN/'results/tabace_goal_progress_latest.json';progress=json.loads(current.read_text())
    for stage in ['tabace_augmented_classifier_v54','tabace_bagged_specific_v56']:
        p=RUN/'results'/stage/'goal_audit.json'
        if p.exists():
            numeric=json.loads(p.read_text());progress['current_full_scope_stage']=stage
            for key in ['completed_metric_checks','passed_metric_checks','missing']:progress[key]=numeric[key]
            progress['failed_checks']=[c for c in numeric['checks'] if not c['pass_all']]
    progress.update(status='PROGRESS_NOT_COMPLETE',goal_satisfied=False,time_unix=time.time(),
        audit_watcher=dict(pid=os.getpid(),status=state['status'],tasks=state['tasks']),Markdown_written=False)
    atomic_json(current,progress)


def run():
    path=ROOT/'execution.json';assert not path.exists()
    state=dict(status='WATCHING_TRAINING_AND_AUDITING',pid=os.getpid(),tasks=[]);children={}
    launch('inherited','audit_inherited_v57.py',[],state,children)
    while True:
        for p,row in children.values():
            if p.poll() is not None:row['exit_code']=p.returncode
        for task,stage in STAGES.items():
            for b in [256,1024,4096,8192]:
                key=f'{task}_{b}'
                if key not in children and groups_complete(stage,task,['clean','seven'],b):
                    script='audit_augmented_classifier_v57.py' if task=='classification' else 'audit_bagged_specific_v57.py'
                    launch(key,script,['--budget',str(b)],state,children)
            key=f'numeric_{stage}'
            if key not in children and full_complete(stage):launch(key,'report_recovery_goal_stage.py',['--stage',stage],state,children)
        for stage in STAGES.values():
            execution=RUN/'results'/stage/'execution.json'
            if execution.exists():
                d=json.loads(execution.read_text());state.setdefault('training',{})[stage]=d
        state['updated_unix']=time.time();atomic_json(path,state);summarize(state)
        expected={'inherited'}|{f'{t}_{b}' for t in STAGES for b in [256,1024,4096,8192]}|{f'numeric_{s}' for s in STAGES.values()}
        if expected<=set(children) and all(p.poll() is not None for p,_ in children.values()):
            state['status']='AUDITS_COMPLETE_REQUIRES_GOAL_REVIEW' if all(p.returncode==0 for p,_ in children.values()) else 'AUDIT_FAILURE_REQUIRES_REPAIR'
            checks=[]
            for task,stage in STAGES.items():
                for b in [256,1024,4096,8192]:
                    q=RUN/'results'/stage/'raw_audits'/f'{b}.json'
                    if q.exists():checks.append(json.loads(q.read_text()))
            inherited=RUN/'results/tabace_bagged_specific_v56/inherited_prediction_audit_v57.json'
            total=sum(x.get('verified_conditions',0) for x in checks)+(json.loads(inherited.read_text())['verified_conditions'] if inherited.exists() else 0)
            state['raw_conditions_verified']=total
            state['full_scope_expected_conditions']=16604
            state['goal_declared_complete']=False
            atomic_json(path,state);summarize(state);break
        time.sleep(15)


if __name__=='__main__':run()
