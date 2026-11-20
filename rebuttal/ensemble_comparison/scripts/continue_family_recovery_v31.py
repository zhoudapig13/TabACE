"""Audit v30, solve the registered grid, evaluate only a validated frozen rule."""
import json
import os
from pathlib import Path
import time
from continue_tabace_v28 import launch,save,RUN

ROOT=RUN/'results/tabicl_family_recovery_v31'
SOURCE=RUN/'results/tabicl_continuous_recovery_v30'


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
    except FileNotFoundError:return False


def action(script,arguments,logname):
    process=launch(script,arguments,RUN/'logs'/logname)
    return process.wait()


def main():
    file=ROOT/'execution.json';assert not file.exists(),'Avoid duplicate frozen-protocol orchestration'
    source_state=json.loads((SOURCE/'execution.json').read_text());pid=source_state['supervisor_pid']
    state=dict(status='WAITING_FOR_SOURCE_VALIDATION',pid=os.getpid(),source_supervisor_pid=pid,workers=[]);save(file,state)
    while alive(pid):time.sleep(10)
    source_state=json.loads((SOURCE/'execution.json').read_text())
    if source_state.get('report_exit_code')!=0 or any(r.get('exit_code')!=0 for r in source_state['workers']):
        state['status']='SOURCE_VALIDATION_FAILED_REQUIRES_REPAIR';save(file,state);return
    state['status']='AUDITING_SOURCE_VALIDATION';save(file,state)
    code=action('audit_continuous_recovery_v30.py',[],'continuous_recovery_v30_full_audit.log')
    state['source_audit_exit_code']=code
    if code:state['status']='SOURCE_AUDIT_FAILED';save(file,state);return
    state['status']='SOLVING_PREDECLARED_SOURCE_GRID';save(file,state)
    code=action('family_recovery_v31.py',['solve'],'family_recovery_v31_source_selection.log')
    state['selection_exit_code']=code
    if code or not (ROOT/'parameters.json').exists():
        state['status']='NO_SOURCE_VALIDATED_RULE_REQUIRES_NEW_HYPOTHESIS';save(file,state);return
    state['status']='FROZEN_RULE_QUERY_EVALUATION';save(file,state);children=[]
    for shard,gpu in enumerate([0,5]):
        log=RUN/'logs'/f'family_recovery_v31_query_gpu{gpu}.log'
        process=launch('query_family_recovery_v31.py',['worker','--shard',str(shard),'--shards','2'],log,gpu)
        record=dict(shard=shard,gpu=gpu,pid=process.pid,log=str(log));children.append((process,record));state['workers'].append(record)
    while True:
        for process,record in children:
            if process.poll() is not None:record['exit_code']=process.returncode
        save(file,state)
        if all(p.poll() is not None for p,_ in children):break
        time.sleep(10)
    code=action('query_family_recovery_v31.py',['report'],'family_recovery_v31_query_report.log')
    state['query_report_exit_code']=code;state['status']='QUERY_EVALUATION_FINISHED_REQUIRES_FULL_AUDIT';save(file,state)


if __name__=='__main__':main()
