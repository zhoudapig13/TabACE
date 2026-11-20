"""Await verified v66 jobs, audit all3920 paired cases, refresh full48 scope."""
import json
import os
from pathlib import Path
import subprocess
import time
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabace_balanced_seven_v66'
PYTHON=str(RUN/'.venv/bin/python')


def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'


def run():
    state=dict(status='WAITING_ON_VERIFIED_V66_CONTROLLER',pid=os.getpid(),audits=[],full_goal_checks=48)
    path=OUT/'raw_audit_execution_v72.json'
    while True:
        current=json.loads((OUT/'execution.json').read_text())
        state.update(controller_pid=current['pid'],controller_verified_live=alive(current['pid']),
                     generation_status=current['status'],updated_unix=time.time())
        if current['status']=='ALL_GROUPS_COMPLETE_REQUIRES_REPLAY_AND_FULL48_AUDIT':break
        if not state['controller_verified_live'] or 'FAILED' in current['status']:
            state['status']='V66_TERMINAL_REQUIRES_INSPECTION';atomic_json(path,state);return
        atomic_json(path,state);time.sleep(20)
    assert all(not alive(j['pid']) for j in current['source_workers']+current['fits'])
    # GPU1 belonged to the verified-terminal v66 source workers. Avoid GPU4,
    # which may still be running the separate Clean calibration audit.
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for budget in [256,1024,4096,8192]:
        log=RUN/'logs'/f'balanced_seven_v66_raw_audit_{budget}.log'
        with log.open('w') as h:
            child=subprocess.Popen([PYTHON,'-u',str(RUN/'scripts/audit_balanced_seven_v66.py'),
                                    '--budget',str(budget),'--fresh'],env=env,stdout=h,stderr=subprocess.STDOUT)
        job=dict(budget=budget,pid=child.pid,gpu=1,log=str(log));state['audits'].append(job)
        state['status']='RAW_AUDITS_RUNNING';atomic_json(path,state)
        while child.poll() is None:time.sleep(10)
        job['exit_code']=child.returncode;atomic_json(path,state)
        if child.returncode:
            state['status']='RAW_AUDIT_FAILED_REQUIRES_INSPECTION';atomic_json(path,state);return
    # Recompute the combined candidate from its current complete files, keeping
    # failed/missing Clean checks visible. This is evidence, not goal completion.
    log=RUN/'logs/balanced_seven_v66_combined_full48_report.log'
    with log.open('w') as h:
        result=subprocess.run([PYTHON,str(RUN/'scripts/report_recovery_goal_stage.py'),
                               '--stage','tabace_balanced_clean_calibration_v71'],stdout=h,stderr=subprocess.STDOUT)
    state.update(status='ALL_SEVEN_RAW_AUDITS_COMPLETE_REQUIRES_GOAL_REVIEW',report_exit_code=result.returncode,
                 goal_complete_claim=False,updated_unix=time.time())
    atomic_json(path,state)


if __name__=='__main__':run()
