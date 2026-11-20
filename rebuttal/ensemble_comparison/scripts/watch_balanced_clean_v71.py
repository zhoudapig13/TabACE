"""Await the specific live v71 controller, then audit and report its full scope."""
import json
import os
from pathlib import Path
import subprocess
import time
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabace_balanced_clean_calibration_v71'
PYTHON=str(RUN/'.venv/bin/python')


def alive(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z'


def run():
    initial=json.loads((OUT/'execution.json').read_text())
    state=dict(status='WAITING_ON_VERIFIED_V71_CONTROLLER',pid=os.getpid(),controller_pid=initial['pid'])
    path=OUT/'audit_execution.json'
    while True:
        current=json.loads((OUT/'execution.json').read_text())
        if current['status']=='ALL_GROUPS_COMPLETE_REQUIRES_RAW_AND_FULL48_AUDIT':break
        state.update(updated_unix=time.time(),controller_verified_live=alive(current['pid']),generation_status=current['status'])
        if not state['controller_verified_live'] or 'FAILED' in current['status']:
            state['status']='CONTROLLER_TERMINAL_REQUIRES_INSPECTION';atomic_json(path,state);return
        atomic_json(path,state);time.sleep(15)
    # Native1 replay needs the just-freed GPU. Confirm every v71 source child
    # and fit child is terminal; never infer that from a timeout.
    assert all(not alive(j['pid']) for j in current['source_workers']+current['fits'])
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log=RUN/'logs/balanced_clean_v71_raw_audit.log'
    with log.open('w') as h:
        child=subprocess.Popen([PYTHON,'-u',str(RUN/'scripts/audit_balanced_clean_v71.py'),'--fresh'],env=env,stdout=h,stderr=subprocess.STDOUT)
    state.update(status='RAW_AUDIT_RUNNING',audit_pid=child.pid,log=str(log));atomic_json(path,state)
    while child.poll() is None:time.sleep(10)
    state.update(audit_exit_code=child.returncode)
    if child.returncode:
        state['status']='RAW_AUDIT_FAILED_REQUIRES_INSPECTION';atomic_json(path,state);return
    log=RUN/'logs/balanced_clean_v71_full48_report.log'
    with log.open('w') as h:
        result=subprocess.run([PYTHON,str(RUN/'scripts/report_recovery_goal_stage.py'),'--stage',OUT.name],stdout=h,stderr=subprocess.STDOUT)
    state.update(report_exit_code=result.returncode,status='AUDITED_CANDIDATE_REQUIRES_GOAL_REVIEW')
    if not result.returncode:
        audit=json.loads((OUT/'goal_audit.json').read_text())
        state.update(required_metric_checks=48,completed_metric_checks=audit['completed_metric_checks'],passed_metric_checks=audit['passed_metric_checks'],
                     missing=audit['missing'],failed_checks=[c for c in audit['checks'] if not c['pass_all']])
    state['goal_complete_claim']=False;state['updated_unix']=time.time();atomic_json(path,state)


if __name__=='__main__':run()
