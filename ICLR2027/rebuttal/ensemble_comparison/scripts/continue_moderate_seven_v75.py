"""Start each budget once all its35x28 source cases are actually complete."""
import json
import os
from pathlib import Path
import subprocess
import time
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabace_moderate_adaptation_v75'
SOURCE=RUN/'results/tabace_balanced_seven_v66'


def live(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z'


def main():
    state=dict(status='WAITING_ON_VERIFIED_SOURCE_PROCESSES',pid=os.getpid(),jobs=[],source_counts={})
    state_path=OUT/'classification_seven_execution.json';assert not state_path.exists()
    pending=[256,1024,4096,8192];active=[]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    while pending or active:
        original=json.loads((SOURCE/'execution.json').read_text())
        handles=original.get('source_workers',[])
        state['source_handles_verified_live']={str(j['pid']):live(j['pid']) for j in handles}
        for budget in pending[:]:
            paths=[p for p in (SOURCE/'source'/str(budget)).glob('*/*.npz') if p.with_suffix('.json').exists()]
            state['source_counts'][str(budget)]=len(paths)
            if len(paths)!=980:continue
            # Full cohort/condition membership is asserted by the fit worker
            # before policy selection; partial datasets are never substituted.
            log=RUN/'logs'/f'moderate_v75_classification_seven_{budget}.log'
            with log.open('w') as h:
                child=subprocess.Popen([str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts/moderate_seven_v75.py'),
                    '--budget',str(budget)],env=env,stdout=h,stderr=subprocess.STDOUT)
            row=dict(budget=budget,pid=child.pid,log=str(log),status='RUNNING')
            state['jobs'].append(row);active.append((child,row));pending.remove(budget)
        remaining=[]
        for child,row in active:
            if child.poll() is None:remaining.append((child,row))
            else:row.update(exit_code=child.returncode,status='COMPLETE' if child.returncode==0 else 'FAILED_REQUIRES_INSPECTION')
        active=remaining
        if pending and not any(state['source_handles_verified_live'].values()):
            state['status']='INCOMPLETE_SOURCE_AND_TERMINAL_HANDLES_REQUIRES_INSPECTION'
            state['pending_budgets']=pending;atomic_json(state_path,state);return
        state['status']='FITTING_READY_BUDGETS' if active else 'WAITING_ON_VERIFIED_SOURCE_PROCESSES'
        state['updated_unix']=time.time();atomic_json(state_path,state)
        if pending or active:time.sleep(10)
    state['status']='SOME_FITS_FAILED_REQUIRES_INSPECTION' if any(j['exit_code'] for j in state['jobs']) else 'ALL_CLASSIFICATION_SEVEN_COMPLETE_REQUIRES_AUDIT'
    state['updated_unix']=time.time();atomic_json(state_path,state)


if __name__=='__main__':main()
