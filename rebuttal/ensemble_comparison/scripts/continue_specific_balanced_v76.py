"""Verify live training handles, then run all four source-only selectors."""
import json
import os
from pathlib import Path
import subprocess
import time
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabace_specific_balanced_bags_v76'


def live(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'


def run():
    training=json.loads((OUT/'execution.json').read_text())
    state=dict(status='WAITING_ON_VERIFIED_BAG_WORKERS',pid=os.getpid(),fits=[])
    path=OUT/'selection_execution.json';assert not path.exists()
    while True:
        statuses=[]
        for job in training['jobs']:
            proof=OUT/'workers'/f'{job["index"]}.json'
            ok=proof.exists() and json.loads(proof.read_text())['status']=='COMPLETE'
            active=live(job['pid']);statuses.append(dict(index=job['index'],pid=job['pid'],live=active,completed=ok))
            if not active and not ok:
                state.update(status='TERMINAL_TRAINING_WORKER_REQUIRES_INSPECTION',workers=statuses);atomic_json(path,state);return
        state.update(workers=statuses,source_bag_cases=len(list((OUT/'source').glob('*/*.json'))),updated_unix=time.time());atomic_json(path,state)
        if all(s['completed'] and not s['live'] for s in statuses):break
        time.sleep(10)
    assert len(list((OUT/'source').glob('*/*.json')))==980
    assert len(list((OUT/'experts').glob('*/*/models_complete.json')))==245
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    active=[]
    for budget in [256,1024,4096,8192]:
        log=RUN/'logs'/f'specific_family_selection_v76_{budget}.log'
        with log.open('w') as h:
            child=subprocess.Popen([str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts/specific_family_selection_v76.py'),'--budget',str(budget)],env=env,stdout=h,stderr=subprocess.STDOUT)
        job=dict(budget=budget,pid=child.pid,log=str(log));active.append((child,job));state['fits'].append(job)
    state['status']='DATASET_CROSSFIT_AND_SOURCE_UNCERTAINTY_SELECTION_RUNNING';atomic_json(path,state)
    while any(p.poll() is None for p,_ in active):time.sleep(10)
    for p,job in active:job['exit_code']=p.returncode
    state['status']='SELECTION_FAILED_REQUIRES_INSPECTION' if any(p.returncode for p,_ in active) else 'ALL_NEW_SPECIFIC_RESULTS_COMPLETE_REQUIRES_AUDIT'
    state['updated_unix']=time.time();atomic_json(path,state)


if __name__=='__main__':run()
