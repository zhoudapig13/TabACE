"""Resume-independent local queue for the new residual-regularization runs."""
import json
import os
from pathlib import Path
import subprocess
import time
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabace_moderate_adaptation_v75'


def main():
    assert not (OUT/'execution.json').exists()
    jobs=[dict(task='classification',block='clean',budget=b,args=['clean','--budget',str(b)]) for b in [1024,4096,8192]]
    jobs += [dict(task='regression',block=k,budget=b,args=['regression','--budget',str(b),'--block',k])
             for k in ['five','seven'] for b in [256,1024,4096,8192]]
    state=dict(status='RUNNING_UPDATED_USER_CONSTRAINTS',pid=os.getpid(),jobs=jobs,seed=0)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    active=[];pending=list(jobs)
    while pending or active:
        while pending and len(active)<6:
            job=pending.pop(0);log=RUN/'logs'/f'moderate_v75_{job["task"]}_{job["block"]}_{job["budget"]}.log'
            with log.open('w') as h:
                child=subprocess.Popen([str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts/moderate_adaptation_v75.py')]+job['args'],
                    env=env,stdout=h,stderr=subprocess.STDOUT)
            job.update(pid=child.pid,log=str(log),status='RUNNING');active.append((child,job))
        remaining=[]
        for child,job in active:
            if child.poll() is None:remaining.append((child,job))
            else:job.update(exit_code=child.returncode,status='COMPLETE' if child.returncode==0 else 'FAILED_REQUIRES_INSPECTION')
        active=remaining;state['updated_unix']=time.time();atomic_json(OUT/'execution.json',state)
        if active:time.sleep(5)
    state['status']='SOME_GROUPS_FAILED_REQUIRES_INSPECTION' if any(j['exit_code'] for j in jobs) else 'CLASSIFICATION_CLEAN_AND_REGRESSION_COMPLETE_REQUIRES_AUDIT'
    state['classification_seven']='Separate source-dependent worker; all35 datasets and all28 conditions required'
    atomic_json(OUT/'execution.json',state)


if __name__=='__main__':main()
