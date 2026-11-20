"""Fit source-calibrated policies when complete immutable v29 banks are ready."""
import json
import os
from pathlib import Path
import time
from continue_tabace_v28 import launch,save,RUN

ROOT=RUN/'results/tabace_reliability_fusion_v33'
BANKS=RUN/'results/tabace_context_adaptation_v29/banks'
EXPECTED={('classification','clean'):72,('classification','five'):1440,
          ('classification','seven'):980,('regression','clean'):35,
          ('regression','five'):700,('regression','seven'):924}


if __name__=='__main__':
    assert json.loads((ROOT/'backend_consistency.json').read_text())['status']=='PASS'
    file=ROOT/'execution.json';assert not file.exists(),'Avoid duplicate policies'
    state=dict(status='WAITING_FOR_BANKS',pid=os.getpid(),fits=[]);children={}
    while True:
        for proc,row in children.values():
            if proc.poll() is not None:row['exit_code']=proc.returncode
        for budget in [256,1024,4096,8192]:
            for (task,block),count in EXPECTED.items():
                key=budget,task,block
                if key in children or sum(p.poll() is None for p,_ in children.values())>=6:continue
                if any(len([p for p in (BANKS/role/str(budget)/task/block).glob('*/*.json')
                            if '.tmp.' not in p.name and p.with_suffix('.npz').exists()])!=count
                       for role in ['validation','query']):continue
                log=RUN/'logs'/f'reliability_v33_{budget}_{task}_{block}.log'
                proc=launch('reliability_fusion_v33.py',['fit','--budget',str(budget),'--task',task,'--block',block],log)
                row=dict(budget=budget,task=task,block=block,pid=proc.pid,log=str(log));children[key]=proc,row;state['fits'].append(row)
        state['status']='RUNNING' if any(p.poll() is None for p,_ in children.values()) else 'WAITING_FOR_BANKS'
        state['updated_unix']=time.time();save(file,state)
        if len(children)==24 and all(p.poll() is not None for p,_ in children.values()):
            state['status']='POLICIES_FINISHED_REQUIRES_FULL_GOAL_AUDIT';save(file,state);break
        time.sleep(20)
