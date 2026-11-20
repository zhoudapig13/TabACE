"""Bounded CPU orchestration for all eight Specific regression refinements."""
import os
import time
import bagged_specific_v56 as experiment
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    path=experiment.OUT/'execution.json';assert not path.exists()
    pending=[(b,k) for k in ['clean','seven'] for b in [256,1024,4096,8192]]
    state=dict(status='RUNNING',pid=os.getpid(),fits=[]);children=[]
    while pending or any(p.poll() is None for p,_ in children):
        for p,row in children:
            if p.poll() is not None:row['exit_code']=p.returncode
        if any(r.get('exit_code',0) for _,r in children):
            state['status']='FAILED_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        busy={r['slot'] for p,r in children if p.poll() is None}
        for slot in range(8):
            if slot in busy or not pending:continue
            budget,block=pending.pop(0);log=RUN/'logs'/f'bagged_specific_v56_{budget}_{block}.log'
            p=launch('bagged_specific_v56.py',['fit','--budget',str(budget),'--block',block],log,slot)
            row=dict(slot=slot,budget=budget,block=block,pid=p.pid,log=str(log));children.append((p,row));state['fits'].append(row)
        state['updated_unix']=time.time();save(path,state);time.sleep(10)
    for p,row in children:row['exit_code']=p.returncode
    state['status']='ALL_GROUPS_COMPLETE_REQUIRES_FULL48_AUDIT' if all(p.returncode==0 for p,_ in children) else 'FAILED_REQUIRES_REPAIR'
    state['updated_unix']=time.time();save(path,state)
    if any(p.returncode for p,_ in children):raise SystemExit(1)
