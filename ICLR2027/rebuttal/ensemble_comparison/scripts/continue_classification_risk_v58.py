"""Fit all eight classifier-bank reliability policies on CPU."""
import json
import os
import time
import classification_risk_v58 as experiment
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    assert json.loads((experiment.OUT/'source_contract_audit.json').read_text())['status']=='PASS'
    path=experiment.OUT/'execution.json';assert not path.exists()
    jobs=[(b,k) for k in ['clean','seven'] for b in [256,1024,4096,8192]]
    state=dict(status='RUNNING',pid=os.getpid(),fits=[]);children=[]
    for slot,(budget,block) in enumerate(jobs):
        log=RUN/'logs'/f'classification_risk_v58_{budget}_{block}.log'
        p=launch('classification_risk_v58.py',['fit','--budget',str(budget),'--block',block],log,slot)
        row=dict(budget=budget,block=block,pid=p.pid,log=str(log));children.append((p,row));state['fits'].append(row)
    while True:
        for p,row in children:
            if p.poll() is not None:row['exit_code']=p.returncode
        state['updated_unix']=time.time();save(path,state)
        if any(row.get('exit_code',0) for _,row in children):state['status']='FAILED_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if all(p.poll() is not None for p,_ in children):state['status']='ALL_GROUPS_COMPLETE_REQUIRES_FULL48_AUDIT';save(path,state);break
        time.sleep(15)
