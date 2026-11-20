"""Infer balanced TRAIN calibration on free GPUs, then source-select adapters."""
import json
import os
import time
from pathlib import Path
import balanced_seven_v66 as experiment
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    path=experiment.OUT/'execution.json';assert not path.exists()
    assert json.loads((experiment.OUT/'source_contract_audit.json').read_text())['status']=='PASS'
    state=dict(status='SOURCE_INFERENCE_RUNNING',pid=os.getpid(),source_workers=[],fits=[]);children=[]
    busy=json.loads((experiment.expert.OUT/'execution.json').read_text())
    for gpu in [1,2,3]:
        for row in busy['fits']:
            if row['gpu']!=gpu:continue
            p=Path('/proc')/str(row['pid'])/'stat'
            assert not p.exists() or p.read_text().split(') ',1)[1].split()[0]=='Z','GPU still owned by live augmentation job'
    for index,gpu in enumerate([1,2,3]):
        log=RUN/'logs'/f'balanced_seven_v66_source_gpu{gpu}.log'
        p=launch('balanced_seven_v66.py',['source','--index',str(index),'--count','3'],log,gpu)
        row=dict(index=index,gpu=gpu,pid=p.pid,log=str(log));children.append((p,row));state['source_workers'].append(row)
    while True:
        for p,row in children:
            if p.poll() is not None:row['exit_code']=p.returncode
        state['updated_unix']=time.time();save(path,state)
        if any(row.get('exit_code',0) for _,row in children):state['status']='SOURCE_FAILED_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if all(p.poll() is not None for p,_ in children):break
        time.sleep(15)
    expected=len(list((experiment.OUT/'source').glob('*/*/*.json')));assert expected==3920,expected
    state['status']='SOURCE_ONLY_SELECTION_RUNNING';children=[]
    for index,b in enumerate([256,1024,4096,8192]):
        log=RUN/'logs'/f'balanced_seven_v66_fit_{b}.log'
        p=launch('balanced_seven_v66.py',['fit','--budget',str(b)],log,index)
        row=dict(budget=b,pid=p.pid,log=str(log));children.append((p,row));state['fits'].append(row)
    while True:
        for p,row in children:
            if p.poll() is not None:row['exit_code']=p.returncode
        state['updated_unix']=time.time();save(path,state)
        if any(row.get('exit_code',0) for _,row in children):state['status']='FIT_FAILED_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if all(p.poll() is not None for p,_ in children):state['status']='ALL_GROUPS_COMPLETE_REQUIRES_REPLAY_AND_FULL48_AUDIT';save(path,state);break
        time.sleep(15)
