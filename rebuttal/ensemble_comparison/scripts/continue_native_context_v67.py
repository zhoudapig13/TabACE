"""Use free GPU4 for native context inference, then source-only selection."""
import json
import os
import time
from pathlib import Path
import native_context_specific_v67 as experiment
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    path=experiment.OUT/'execution.json';assert not path.exists()
    occupied=json.loads((RUN/'results/tabace_augmented_classifier_v54/execution.json').read_text())
    for row in occupied['fits']:
        if row['gpu']!=4:continue
        p=Path('/proc')/str(row['pid'])/'stat'
        assert not p.exists() or p.read_text().split(') ',1)[1].split()[0]=='Z'
    log=RUN/'logs/native_context_v67_gpu4.log';p=launch('native_context_specific_v67.py',['banks'],log,4)
    state=dict(status='NATIVE_BANKS_RUNNING',pid=os.getpid(),native=dict(pid=p.pid,gpu=4,log=str(log)),fits=[])
    while p.poll() is None:state['updated_unix']=time.time();save(path,state);time.sleep(15)
    state['native']['exit_code']=p.returncode
    if p.returncode:state['status']='NATIVE_BANK_FAILURE_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
    assert len(list((experiment.OUT/'banks').glob('*/*/*.json')))==432
    children=[];state['status']='SOURCE_SELECTION_RUNNING'
    for index,b in enumerate(experiment.BUDGETS):
        log=RUN/'logs'/f'native_context_v67_fit_{b}.log';q=launch('native_context_specific_v67.py',['fit','--budget',str(b)],log,index)
        row=dict(budget=b,pid=q.pid,log=str(log));children.append((q,row));state['fits'].append(row)
    while True:
        for q,row in children:
            if q.poll() is not None:row['exit_code']=q.returncode
        state['updated_unix']=time.time();save(path,state)
        if any(row.get('exit_code',0) for _,row in children):state['status']='FIT_FAILED_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if all(q.poll() is not None for q,_ in children):state['status']='ALL_GROUPS_COMPLETE_REQUIRES_REPLAY_AND_FULL48_AUDIT';save(path,state);break
        time.sleep(15)
