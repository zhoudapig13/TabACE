"""Execute and audit the predeclared observed-source tail diagnostic."""
import json
import os
from pathlib import Path
import time
from continue_tabace_v28 import launch,save,RUN

ROOT=RUN/'results/tabicl_tail_recovery_v32'


if __name__=='__main__':
    assert json.loads((ROOT/'constructor_audit.json').read_text())['status']=='PASS'
    path=ROOT/'execution.json';assert not path.exists(),'Avoid duplicate validation workers'
    state=dict(status='SOURCE_VALIDATION_RUNNING',pid=os.getpid(),workers=[],query_predictions=0);children=[]
    for shard,gpu in enumerate([0,5]):
        log=RUN/'logs'/f'tail_recovery_v32_validation_gpu{gpu}.log'
        process=launch('tail_recovery_v32.py',['worker','--shard',str(shard),'--shards','2'],log,gpu)
        record=dict(shard=shard,gpu=gpu,pid=process.pid,log=str(log));children.append((process,record));state['workers'].append(record)
    while True:
        for process,record in children:
            if process.poll() is not None:record['exit_code']=process.returncode
        save(path,state)
        if all(p.poll() is not None for p,_ in children):break
        time.sleep(10)
    if any(p.returncode for p,_ in children):
        state['status']='VALIDATION_FAILED_REQUIRES_REPAIR';save(path,state)
    else:
        state['status']='AUDITING';save(path,state)
        for phase in ['audit','select']:
            process=launch('tail_recovery_v32.py',[phase],RUN/'logs'/f'tail_recovery_v32_{phase}.log')
            state[phase+'_exit_code']=process.wait();save(path,state)
            if process.returncode:break
        state['status']='SOURCE_DIAGNOSTIC_FINISHED_REQUIRES_REVIEW';save(path,state)
