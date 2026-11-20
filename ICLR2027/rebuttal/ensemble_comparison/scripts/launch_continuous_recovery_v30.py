"""Two low-memory validation workers; report after both actually terminate."""
import json
import os
import time
from continue_tabace_v28 import launch,save,RUN

OUT=RUN/'results/tabicl_continuous_recovery_v30'


if __name__=='__main__':
    assert json.loads((OUT/'constructor_audit.json').read_text())['status']=='PASS'
    path=OUT/'execution.json';assert not path.exists(),'Avoid duplicate validation workers'
    state=dict(status='RUNNING',supervisor_pid=os.getpid(),workers=[],original_query_runs=0)
    children=[]
    for shard,gpu in enumerate([0,5]):
        log=RUN/'logs'/f'continuous_recovery_v30_validation_gpu{gpu}.log'
        process=launch('continuous_recovery_v30.py',['worker','--shard',str(shard),'--shards','2'],log,gpu)
        record=dict(shard=shard,gpu=gpu,pid=process.pid,log=str(log))
        children.append((process,record));state['workers'].append(record)
    while True:
        for process,record in children:
            if process.poll() is not None:record['exit_code']=process.returncode
        state['updated_unix']=time.time();save(path,state)
        if all(p.poll() is not None for p,_ in children):break
        time.sleep(10)
    process=launch('continuous_recovery_v30.py',['report'],RUN/'logs/continuous_recovery_v30_validation_report.log')
    state['report_exit_code']=process.wait();state['status']='SOURCE_VALIDATION_FINISHED_REQUIRES_AUDIT'
    save(path,state)
