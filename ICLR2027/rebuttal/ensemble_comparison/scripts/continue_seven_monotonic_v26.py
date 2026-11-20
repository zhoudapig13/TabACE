"""GPU queue for a single frozen Base probe; wait for raw validation first."""
import json
import os
import time
from continue_tabace_v28 import alive,launch,save,RUN

RAW=RUN/'results/tabace_context_reproduction_v28'
OUT=RUN/'results/tabicl_validation_recovery_v26/frozen_seven_monotonic_query'


def main():
    path=OUT/'execution.json';assert not path.exists(),'Do not duplicate probe workers'
    state=dict(status='QUEUED',supervisor_pid=os.getpid(),workers=[]);active={}
    while True:
        raw=json.loads((RAW/'higher_supervisor.json').read_text())
        for gpu in range(8):
            if gpu in active:
                process,record=active[gpu]
                if process.poll() is not None:record['exit_code']=process.returncode
                continue
            predecessor=[r for r in raw['validation'] if r['gpu']==gpu]
            if not predecessor or any(alive(r['pid']) for r in predecessor):continue
            log=RUN/'logs'/f'seven_monotonic_v26_query_gpu{gpu}.log'
            process=launch('query_seven_monotonic_v26.py',['worker','--shard',str(gpu),'--shards','8'],log,gpu)
            record=dict(gpu=gpu,pid=process.pid,log=str(log));active[gpu]=process,record;state['workers'].append(record)
        state['status']='RUNNING' if active else 'QUEUED'
        state['updated_unix']=time.time();save(path,state)
        if len(active)==8 and all(p.poll() is not None for p,_ in active.values()):
            log=RUN/'logs'/'seven_monotonic_v26_query_report.log'
            process=launch('query_seven_monotonic_v26.py',['report'],log)
            state['report_exit_code']=process.wait();state['status']='PROBE_FINISHED_REQUIRES_RESULT_REVIEW'
            save(path,state);return
        time.sleep(10)


if __name__=='__main__':main()
