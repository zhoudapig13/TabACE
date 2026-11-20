"""Finish low-memory regression probes beside low-dimensional v29 workers.

GPUs 0 and 5's concurrent v29 shards have <=116 input features, unlike the image
shards. At most one extra regression process per device, with a 6 GiB free-memory
guard. This changes scheduling only, never inputs, seeds, or frozen parameters.
"""
import json
import os
import subprocess
import time
from continue_tabace_v28 import launch,save,RUN

OUT=RUN/'results/tabicl_validation_recovery_v26/frozen_seven_monotonic_query'


def main():
    path=OUT/'execution.json';state=json.loads(path.read_text())
    assert all(r.get('exit_code')==0 for r in state['workers'])
    pending=[gpu for gpu in range(8) if gpu not in {r['gpu'] for r in state['workers']}]
    state['supervisor_pid']=os.getpid();state['dispatch_revision']='Low-memory remaining shards on physical devices0,5'
    active={}
    while pending or active:
        for device in list(active):
            process,record=active[device]
            if process.poll() is not None:
                record['exit_code']=process.returncode
                del active[device]
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(row.split(',')[0]):int(row.split(',')[1]) for row in output.strip().splitlines()}
        for device in [0,5]:
            if device in active or not pending or free[device]<6144:continue
            shard=pending.pop(0);log=RUN/'logs'/f'seven_monotonic_v26_query_gpu{shard}.log'
            process=launch('query_seven_monotonic_v26.py',['worker','--shard',str(shard),'--shards','8'],log,device)
            record=dict(gpu=shard,logical_shard=shard,physical_gpu=device,pid=process.pid,log=str(log))
            state['workers'].append(record);active[device]=process,record
        state['status']='RUNNING';state['pending_shards']=pending;state['updated_unix']=time.time();save(path,state)
        time.sleep(5)
    process=launch('query_seven_monotonic_v26.py',['report'],RUN/'logs/seven_monotonic_v26_query_report.log')
    state['report_exit_code']=process.wait();state['status']='PROBE_FINISHED_REQUIRES_RESULT_REVIEW';save(path,state)


if __name__=='__main__':main()
