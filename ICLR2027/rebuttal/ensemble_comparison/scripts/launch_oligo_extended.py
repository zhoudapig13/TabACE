"""Queue 400-candidate regression workers behind memory-heavy GPU jobs."""
import json
import os
from pathlib import Path
import subprocess
import time
RUN=Path(__file__).resolve().parents[1]


def live(pid):
    try:command=Path(f'/proc/{pid}/cmdline').read_text()
    except FileNotFoundError:return False
    return '/ensemble_comparison/scripts/' in command or 'ensemble_comparison/scripts/' in command


dependencies={g:[] for g in range(8)}
for name in ['reference_tabpfn_workers.json','feature_views_workers.json','tabpfn_workers.json']:
    p=RUN/'results'/name
    if p.exists():
        for j in json.loads(p.read_text()):
            if live(j['pid']):dependencies[int(j['gpu'])].append(j['pid'])
jobs={};logs={};last_status=[]
while True:
    for gpu in range(8):
        if gpu in jobs or any(live(pid) for pid in dependencies[gpu]):continue
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
        logs[gpu]=(RUN/'logs'/f'oligo_extended_gpu{gpu}.log').open('a',buffering=1)
        command=[str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts/run_oligo_extended.py'),'--candidates','400','--shard',str(gpu),'--shards','8']
        jobs[gpu]=subprocess.Popen(command,stdout=logs[gpu],stderr=subprocess.STDOUT,env=env)
        print('STARTED',gpu,jobs[gpu].pid,flush=True)
    status=[]
    for gpu in range(8):
        if gpu not in jobs:
            status.append({'gpu':gpu,'status':'WAITING_FOR_GPU_JOBS','dependencies':[p for p in dependencies[gpu] if live(p)]})
        else:
            process=jobs[gpu];code=process.poll()
            status.append({'gpu':gpu,'pid':process.pid,'exit_code':code,'status':'RUNNING' if code is None else 'COMPLETE' if code==0 else 'FAILED'})
    temp=RUN/'results/oligo_extended_workers.tmp.json';temp.write_text(json.dumps(status,indent=2));temp.replace(RUN/'results/oligo_extended_workers.json')
    if len(jobs)==8 and all(p.poll() is not None for p in jobs.values()):break
    time.sleep(10)
for f in logs.values():f.close()
raise SystemExit(0 if all(p.returncode==0 for p in jobs.values()) else 1)
