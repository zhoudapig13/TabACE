"""Eight GPU workers for the audited reconstructed seven-shift benchmark."""
import json
import os
from pathlib import Path
import subprocess
import time

RUN=Path(__file__).resolve().parents[1]
PYTHON=RUN/'.venv/bin/python'
jobs=[]
for gpu in range(8):
    stream=(RUN/'logs'/f'tabm_gpu{gpu}.log').open('a',buffering=1)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    command=[str(PYTHON),'-u',str(RUN/'scripts/run_tabm.py'),
             '--shard',str(gpu),'--shards','8']
    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
    jobs.append((gpu,process,stream));print('STARTED',gpu,process.pid,flush=True)
while any(p.poll() is None for _,p,_ in jobs):
    status=[{'gpu':g,'pid':p.pid,'exit_code':p.poll(),
             'status':'RUNNING' if p.poll() is None else ('COMPLETE' if p.returncode==0 else 'FAILED')}
            for g,p,s in jobs]
    temp=RUN/'results/tabm_workers.tmp.json';temp.write_text(json.dumps(status,indent=2))
    temp.replace(RUN/'results/tabm_workers.json')
    time.sleep(5)
codes=[]
for gpu,process,stream in jobs:
    codes.append(process.returncode);stream.close();print('FINISHED',gpu,process.returncode,flush=True)
(RUN/'results/tabm_workers.json').write_text(json.dumps([{'gpu':g,'pid':p.pid,'exit_code':p.returncode,'status':'COMPLETE' if p.returncode==0 else 'FAILED'} for g,p,s in jobs],indent=2))
raise SystemExit(0 if all(x==0 for x in codes) else 1)
