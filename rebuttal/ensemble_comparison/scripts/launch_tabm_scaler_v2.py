"""Run all TabM conditions with the standard constant-feature handling fix."""
import json,os,subprocess,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
jobs=[]
for gpu in range(8):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    stream=(RUN/'logs'/f'tabm_scaler_v2_gpu{gpu}.log').open('a',buffering=1)
    command=[str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts/run_tabm.py'),'--track','all','--shard',str(gpu),'--shards','8']
    proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env);jobs.append((gpu,proc,stream))
while True:
    status=[{'gpu':g,'pid':p.pid,'exit_code':p.poll(),'status':'RUNNING' if p.returncode is None else 'COMPLETE' if p.returncode==0 else 'FAILED'} for g,p,s in jobs]
    tmp=RUN/'results/tabm_scaler_v2_workers.tmp.json';tmp.write_text(json.dumps(status,indent=2));tmp.replace(RUN/'results/tabm_scaler_v2_workers.json')
    if all(p.returncode is not None for g,p,s in jobs):break
    time.sleep(5)
for g,p,s in jobs:s.close();print('FINISHED',g,p.returncode,flush=True)
if any(p.returncode for g,p,s in jobs):raise SystemExit(1)
