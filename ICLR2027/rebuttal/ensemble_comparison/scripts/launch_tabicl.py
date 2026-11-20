"""Run one resumable worker per GPU and persist their actual exit status."""
import json
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parents[1]
jobs=[]
for gpu in range(8):
    log=(RUN/'logs'/f'tabicl_gpu{gpu}.log').open('a',buffering=1)
    command=[sys.executable,'-u',str(RUN/'scripts/run_tabicl.py'),
             '--gpu',str(gpu),'--shard',str(gpu),'--shards','8']
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    jobs.append((gpu,process,log,command))
    print('STARTED',gpu,process.pid,flush=True)
(RUN/'results/tabicl_workers.json').write_text(json.dumps([
    {'gpu':g,'pid':p.pid,'command':c,'status':'RUNNING'} for g,p,l,c in jobs],indent=2))
while any(p.poll() is None for _,p,_,_ in jobs):
    time.sleep(5)
    status=[{'gpu':g,'pid':p.pid,'exit_code':p.poll(),
             'status':'RUNNING' if p.poll() is None else ('COMPLETE' if p.returncode==0 else 'FAILED')}
            for g,p,l,c in jobs]
    temp=RUN/'results/tabicl_workers.tmp.json'
    temp.write_text(json.dumps(status,indent=2))
    temp.replace(RUN/'results/tabicl_workers.json')
codes=[]
for gpu,process,log,command in jobs:
    codes.append(process.returncode)
    log.close()
    print('FINISHED',gpu,'exit',process.returncode,flush=True)
raise SystemExit(0 if all(c==0 for c in codes) else 1)
