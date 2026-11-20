import argparse
import json
import os
from pathlib import Path
import subprocess
import time

RUN=Path(__file__).resolve().parents[1]
PYTHON=RUN.parents[1].parent/'PFN-Research/.tabpfn3-venv/bin/python'
parser=argparse.ArgumentParser();parser.add_argument('--backbone',choices=['tabicl','tabpfn'],default='tabicl');args=parser.parse_args()
if args.backbone=='tabicl':PYTHON=RUN/'.venv/bin/python'
prefix='specific_'+args.backbone
jobs=[]
for gpu in range(8):
    stream=(RUN/'logs'/f'{prefix}_gpu{gpu}.log').open('a',buffering=1)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    command=[str(PYTHON),'-u',str(RUN/'scripts/specific_reference.py'),
             '--shard',str(gpu),'--shards','8','--backbone',args.backbone]
    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
    jobs.append((gpu,process,stream));print('STARTED',gpu,process.pid,flush=True)
while any(p.poll() is None for _,p,_ in jobs):
    status=[{'gpu':g,'pid':p.pid,'exit_code':p.poll(),
             'status':'RUNNING' if p.poll() is None else ('COMPLETE' if p.returncode==0 else 'FAILED')}
            for g,p,s in jobs]
    temp=RUN/'results'/f'{prefix}_workers.tmp.json';temp.write_text(json.dumps(status,indent=2))
    temp.replace(RUN/'results'/f'{prefix}_workers.json')
    time.sleep(5)
codes=[]
for gpu,process,stream in jobs:
    codes.append(process.returncode);stream.close();print('FINISHED',gpu,process.returncode,flush=True)
(RUN/'results'/f'{prefix}_workers.json').write_text(json.dumps([{'gpu':g,'pid':p.pid,'exit_code':p.returncode,'status':'COMPLETE' if p.returncode==0 else 'FAILED'} for g,p,s in jobs],indent=2))
raise SystemExit(0 if all(x==0 for x in codes) else 1)
