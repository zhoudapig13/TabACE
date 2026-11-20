"""Run a corrected-classification stage on all eight GPUs, preserving exits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
RUN=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['baseline','reference','specific','features'],required=True)
parser.add_argument('--backbone',choices=['tabicl','tabpfn','tabm'],default='tabicl');args=parser.parse_args()
python=RUN.parents[1].parent/'PFN-Research/.tabpfn3-venv/bin/python' if args.backbone=='tabpfn' else RUN/'.venv/bin/python'
scripts={'baseline':'run_corrected_baselines.py','reference':'shared_reference.py','specific':'specific_reference.py','features':'run_feature_views.py'}
prefix=f'corrected_{args.stage}_{args.backbone}';jobs=[]
stage_lock=(RUN/'results'/f'{prefix}.lock').open('a')
fcntl.flock(stage_lock,fcntl.LOCK_EX)
for gpu in range(8):
    stream=(RUN/'logs'/f'{prefix}_gpu{gpu}.log').open('a',buffering=1)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[str(python),'-u',str(RUN/'scripts'/scripts[args.stage]),'--shard',str(gpu),'--shards','8']
    if args.stage!='features':command+=['--backbone',args.backbone]
    if args.stage!='baseline':command+=['--track','five_corrected']
    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
    jobs.append((gpu,process,stream));print('STARTED',gpu,process.pid,flush=True)
while True:
    status=[{'gpu':g,'pid':p.pid,'exit_code':p.poll(),
        'status':'RUNNING' if p.returncode is None else 'COMPLETE' if p.returncode==0 else 'FAILED'} for g,p,s in jobs]
    temporary=RUN/'results'/f'{prefix}_workers.tmp.json';temporary.write_text(json.dumps(status,indent=2));temporary.replace(RUN/'results'/f'{prefix}_workers.json')
    if all(p.returncode is not None for g,p,s in jobs):break
    time.sleep(5)
for g,p,s in jobs:s.close();print('FINISHED',g,p.returncode,flush=True)
raise SystemExit(0 if all(p.returncode==0 for g,p,s in jobs) else 1)
