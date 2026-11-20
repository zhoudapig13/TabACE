"""Resumable development jobs with persisted worker status; does not select results."""
import argparse, fcntl, json, os, subprocess, time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['feature','tuning','holdout','full_holdout'],required=True)
p.add_argument('--workers',type=int,default=8)
p.add_argument('--backbone',choices=['tabicl','tabpfn'],default='tabicl');a=p.parse_args()
prefix=f'benchmark_development_{a.stage}_{a.backbone}'
lock=(RUN/'results'/f'{prefix}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
commands=[]
if a.stage in ['feature','holdout','full_holdout']:
    python=RUN/'.venv/bin/python' if a.backbone=='tabicl' else RUN.parents[1].parent/'PFN-Research/.tabpfn3-venv/bin/python'
    script={'feature':'run_feature_calibrated.py','holdout':'evaluate_benchmark_tuning_holdout.py','full_holdout':'evaluate_specific_full_holdout.py'}[a.stage]
    for worker in range(a.workers):commands.append((f'worker{worker}',[str(python),'-u',str(RUN/'scripts'/script),'--backbone',a.backbone,'--shard',str(worker),'--shards',str(a.workers)],worker%8))
else:
    for track in ['five_feature_calibrated','five','seven_reimplemented']:
        for variant in ['shared','specific']:
            commands.append((f'{track}_{variant}',[str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts/fit_benchmark_reference.py'),
                '--track',track,'--backbone',a.backbone,'--variant',variant,'--wait'],None))
jobs=[]
for label,command,gpu in commands:
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    stream=(RUN/'logs'/f'{prefix}_{label}.log').open('a',buffering=1)
    proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env);jobs.append((label,proc,stream))
while True:
    states=[{'label':label,'pid':proc.pid,'exit_code':proc.poll(),'status':'RUNNING' if proc.returncode is None else 'COMPLETE' if proc.returncode==0 else 'FAILED'} for label,proc,s in jobs]
    path=RUN/'results'/f'{prefix}_workers.json';tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(states,indent=2));tmp.replace(path)
    if all(proc.returncode is not None for _,proc,_ in jobs):break
    time.sleep(5)
for label,proc,s in jobs:s.close();print('FINISHED',label,proc.returncode,flush=True)
if any(proc.returncode for _,proc,_ in jobs):raise SystemExit(1)
