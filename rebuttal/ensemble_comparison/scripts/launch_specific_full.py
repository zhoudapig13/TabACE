"""Run all registered Specific contexts without a candidate or branch cap."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True)
p.add_argument('--track',choices=['five','five_feature_calibrated'],required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
prefix=f'specific_full_{a.backbone}_{a.track}';lock=(RUN/'results'/f'{prefix}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
python=RUN/'.venv/bin/python' if a.backbone=='tabicl' else RUN.parents[1].parent/'PFN-Research/.tabpfn3-venv/bin/python'
jobs=[]
for worker in range(a.workers):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=str(worker%8))
    stream=(RUN/'logs'/f'{prefix}_worker{worker}.log').open('a',buffering=1)
    proc=subprocess.Popen([str(python),'-u',str(RUN/'scripts/run_specific_full_pool.py'),'--backbone',a.backbone,
        '--track',a.track,'--shard',str(worker),'--shards',str(a.workers)],stdout=stream,stderr=subprocess.STDOUT,env=env)
    jobs.append((worker,proc,stream))
fitlog=(RUN/'logs'/f'{prefix}_fit.log').open('a',buffering=1)
fitenv=os.environ.copy();fitenv.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
fitter=subprocess.Popen([str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts/fit_benchmark_specific_uncapped.py'),
    '--backbone',a.backbone,'--track',a.track,'--pool','registered'],stdout=fitlog,stderr=subprocess.STDOUT,env=fitenv)
jobs.append(('fit',fitter,fitlog))
while True:
    states=[{'worker':w,'pid':proc.pid,'exit_code':proc.poll(),'status':'RUNNING' if proc.returncode is None else 'COMPLETE' if proc.returncode==0 else 'FAILED'} for w,proc,s in jobs]
    path=RUN/'results'/f'{prefix}_workers.json';tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(states,indent=2));tmp.replace(path)
    if all(proc.returncode is not None for _,proc,_ in jobs):break
    time.sleep(5)
for worker,proc,s in jobs:s.close();print('FINISHED',worker,proc.returncode,flush=True)
if any(proc.returncode for _,proc,_ in jobs):raise SystemExit(1)
