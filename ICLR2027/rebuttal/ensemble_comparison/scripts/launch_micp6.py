import fcntl,json,os,subprocess,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1];lock=(RUN/'results/final_micp6.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
python=RUN.parents[1].parent/'PFN-Research/.tabpfn3-venv/bin/python';jobs=[]
for worker in range(16):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(worker%8),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    stream=(RUN/'logs'/f'final_micp6_worker{worker}.log').open('a',buffering=1)
    proc=subprocess.Popen([str(python),'-u',str(RUN/'scripts/run_micp6.py'),'--shard',str(worker),'--shards','16'],stdout=stream,stderr=subprocess.STDOUT,env=env);jobs.append((worker,proc,stream))
while True:
    states=[{'worker':w,'pid':p.pid,'exit_code':p.poll(),'status':'RUNNING' if p.returncode is None else 'COMPLETE' if p.returncode==0 else 'FAILED'} for w,p,s in jobs]
    path=RUN/'results/final_micp6_workers.json';tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(states,indent=2));tmp.replace(path)
    if all(p.returncode is not None for w,p,s in jobs):break
    time.sleep(5)
for w,p,s in jobs:s.close();print('FINISHED',w,p.returncode,flush=True)
if any(p.returncode for w,p,s in jobs):raise SystemExit(1)
