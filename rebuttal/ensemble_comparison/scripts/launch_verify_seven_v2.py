import fcntl,json,os,subprocess,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
lock=(RUN/'results/seven_v2_verification.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
jobs=[]
for gpu in range(8):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    stream=(RUN/'logs'/f'seven_v2_verify_gpu{gpu}.log').open('a',buffering=1)
    proc=subprocess.Popen([str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts/verify_seven_regression_v2.py'),
        '--shard',str(gpu),'--shards','8'],stdout=stream,stderr=subprocess.STDOUT,env=env);jobs.append((gpu,proc,stream))
while True:
    states=[{'gpu':g,'pid':p.pid,'exit_code':p.poll(),'status':'RUNNING' if p.returncode is None else 'COMPLETE' if p.returncode==0 else 'FAILED'} for g,p,s in jobs]
    path=RUN/'results/seven_v2_verification_workers.json';tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(states,indent=2));tmp.replace(path)
    if all(p.returncode is not None for g,p,s in jobs):break
    time.sleep(5)
for g,p,s in jobs:s.close();print('FINISHED',g,p.returncode,flush=True)
if any(p.returncode for g,p,s in jobs):raise SystemExit(1)
