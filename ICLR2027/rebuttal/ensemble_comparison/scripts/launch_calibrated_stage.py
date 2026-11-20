"""Eight GPU worker launcher with a per-stage lock and explicit completion."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['baseline','reference','specific','features','oligo'],required=True)
p.add_argument('--backbone',choices=['tabicl','tabpfn','tabm'],default='tabicl');p.add_argument('--track',choices=['five_calibrated','five_feature_calibrated','seven_calibrated','seven_calibrated_v2'],required=True)
p.add_argument('--workers',type=int,default=8);p.add_argument('--candidates',type=int,choices=[40,60,400],default=400);a=p.parse_args()
prefix=f'{a.track}_{a.stage}_{a.backbone}'
lock=(RUN/'results'/f'{prefix}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
python=RUN.parents[1].parent/'PFN-Research/.tabpfn3-venv/bin/python' if a.backbone=='tabpfn' else RUN/'.venv/bin/python'
script={'baseline':'run_calibrated_baselines.py','reference':'shared_reference.py','specific':'specific_reference.py',
        'features':'run_feature_views.py','oligo':'run_oligo_extended.py'}[a.stage]
jobs=[]
for gpu in range(a.workers):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu%8),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[str(python),'-u',str(RUN/'scripts'/script),'--track',a.track,'--shard',str(gpu),'--shards',str(a.workers)]
    if a.stage in ['baseline','reference','specific']:command+=['--backbone',a.backbone]
    if a.stage=='oligo':command+=['--candidates',str(a.candidates),'--condition-sharding']
    stream=(RUN/'logs'/f'{prefix}_gpu{gpu}.log').open('a',buffering=1)
    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env);jobs.append((gpu,process,stream))
while True:
    states=[{'gpu':g,'pid':p.pid,'exit_code':p.poll(),'status':'RUNNING' if p.returncode is None else 'COMPLETE' if p.returncode==0 else 'FAILED'} for g,p,s in jobs]
    file=RUN/'results'/f'{prefix}_workers.json';temp=file.with_suffix('.tmp.json');temp.write_text(json.dumps(states,indent=2));temp.replace(file)
    if all(p.returncode is not None for g,p,s in jobs):break
    time.sleep(5)
for g,p,s in jobs:s.close();print('FINISHED',g,p.returncode,flush=True)
if any(p.returncode for g,p,s in jobs):raise SystemExit(1)
