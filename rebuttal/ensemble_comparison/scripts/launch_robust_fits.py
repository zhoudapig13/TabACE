import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--scoring',choices=['nll','acc_auc'],default='nll');parser.add_argument('--stable',action='store_true');a=parser.parse_args()
prefix='robust_six_fits' if a.scoring=='nll' else 'robust_six_fits_acc_auc'
if a.stable:prefix='robust_six_fits_stable'
lock=(RUN/'results'/f'{prefix}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
jobs=[]
for track in ['five_feature_calibrated','seven_reimplemented']:
    for fold in range(5):
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        stream=(RUN/'logs'/f'{prefix}_{track}_{fold}.log').open('a',buffering=1)
        command=[str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts/fit_robust_six.py'),'--track',track,'--fold',str(fold),'--scoring',a.scoring]
        if a.stable:command+=['--clean-tolerance','.005','--methods','Native','Random','DENSE','GES','Convex','Loglinear']
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env);jobs.append((track,fold,process,stream))
while True:
    states=[{'track':t,'fold':f,'pid':p.pid,'exit_code':p.poll(),'status':'RUNNING' if p.returncode is None else 'COMPLETE' if p.returncode==0 else 'FAILED'} for t,f,p,s in jobs]
    path=RUN/'results'/f'{prefix}_workers.json';tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(states,indent=2));tmp.replace(path)
    if all(p.returncode is not None for t,f,p,s in jobs):break
    time.sleep(5)
for t,f,p,s in jobs:s.close();print('FINISHED',t,f,p.returncode,flush=True)
if any(p.returncode for t,f,p,s in jobs):raise SystemExit(1)
