"""Execute one predeclared calibration sweep on all GPUs and retain exits."""
import argparse,json,os,subprocess,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['five_classification','seven_regression','seven_benchmark_regression','historical_feature'],default='five_classification');args=parser.parse_args()
worker={'five_classification':'run_severity_calibration.py','seven_regression':'run_seven_severity_calibration.py','seven_benchmark_regression':'run_seven_benchmark_severity.py','historical_feature':'run_historical_feature_severity.py'}[args.task]
prefix={'five_classification':'severity_calibration','seven_regression':'seven_severity_calibration','seven_benchmark_regression':'seven_benchmark_severity','historical_feature':'historical_feature_severity'}[args.task]
jobs=[]
for gpu in range(8):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    stream=(RUN/'logs'/f'{prefix}_gpu{gpu}.log').open('a',buffering=1)
    proc=subprocess.Popen([str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts'/worker),
        '--shard',str(gpu),'--shards','8'],stdout=stream,stderr=subprocess.STDOUT,env=env)
    jobs.append((gpu,proc,stream))
while True:
    status=[{'gpu':g,'pid':p.pid,'exit_code':p.poll(),'status':'RUNNING' if p.returncode is None else 'COMPLETE' if p.returncode==0 else 'FAILED'} for g,p,s in jobs]
    path=RUN/'results'/f'{prefix}_workers.json';tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(status,indent=2));tmp.replace(path)
    if all(p.returncode is not None for g,p,s in jobs):break
    time.sleep(5)
for g,p,s in jobs:s.close();print('FINISHED',g,p.returncode,flush=True)
if any(p.returncode for g,p,s in jobs):raise SystemExit(1)
