"""Parallel, resumable Base probes; never modifies the completed v1 experiment."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import time

RUN=Path(__file__).resolve().parents[1]
OUT=RUN/'results/internal_regression_benchmark_v2'
PY=RUN/'.venv/bin/python'


def save(value):
    p=OUT/'calibration_status.json';tmp=p.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe-plan',default='probe_plan_v1.json')
    parser.add_argument('--round-tag',default='round1');args=parser.parse_args()
    workers=[]
    for gpu in range(8):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
        path=RUN/'logs'/f'tabicl_regression_v2_{args.round_tag}_base_probe_{gpu}.log';h=path.open('a',buffering=1)
        p=subprocess.Popen([str(PY),'-u',str(RUN/'scripts/recalibrate_tabicl_regression_v2.py'),'probe',
            '--shard',str(gpu),'--shards','8','--probe-plan',args.probe_plan],cwd=RUN,env=env,stdout=h,stderr=subprocess.STDOUT)
        workers.append((p,h,path));print('V2_PROBE_STARTED',gpu,p.pid,flush=True)
    while True:
        rows=[{'pid':p.pid,'exit_code':p.poll(),'log':str(path.relative_to(RUN))} for p,h,path in workers]
        done=all(r['exit_code'] is not None for r in rows)
        save({'stage':'Base_probes','status':'COMPLETE' if done and all(r['exit_code']==0 for r in rows) else 'FAILED' if done else 'RUNNING',
              'workers':rows,'updated_unix':time.time()})
        if done:break
        time.sleep(5)
    for p,h,path in workers:h.close()
    if any(p.returncode for p,h,path in workers):raise SystemExit('Probe worker failure; inspect logs')
    with (RUN/'logs/tabicl_regression_v2_selection.log').open('a') as h:
        subprocess.run([str(PY),'scripts/recalibrate_tabicl_regression_v2.py','select','--include-tail'],cwd=RUN,stdout=h,stderr=subprocess.STDOUT,check=True)
    result=json.loads((OUT/'source_tail_profiles_selection.json').read_text())
    if not result['within_005_each']:
        save({'stage':'selection','status':'PROBES_COMPLETE_NO_PASS','metrics':result['metrics'],'difference':result['difference']})
        print('V2_NO_PASS_YET',result['metrics'],result['difference'],flush=True)
    else:
        save({'stage':'materialize','status':'RUNNING','metrics':result['metrics']})
        with (RUN/'logs/tabicl_regression_v2_materialize.log').open('a') as h:
            subprocess.run([str(PY),'scripts/recalibrate_tabicl_regression_v2.py','materialize','--stage','source_tail_profiles'],cwd=RUN,stdout=h,stderr=subprocess.STDOUT,check=True)
        save({'stage':'benchmark_frozen','status':'COMPLETE','metrics':result['metrics'],'all_six_checks_final_audit_pending':True})
        print('STRICT_V2_BENCHMARK_FROZEN',result['metrics'],flush=True)
