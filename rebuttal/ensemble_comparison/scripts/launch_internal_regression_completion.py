"""Resume frozen TabICL regression comparison, preserving every variant."""
import json
import os
from pathlib import Path
import subprocess
import time

RUN=Path(__file__).resolve().parents[1]
OUT=RUN/'results/internal_regression_component_v1'
PY=RUN/'.venv/bin/python'


def save(value):
    p=OUT/'completion_status.json';tmp=p.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


def stage(name,commands,gpu=False):
    workers=[]
    for i,command in enumerate(commands):
        env=os.environ.copy()
        for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
        if gpu:env['CUDA_VISIBLE_DEVICES']=str(i)
        path=RUN/'logs'/f'internal_regression_completion_{name}_{i}.log'
        h=path.open('a',buffering=1)
        p=subprocess.Popen([str(PY),'-u',*command],cwd=RUN,env=env,stdout=h,stderr=subprocess.STDOUT)
        workers.append((p,h,path));print('START',name,i,p.pid,flush=True)
    while True:
        status=[{'pid':p.pid,'exit_code':p.poll(),'log':str(path.relative_to(RUN))} for p,h,path in workers]
        done=all(r['exit_code'] is not None for r in status)
        save({'stage':name,'status':'COMPLETE' if done and all(r['exit_code']==0 for r in status) else 'FAILED' if done else 'RUNNING',
              'workers':status,'updated_unix':time.time(),'Markdown_written':False})
        if done:break
        time.sleep(5)
    for p,h,path in workers:h.close()
    if any(p.returncode!=0 for p,h,path in workers):raise SystemExit(f'Stage failed: {name}; inspect worker logs')


if __name__=='__main__':
    for name in ['clean','five','seven']:assert (OUT/f'frozen_{name}.json').exists()
    stage('five_banks',[[str(RUN/'scripts/run_internal_regression_ensembles.py'),'--tracks','five','--shard',str(i),'--shards','8'] for i in range(8)],True)
    stage('five_fit',[[str(RUN/'scripts/fit_internal_regression_ensembles.py'),'--track','five','--fold',str(i)] for i in range(5)])
    stage('five_geometry',[[str(RUN/'scripts/refine_internal_regression.py'),'geometry','--track','five','--shard',str(i),'--shards','8'] for i in range(8)])
    stage('five_refinement',[[str(RUN/'scripts/refine_internal_regression.py'),'fit','--track','five','--fold',str(i)] for i in range(5)])
    stage('report',[[str(RUN/'scripts/report_internal_regression.py')]])
    save({'status':'COMPLETE','stage':'report','summary':str((OUT/'latest_summary.json').relative_to(RUN)),
          'Markdown_written':False,'updated_unix':time.time()})
    print('TABICL_REGRESSION_COMPARISON_COMPLETE',flush=True)
