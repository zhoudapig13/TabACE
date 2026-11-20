"""Sequential bounded calibration, matched portfolios, then honest reporting."""
import json
import os
from pathlib import Path
import subprocess
import time

RUN=Path(__file__).resolve().parents[1]
OUT=RUN/'results/internal_tabpfn_regression_v1'
PY=RUN.parents[1].parent/'PFN-Research/.tabpfn3-venv/bin/python'


def save(value):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/'completion_status.json';tmp=p.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


def stage(name,commands,gpu=False):
    workers=[]
    for i,command in enumerate(commands):
        env=os.environ.copy()
        for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
        if gpu:env['CUDA_VISIBLE_DEVICES']=str(i)
        path=RUN/'logs'/f'internal_tabpfn_regression_{name}_{i}.log';h=path.open('a',buffering=1)
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
    if any(p.returncode!=0 for p,h,path in workers):raise SystemExit(f'Stage failed: {name}; inspect logs')


if __name__=='__main__':
    state=json.loads((RUN/'results/internal_regression_component_v1/completion_status.json').read_text())
    assert state['status']=='COMPLETE' and state['stage']=='report','Finish TabICL first'
    stage('declare',[[str(RUN/'scripts/internal_tabpfn_regression.py'),'declare']])
    for track in ['clean','five','seven']:
        if (OUT/f'frozen_{track}.json').exists():continue
        stage(track+'_Base',[[str(RUN/'scripts/internal_tabpfn_regression.py'),'run','--track',track,'--shard',str(i),'--shards','8'] for i in range(8)],True)
        stage(track+'_freeze',[[str(RUN/'scripts/internal_tabpfn_regression.py'),'select','--track',track]])
    for phase in ['banks','PHE6','RF_PFN6']:
        stage(phase,[[str(RUN/'scripts/run_internal_tabpfn_regression.py'),phase,'--shard',str(i),'--shards','8'] for i in range(8)],True)
    for track in ['five','seven']:
        stage(track+'_fit',[[str(RUN/'scripts/fit_internal_tabpfn_regression.py'),'--track',track,'--fold',str(i)] for i in range(5)])
    stage('report',[[str(RUN/'scripts/report_internal_tabpfn_regression.py')]])
    save({'status':'COMPLETE','stage':'report','summary':str((OUT/'latest_summary.json').relative_to(RUN)),
          'updated_unix':time.time(),'Markdown_written':False})
    print('TABPFN_REGRESSION_COMPARISON_COMPLETE',flush=True)
