"""Resume v29 orchestration after an explicitly recorded case redistribution."""
import json
import os
from pathlib import Path
import time
from continue_tabace_v28 import launch,save,RUN

ROOT=RUN/'results/tabace_context_adaptation_v29'
DEVICES=[0,1,2,3,5,7]
EXPECTED={('classification','clean'):72,('classification','five'):1440,('classification','seven'):980,
          ('regression','clean'):35,('regression','five'):700,('regression','seven'):924}


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
    except FileNotFoundError:return False


def main():
    file=ROOT/'supervisor.json';state=json.loads(file.read_text())
    assert state.get('scheduler_revision')!='v36','Avoid duplicate resume supervisors'
    state['previous_supervisor_pid']=state['pid'];state['pid']=os.getpid();state['scheduler_revision']='v36'
    children=[];fits={}
    devices=json.loads((ROOT/'redistribution_v36/resources.json').read_text())['devices']
    assert devices
    for index,gpu in enumerate(devices):
        log=RUN/'logs'/f'canonical_v36_worker{index}_gpu{gpu}.log'
        process=launch('finish_canonical_banks_v36.py',['--index',str(index),'--count',str(len(devices))],log,gpu)
        row=dict(gpu=gpu,stage=3,role='query',budgets='1024,4096,8192',
            source_logical_shards=[2,3],case_partition=index,case_partitions=len(devices),pid=process.pid,log=str(log))
        state['workers'].append(row);children.append((process,row))
    for row in state['fusion']:
        fits[row['budget'],row['task'],row['block']]=(None,row)
    state['status']='RUNNING_REDISTRIBUTED_CASES';save(file,state)
    while True:
        for process,row in children:
            if process.poll() is not None:row['exit_code']=process.returncode
        for key,(process,row) in list(fits.items()):
            if process is not None and process.poll() is not None:row['exit_code']=process.returncode
            if process is None and not alive(row['pid']):
                done=ROOT/'policies'/str(row['budget'])/row['task']/row['block']/'completion.json'
                if done.exists():row['exit_code']=0
                elif row.get('exit_code') is None:row['exit_code']='TERMINAL_WITHOUT_COMPLETION'
        for budget in [256,1024,4096,8192]:
            for (task,block),count in EXPECTED.items():
                key=budget,task,block
                if key in fits:continue
                if sum((p.poll() is None if p is not None else alive(r['pid'])) for p,r in fits.values())>=6:continue
                ready=all(len([p for p in (ROOT/'banks'/role/str(budget)/task/block).glob('*/*.json')
                               if '.tmp.' not in p.name and p.with_suffix('.npz').exists()])==count for role in ['validation','query'])
                if not ready:continue
                log=RUN/'logs'/f'tabace_v29_fusion_{budget}_{task}_{block}.log'
                process=launch('tabace_adapted_fusion_v29.py',['--budget',str(budget),'--task',task,'--block',block],log)
                row=dict(budget=budget,task=task,block=block,pid=process.pid,log=str(log));fits[key]=process,row;state['fusion'].append(row)
        all_workers_done=all(p.poll() is not None for p,_ in children)
        if all_workers_done:
            clean=all(p.returncode==0 for p,_ in children)
            if clean:
                # Verify every expected query case of the original logical
                # shards before releasing their devices to v34.
                from finish_canonical_banks_v36 import jobs
                import tabace_context_adaptation_v29 as experiment
                missing=[]
                for task,track,meta in jobs():
                    for budget in [1024,4096,8192]:
                        for block in ['clean','five'] if track=='five' else ['seven']:
                            for family,rho in experiment.raw.case_grid(block):
                                path=ROOT/'banks/query'/str(budget)/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                                if not path.exists() or not path.with_suffix('.json').exists():missing.append(str(path))
                state['redistribution_missing_cases']=missing
                if not missing:state['stages']['2']=4;state['stages']['3']=4
            else:state['failed_gpus']=sorted({r['gpu'] for p,r in children if p.returncode})
        state['updated_unix']=time.time();save(file,state)
        all_fits_done=all((p.poll() is not None if p is not None else not alive(r['pid'])) for p,r in fits.values())
        if all_workers_done and all_fits_done:
            state['status']='FINISHED_REQUIRES_FULL_AUDIT' if len(fits)==24 and not state.get('failed_gpus') and not state.get('redistribution_missing_cases') else 'FAILED_OR_INCOMPLETE_REQUIRES_REPAIR'
            save(file,state);return
        time.sleep(20)


if __name__=='__main__':main()
