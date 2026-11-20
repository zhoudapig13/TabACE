"""Wait for actual v29 shard termination, restore pools, fit complete blocks."""
import json
import os
from pathlib import Path
import time
from continue_tabace_v28 import launch,save,RUN

ROOT=RUN/'results/tabace_full_pool_v34'
OLD=RUN/'results/tabace_context_adaptation_v29'
EXPECTED={('classification','clean'):72,('classification','five'):1440,('classification','seven'):980,
          ('regression','clean'):35,('regression','five'):700,('regression','seven'):924}


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
    except FileNotFoundError:return False


if __name__=='__main__':
    assert json.loads((ROOT/'pool_constructor_audit.json').read_text())['status']=='PASS'
    file=ROOT/'execution.json';assert not file.exists(),'Do not duplicate restored-pool workers'
    state=dict(status='WAITING_FOR_ARCHIVE',pid=os.getpid(),workers=[],fits=[]);workers={};fits={}
    while True:
        prior=json.loads((OLD/'supervisor.json').read_text())
        for gpu in range(8):
            if gpu in workers:
                process,row=workers[gpu]
                if process.poll() is not None:row['exit_code']=process.returncode
                continue
            if prior['stages'].get(str(gpu),0)!=4:continue
            handles=[r['pid'] for r in prior['workers'] if r['gpu']==gpu]
            if not handles or any(alive(pid) for pid in handles):continue
            log=RUN/'logs'/f'full_pool_v34_gpu{gpu}.log'
            process=launch('full_pool_banks_v34.py',['--shard',str(gpu),'--shards','8'],log,gpu)
            row=dict(gpu=gpu,pid=process.pid,log=str(log));workers[gpu]=process,row;state['workers'].append(row)
        for process,row in fits.values():
            if process.poll() is not None:row['exit_code']=process.returncode
        for budget in [256,1024,4096,8192]:
            for (task,block),count in EXPECTED.items():
                key=budget,task,block
                if key in fits or sum(p.poll() is None for p,_ in fits.values())>=6:continue
                if any(len([p for p in (ROOT/'banks'/role/str(budget)/task/block).glob('*/*.json')
                            if '.tmp.' not in p.name and p.with_suffix('.npz').exists()])!=count for role in ['validation','query']):continue
                log=RUN/'logs'/f'full_pool_v34_fusion_{budget}_{task}_{block}.log'
                process=launch('full_pool_fusion_v34.py',['--budget',str(budget),'--task',task,'--block',block],log)
                row=dict(budget=budget,task=task,block=block,pid=process.pid,log=str(log));fits[key]=process,row;state['fits'].append(row)
        state['status']='RUNNING' if workers or fits else 'WAITING_FOR_ARCHIVE';state['updated_unix']=time.time();save(file,state)
        if len(workers)==8 and all(p.poll() is not None for p,_ in workers.values()) and all(p.poll() is not None for p,_ in fits.values()):
            state['status']='ALL_POLICIES_FINISHED_REQUIRES_GOAL_AUDIT' if len(fits)==24 and all(p.returncode==0 for p,_ in workers.values()) else 'FAILED_OR_INCOMPLETE_REQUIRES_REPAIR'
            save(file,state);break
        time.sleep(20)
