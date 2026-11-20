"""Queue adapted candidate inference behind raw validation, without GPU overlap."""
import json
import os
from pathlib import Path
import time
from continue_tabace_v28 import alive, launch, save, RUN

RAW = RUN/'results/tabace_context_reproduction_v28'
OUT = RUN/'results/tabace_context_adaptation_v29'


def predecessor_busy(gpu):
    records=json.loads((RAW/'query_higher_execution.json').read_text())
    supervisor=RAW/'higher_supervisor.json'
    if not supervisor.exists():return True
    state=json.loads(supervisor.read_text())
    validation=[r for r in state['validation'] if r['gpu']==gpu]
    if not validation:return True
    return any(alive(r['pid']) for r in records+validation if r['gpu']==gpu)


def main():
    statefile=OUT/'supervisor.json'
    assert not statefile.exists(),'Duplicate adapted supervisor'
    jobs=[('validation','256'),('query','256'),
          ('validation','1024,4096,8192'),('query','1024,4096,8192')]
    state=dict(status='QUEUED_BEHIND_RAW_VALIDATION',pid=os.getpid(),workers=[],fusion=[])
    active={};stages={gpu:0 for gpu in range(8)};failed=set();fits={}
    expected={('classification','clean'):72,('classification','five'):1440,
              ('classification','seven'):980,('regression','clean'):35,
              ('regression','five'):700,('regression','seven'):924}
    while True:
        for gpu in range(8):
            if gpu in active:
                process,record=active[gpu]
                code=process.poll()
                if code is None:continue
                record['exit_code']=code
                if code:failed.add(gpu)
                else:stages[gpu]+=1
                del active[gpu]
            if gpu in failed or stages[gpu]>=len(jobs) or predecessor_busy(gpu):continue
            role,budgets=jobs[stages[gpu]]
            log=RUN/'logs'/f'tabace_v29_{role}_{budgets.replace(",","-")}_gpu{gpu}.log'
            process=launch('tabace_context_adaptation_v29.py',
                ['worker','--role',role,'--budgets',budgets,'--shard',str(gpu),'--shards','8'],log,gpu)
            record=dict(gpu=gpu,stage=stages[gpu],role=role,budgets=budgets,pid=process.pid,log=str(log))
            state['workers'].append(record);active[gpu]=process,record
        for key,(process,record) in fits.items():
            if process.poll() is not None:record['exit_code']=process.returncode
        for budget in [256,1024,4096,8192]:
            for (task,block),expected_count in expected.items():
                key=budget,task,block
                if key in fits or sum(p.poll() is None for p,_ in fits.values())>=6:continue
                if any(len([p for p in (OUT/'banks'/role/str(budget)/task/block).glob('*/*.json')
                            if '.tmp.' not in p.name and p.with_suffix('.npz').exists()])!=expected_count
                       for role in ['validation','query']):continue
                log=RUN/'logs'/f'tabace_v29_fusion_{budget}_{task}_{block}.log'
                process=launch('tabace_adapted_fusion_v29.py',
                    ['--budget',str(budget),'--task',task,'--block',block],log)
                record=dict(budget=budget,task=task,block=block,pid=process.pid,log=str(log))
                fits[key]=process,record;state['fusion'].append(record)
        state['stages']=stages;state['failed_gpus']=sorted(failed);state['updated_unix']=time.time()
        state['status']='RUNNING' if active else 'WAITING_OR_FINISHING'
        if all(stages[g]>=len(jobs) or g in failed for g in range(8)) and all(p.poll() is not None for p,_ in fits.values()):
            state['status']='FINISHED_REQUIRES_FULL_AUDIT' if len(fits)==24 and not failed else 'FAILED_OR_INCOMPLETE_REQUIRES_REPAIR'
            save(statefile,state)
            return
        save(statefile,state)
        time.sleep(20)


if __name__=='__main__':main()
