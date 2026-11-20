"""Fit all changed groups; retain every completed unchanged group explicitly."""
import json
import os
import time
import recovery_contexts_v44 as banks
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    path=banks.OUT/'fusion_execution.json';assert not path.exists()
    state=dict(status='WAITING_FOR_BANKS',pid=os.getpid(),fits=[],reused_groups=[]);children={};reused=set()
    while True:
        for proc,row in children.values():
            if proc.poll() is not None:row['exit_code']=proc.returncode
        for budget in [256,1024,4096,8192]:
            for task,blocks in [('classification',['clean','five','seven']),('regression',['clean'])]:
                for block in blocks:
                    key=budget,task,block
                    if key in reused:continue
                    complete=banks.full.OUT/'policies'/str(budget)/task/block/'completion.json'
                    if not complete.exists():continue
                    for name in ['policies','predictions']:
                        dest=banks.OUT/name/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
                        assert not dest.exists();dest.symlink_to(banks.full.OUT/name/str(budget)/task/block,target_is_directory=True)
                    reused.add(key);state['reused_groups'].append(dict(budget=budget,task=task,block=block,source=str(complete)))
            for block,count in [('five',700),('seven',924)]:
                key=budget,block
                if key in children or sum(p.poll() is None for p,_ in children.values())>=4:continue
                if any(sum(p.with_suffix('.npz').exists() for p in (banks.OUT/'banks'/role/str(budget)/'regression'/block).glob('*/*.json') if '.tmp.' not in p.name)!=count for role in ['validation','query']):continue
                log=RUN/'logs'/f'recovery_v44_fusion_{budget}_{block}.log'
                proc=launch('recovery_fusion_v44.py',['--budget',str(budget),'--block',block],log)
                row=dict(budget=budget,block=block,pid=proc.pid,log=str(log));children[key]=proc,row;state['fits'].append(row)
        state['status']='RUNNING' if any(p.poll() is None for p,_ in children.values()) else 'WAITING_FOR_BANKS'
        state['updated_unix']=time.time();save(path,state)
        if any(r.get('exit_code',0) for _,r in children.values()):
            state['status']='FIT_FAILURE_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if len(reused)==16 and len(children)==8 and all(p.poll() is not None for p,_ in children.values()):
            state['status']='FULL_SCOPE_FITS_COMPLETE_REQUIRES_48_METRIC_AUDIT';save(path,state);break
        time.sleep(20)
