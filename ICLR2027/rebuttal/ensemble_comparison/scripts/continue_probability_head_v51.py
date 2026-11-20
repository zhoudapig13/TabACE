"""Fit all twelve classification groups and retain v47 regression results."""
import json
import os
import time
import probability_head_v51 as head
import sparse_specific_v47 as regression
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    assert json.loads((head.OUT/'contract_audit.json').read_text())['status']=='PASS'
    path=head.OUT/'execution.json';assert not path.exists()
    pending=[(b,k) for k in ['five','seven','clean'] for b in [256,1024,4096,8192]]
    state=dict(status='RUNNING',pid=os.getpid(),fits=[],reused_groups=[]);children=[];reused=set()
    while True:
        for proc,row in children:
            if proc.poll() is not None:row['exit_code']=proc.returncode
        busy={row['gpu'] for proc,row in children if proc.poll() is None}
        for gpu in range(8):
            if gpu in busy or not pending:continue
            budget,block=pending.pop(0);log=RUN/'logs'/f'probability_head_v51_{budget}_{block}_gpu{gpu}.log'
            proc=launch('probability_head_v51.py',['fit','--budget',str(budget),'--block',block],log,gpu)
            row=dict(budget=budget,block=block,gpu=gpu,pid=proc.pid,log=str(log));children.append((proc,row));state['fits'].append(row)
        for budget in [256,1024,4096,8192]:
            for block in ['clean','five','seven']:
                key=budget,block
                if key in reused:continue
                source=regression.OUT/'policies'/str(budget)/'regression'/block/'completion.json'
                if not source.exists():continue
                for name in ['policies','predictions']:
                    dest=head.OUT/name/str(budget)/'regression'/block;dest.parent.mkdir(parents=True,exist_ok=True)
                    assert not dest.exists();dest.symlink_to(regression.OUT/name/str(budget)/'regression'/block,target_is_directory=True)
                reused.add(key);state['reused_groups'].append(dict(budget=budget,block=block,source=str(source)))
        state['updated_unix']=time.time();save(path,state)
        if any(row.get('exit_code',0) for _,row in children):state['status']='FAILURE_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if not pending and len(reused)==12 and all(proc.poll() is not None for proc,_ in children):state['status']='ALL_GROUPS_COMPLETE_REQUIRES_FULL48_AUDIT';save(path,state);break
        time.sleep(15)
