"""Run the declared Specific expansion and keep all unchanged goal groups."""
import json
import os
import time
import sparse_specific_v47 as sparse
import clean_role_head_v46 as clean
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    assert json.loads((sparse.OUT/'frozen_Shared_OOF_replay_audit.json').read_text())['status']=='PASS'
    path=sparse.OUT/'execution.json';assert not path.exists()
    state=dict(status='RUNNING',pid=os.getpid(),fits=[],reused_groups=[]);children={};reused=set()
    while True:
        for proc,row in children.values():
            if proc.poll() is not None:row['exit_code']=proc.returncode
        for budget in [256,1024,4096,8192]:
            for task,blocks in [('classification',['clean','five','seven']),('regression',['clean'])]:
                for block in blocks:
                    key=budget,task,block
                    if key in reused:continue
                    source=clean.OUT/'policies'/str(budget)/task/block/'completion.json'
                    if not source.exists():continue
                    for name in ['policies','predictions']:
                        dest=sparse.OUT/name/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
                        assert not dest.exists();dest.symlink_to(clean.OUT/name/str(budget)/task/block,target_is_directory=True)
                    reused.add(key);state['reused_groups'].append(dict(budget=budget,task=task,block=block,source=str(source)))
            for block in ['five','seven']:
                key=budget,block
                if key in children or sum(p.poll() is None for p,_ in children.values())>=4:continue
                source=sparse.h.OUT/'policies'/str(budget)/'regression'/block/'completion.json'
                if not source.exists():continue
                log=RUN/'logs'/f'sparse_specific_v47_{budget}_{block}.log'
                proc=launch('sparse_specific_v47.py',['fit','--budget',str(budget),'--block',block],log)
                row=dict(budget=budget,block=block,pid=proc.pid,log=str(log));children[key]=proc,row;state['fits'].append(row)
        state['updated_unix']=time.time();save(path,state)
        if any(r.get('exit_code',0) for _,r in children.values()):state['status']='FAILURE_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if len(reused)==16 and len(children)==8 and all(p.poll() is not None for p,_ in children.values()):state['status']='FULL_SCOPE_COMPLETE_REQUIRES_GOAL_AUDIT';save(path,state);break
        time.sleep(20)
