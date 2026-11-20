"""Keep the full comparison matrix while correcting only Clean role handling."""
import json
import os
import time
import clean_role_head_v46 as corrected
from continue_tabace_v28 import RUN,launch,save

if __name__=='__main__':
    path=corrected.OUT/'execution.json';assert not path.exists()
    state=dict(status='RUNNING',pid=os.getpid(),fits=[],reused_groups=[]);children=[];reused=set()
    for budget in [256,1024,4096,8192]:
        log=RUN/'logs'/f'clean_role_head_v46_{budget}.log';proc=launch('clean_role_head_v46.py',['fit','--budget',str(budget)],log)
        row=dict(budget=budget,pid=proc.pid,log=str(log));children.append((proc,row));state['fits'].append(row)
    while True:
        for proc,row in children:
            if proc.poll() is not None:row['exit_code']=proc.returncode
        for budget in [256,1024,4096,8192]:
            for task,blocks in [('classification',['clean','five','seven']),('regression',['five','seven'])]:
                for block in blocks:
                    key=budget,task,block
                    if key in reused:continue
                    complete=corrected.head.OUT/'policies'/str(budget)/task/block/'completion.json'
                    if not complete.exists():continue
                    for name in ['policies','predictions']:
                        dest=corrected.OUT/name/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
                        assert not dest.exists();dest.symlink_to(corrected.head.OUT/name/str(budget)/task/block,target_is_directory=True)
                    reused.add(key);state['reused_groups'].append(dict(budget=budget,task=task,block=block,source=str(complete)))
        state['updated_unix']=time.time();save(path,state)
        if any(r.get('exit_code',0) for _,r in children):state['status']='FAILURE_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if len(reused)==20 and all(p.poll() is not None for p,_ in children):state['status']='ALL_GROUPS_COMPLETE_REQUIRES_FULL_AUDIT';save(path,state);break
        time.sleep(20)
