"""Complete all regression head groups while preserving full-scope audits."""
import json
import os
import time
import calibration_head_v45 as head
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    assert json.loads((head.OUT/'sample_weight_contract.json').read_text())['status']=='PASS'
    path=head.OUT/'execution.json';assert not path.exists()
    state=dict(status='RUNNING',pid=os.getpid(),fits=[],reused_classification_groups=[]);children={};reused=set()
    while True:
        for proc,row in children.values():
            if proc.poll() is not None:row['exit_code']=proc.returncode
        for budget in [256,1024,4096,8192]:
            for block in ['clean','five','seven']:
                key=budget,block
                completed=head.source.OUT/'policies'/str(budget)/'classification'/block/'completion.json'
                if key not in reused and completed.exists():
                    for name in ['policies','predictions']:
                        dest=head.OUT/name/str(budget)/'classification'/block;dest.parent.mkdir(parents=True,exist_ok=True)
                        assert not dest.exists();dest.symlink_to(head.source.OUT/name/str(budget)/'classification'/block,target_is_directory=True)
                    reused.add(key);state['reused_classification_groups'].append(dict(budget=budget,block=block,source=str(completed)))
                if key in children or sum(p.poll() is None for p,_ in children.values())>=4:continue
                count=dict(clean=35,five=700,seven=924)[block]
                if any(sum(p.with_suffix('.npz').exists() for p in (head.source.OUT/'banks'/role/str(budget)/'regression'/block).glob('*/*.json') if '.tmp.' not in p.name)!=count for role in ['validation','query']):continue
                log=RUN/'logs'/f'calibration_head_v45_{budget}_{block}.log'
                proc=launch('calibration_head_v45.py',['fit','--budget',str(budget),'--block',block],log)
                row=dict(budget=budget,block=block,pid=proc.pid,log=str(log));children[key]=proc,row;state['fits'].append(row)
        state['updated_unix']=time.time();save(path,state)
        if any(r.get('exit_code',0) for _,r in children.values()):
            state['status']='HEAD_FAILURE_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if len(children)==12 and len(reused)==12 and all(p.poll() is not None for p,_ in children.values()):
            state['status']='ALL_GROUPS_COMPLETE_REQUIRES_FULL_GOAL_AUDIT';save(path,state);break
        time.sleep(20)
