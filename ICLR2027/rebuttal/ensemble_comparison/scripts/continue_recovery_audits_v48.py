"""Audit completed stages without claiming metadata alone proves the goal."""
import json
import os
import time
from pathlib import Path
from continue_tabace_v28 import RUN,launch,save

if __name__=='__main__':
    root=RUN/'results/tabace_recovery_audits_v48';root.mkdir(exist_ok=True)
    path=root/'execution.json';assert not path.exists()
    stages=['tabace_recovery_contexts_v44','tabace_calibration_head_v45','tabace_clean_role_head_v46','tabace_sparse_specific_v47']
    state=dict(status='RUNNING',pid=os.getpid(),numeric_audits=[],bank_audit=None);done=set();children={};bank=None
    while True:
        if bank is None:
            workers=list((RUN/'results/tabace_recovery_contexts_v44/workers').glob('*.json'))
            if len(workers)==8 and all(json.loads(p.read_text())['status']=='COMPLETE' for p in workers):
                log=RUN/'logs/recovery_v44_bank_audit.log';proc=launch('audit_recovery_banks_v44.py',[],log);row=dict(pid=proc.pid,log=str(log));bank=proc,row;state['bank_audit']=row
        if bank is not None and bank[0].poll() is not None:bank[1]['exit_code']=bank[0].returncode
        for name in stages:
            if name in children:continue
            stage=RUN/'results'/name
            complete=sum((stage/'policies'/str(budget)/task/block/'completion.json').exists() for budget in [256,1024,4096,8192] for task in ['classification','regression'] for block in ['clean','five','seven'])
            if complete!=24:continue
            log=RUN/'logs'/f'{name}_full_numeric_audit.log'
            proc=launch('report_recovery_goal_stage.py',['--stage',name],log)
            row=dict(stage=name,pid=proc.pid,log=str(log));children[name]=proc,row;state['numeric_audits'].append(row)
        for proc,row in children.values():
            if proc.poll() is not None:row['exit_code']=proc.returncode
        state['updated_unix']=time.time();save(path,state)
        if any(r.get('exit_code',0) for _,r in children.values()) or (bank is not None and bank[1].get('exit_code',0)):
            state['status']='AUDIT_FAILURE_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if len(children)==4 and all(p.poll() is not None for p,_ in children.values()) and bank is not None and bank[0].poll() is not None:
            state['status']='AUDITS_FINISHED_NOT_A_GOAL_COMPLETION_CLAIM';save(path,state);break
        time.sleep(20)
