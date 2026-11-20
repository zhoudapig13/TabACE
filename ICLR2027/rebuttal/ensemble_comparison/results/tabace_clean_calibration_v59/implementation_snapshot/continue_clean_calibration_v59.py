"""Run expanded source calibration, then freeze and evaluate its one recipe."""
import os
import time
import clean_calibration_v59 as experiment
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    path=experiment.OUT/'execution.json';assert not path.exists()
    state=dict(status='SOURCE_CALIBRATION_RUNNING',pid=os.getpid(),workers=[]);children=[]
    p=launch('clean_calibration_v59.py',['native'],RUN/'logs/clean_calibration_v59_native_gpu3.log',3)
    row=dict(phase='native',pid=p.pid,gpu=3,torch_memory_fraction=.2,log=str(RUN/'logs/clean_calibration_v59_native_gpu3.log'))
    children.append((p,row));state['workers'].append(row)
    for index in range(8):
        log=RUN/'logs'/f'clean_calibration_v59_experts_{index}.log'
        p=launch('clean_calibration_v59.py',['experts','--index',str(index),'--count','8'],log,index)
        row=dict(phase='experts',index=index,pid=p.pid,log=str(log));children.append((p,row));state['workers'].append(row)
    for phase in ['source','select','query']:
        if phase!='source':
            state['status']=f'{phase.upper()}_RUNNING';log=RUN/'logs'/f'clean_calibration_v59_{phase}.log'
            p=launch('clean_calibration_v59.py',[phase],log,0);row=dict(phase=phase,pid=p.pid,log=str(log));children=[(p,row)];state['workers'].append(row)
        while True:
            for p,row in children:
                if p.poll() is not None:row['exit_code']=p.returncode
            state['updated_unix']=time.time();save(path,state)
            if any(r.get('exit_code',0) for _,r in children):
                state['status']='FAILED_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
            if all(p.poll() is not None for p,_ in children):break
            time.sleep(10)
    state['status']='TARGETED_GROUP_COMPLETE_REQUIRES_REPLAY_AND_FULL48_AUDIT';save(path,state)
