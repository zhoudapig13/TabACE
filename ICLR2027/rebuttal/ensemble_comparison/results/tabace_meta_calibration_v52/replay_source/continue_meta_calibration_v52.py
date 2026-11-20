"""Generate TRAIN meta OOF data, freeze policies, then evaluate queries."""
import json
import os
import time
import meta_calibration_v52 as meta
import probability_head_v51 as classification
import sparse_specific_v47 as parent
from continue_tabace_v28 import RUN,launch,save


if __name__=='__main__':
    path=meta.OUT/'execution.json';assert not path.exists()
    assert json.loads((meta.OUT/'smoke_audit.json').read_text())['status']=='PASS'
    state=dict(status='VALIDATION_RUNNING',pid=os.getpid(),workers=[],selectors=[],query_workers=[],reused_groups=[])
    children=[];selectors=[];queries=[];reused=set();finished=False
    for i in range(8):
        log=RUN/'logs'/f'meta_v52_validation_gpu{i}.log';proc=launch('meta_calibration_v52.py',['validation','--index',str(i),'--count','8'],log,i)
        row=dict(index=i,gpu=i,pid=proc.pid,log=str(log));children.append((proc,row));state['workers'].append(row)
    while True:
        for proc,row in children+selectors+queries:
            if proc.poll() is not None:row['exit_code']=proc.returncode
        if any(r.get('exit_code',0) for _,r in children+selectors+queries):
            state['status']='FAILURE_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if all(p.poll() is not None for p,_ in children) and not selectors:
            assert all(p.returncode==0 for p,_ in children)
            state['status']='SOURCE_SELECTION_RUNNING'
            for budget in [256,1024,4096,8192]:
                log=RUN/'logs'/f'meta_v52_select_{budget}.log';proc=launch('select_meta_calibration_v52.py',['select','--budget',str(budget)],log)
                row=dict(budget=budget,pid=proc.pid,log=str(log));selectors.append((proc,row));state['selectors'].append(row)
        if len(selectors)==4 and all(p.poll() is not None for p,_ in selectors) and not queries:
            assert all(p.returncode==0 for p,_ in selectors)
            state['status']='FROZEN_QUERY_RUNNING'
            for i in range(8):
                log=RUN/'logs'/f'meta_v52_query_gpu{i}.log';proc=launch('select_meta_calibration_v52.py',['query','--index',str(i),'--count','8'],log,i)
                row=dict(index=i,gpu=i,pid=proc.pid,log=str(log));queries.append((proc,row));state['query_workers'].append(row)
        if len(queries)==8 and all(p.poll() is not None for p,_ in queries) and not finished:
            assert all(p.returncode==0 for p,_ in queries)
            log=RUN/'logs/meta_v52_finish.log';proc=launch('select_meta_calibration_v52.py',['finish'],log);code=proc.wait();assert code==0
            state['status']='META_GROUPS_COMPLETE_WAITING_UNCHANGED_GROUPS';finished=True
        for budget in [256,1024,4096,8192]:
            for task,blocks,source in [('classification',['clean','five','seven'],classification.OUT),('regression',['clean','five'],parent.OUT)]:
                for block in blocks:
                    key=budget,task,block
                    if key in reused:continue
                    complete=source/'policies'/str(budget)/task/block/'completion.json'
                    if not complete.exists():continue
                    for name in ['policies','predictions']:
                        dest=meta.OUT/name/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
                        assert not dest.exists();dest.symlink_to(source/name/str(budget)/task/block,target_is_directory=True)
                    reused.add(key);state['reused_groups'].append(dict(budget=budget,task=task,block=block,source=str(complete)))
        state['updated_unix']=time.time();save(path,state)
        if any(r.get('exit_code',0) for _,r in children+selectors+queries):state['status']='FAILURE_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        if finished and len(reused)==20:state['status']='FULL_SCOPE_COMPLETE_REQUIRES_FULL48_AUDIT';save(path,state);break
        time.sleep(15)
