"""Run eight classifier groups, giving the live smoke test sole use of GPU0."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import augmented_classifier_v54 as experiment
from continue_tabace_v28 import launch,save,RUN


def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists():return False
    return p.read_text().split(') ',1)[1].split()[0]!='Z'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke-pid',type=int,required=True);args=parser.parse_args()
    assert json.loads((experiment.OUT/'generator_contract_audit.json').read_text())['status']=='PASS'
    path=experiment.OUT/'execution.json';assert not path.exists()
    pending=[(b,k) for k in ['clean','seven'] for b in [256,1024,4096,8192] if (b,k)!=(256,'seven')]
    state=dict(status='RUNNING',pid=os.getpid(),smoke_pid=args.smoke_pid,fits=[]);children=[];smoke_checked=False
    while True:
        for p,row in children:
            if p.poll() is not None:row['exit_code']=p.returncode
        if any(row.get('exit_code',0) for _,row in children):
            state['status']='FAILED_REQUIRES_REPAIR';save(path,state);raise SystemExit(1)
        smoke_live=alive(args.smoke_pid)
        if not smoke_live and not smoke_checked:
            meta=experiment.nn.source.adaptation.raw.frozen.cohort('classification','seven')[0]
            root=experiment.OUT/'experts/256/seven'/str(meta['task_id'])
            for role in ['Shared']+[f'Specific_{f}' for f in experiment.FAMILIES]:
                p=root/role/'source_predictions.npz';info=json.loads(p.with_suffix('.json').read_text())
                with np.load(p) as z:
                    assert not np.intersect1d(z['training_original_ids'],z['validation_original_ids']).size
                    assert len(z['training_original_ids'])<=256
                    assert all(np.isfinite(z[k]).all() for k in z.files)
                assert all(Path(p).exists() for p in info['query_model_paths'].values())
            save(experiment.OUT/'smoke_audit.json',dict(status='PASS',task_id=meta['task_id'],roles=8,
                source_validation_disjoint=True,unique_row_cap=True,all_saved_predictions_finite=True))
            smoke_checked=True;pending.append((256,'seven'))
        busy={row['gpu'] for p,row in children if p.poll() is None}
        if smoke_live:busy.add(0)
        for gpu in range(8):
            if gpu in busy or not pending:continue
            budget,block=pending.pop(0);log=RUN/'logs'/f'augmented_classifier_v54_{budget}_{block}_gpu{gpu}.log'
            p=launch('augmented_classifier_v54.py',['fit','--budget',str(budget),'--block',block],log,gpu)
            row=dict(budget=budget,block=block,gpu=gpu,pid=p.pid,log=str(log));children.append((p,row));state['fits'].append(row)
        state['updated_unix']=time.time();state['smoke_live']=smoke_live;save(path,state)
        if smoke_checked and not pending and all(p.poll() is not None for p,_ in children):
            state['status']='ALL_GROUPS_COMPLETE_REQUIRES_FULL48_AUDIT';save(path,state);break
        time.sleep(10)
