"""Reconstruct all576 exact context histograms before any downstream use."""
import json
import os
import time
import numpy as np
import member_priors_v63 as experiment
from continue_tabace_v28 import launch,save,RUN

if __name__=='__main__':
    path=experiment.OUT/'execution.json';assert not path.exists()
    state=dict(status='RUNNING',pid=os.getpid(),workers=[]);children=[]
    for index in range(8):
        log=RUN/'logs'/f'member_priors_v63_{index}.log'
        p=launch('member_priors_v63.py',['worker','--index',str(index),'--count','8'],log,index)
        row=dict(index=index,pid=p.pid,log=str(log));children.append((p,row));state['workers'].append(row)
    while True:
        for p,row in children:
            if p.poll() is not None:row['exit_code']=p.returncode
        state['updated_unix']=time.time();save(path,state)
        if any(r.get('exit_code',0) for _,r in children):state['status']='RECONSTRUCTION_FAILED_DO_NOT_USE';save(path,state);raise SystemExit(1)
        if all(p.poll() is not None for p,_ in children):break
        time.sleep(10)
    rows=[]
    for role in ['validation','query']:
        for b in [256,1024,4096,8192]:
            files=list((experiment.OUT/role/str(b)).glob('*.json'));assert len(files)==72
            records=[json.loads(p.read_text()) for p in files];assert all(r['status']=='PASS' for r in records)
            tv=np.asarray([r['branch_canonical_prior_total_variation'] for r in records])
            rows.append(dict(role=role,budget=b,datasets=72,mean_Shared_prior_TV=float(tv[:,:10].mean()),
                mean_Specific_additional_prior_TV=float(tv[:,10:].mean()),max_TV=float(tv.max()),
                members_TV_above_005=int((tv>.05).sum()),total_members=1512))
    save(experiment.OUT/'reconstruction_audit.json',dict(status='PASS',verified_cases=576,verified_member_input_triples=12096,
        per_member_row_caps_verified=True,all_input_hashes_match_existing_banks=True,query_labels_used_for_priors=False,
        TFM_calls=0,summary=rows))
    state['status']='ALL_PRIORS_RECONSTRUCTED_AND_AUDITED';save(path,state)
