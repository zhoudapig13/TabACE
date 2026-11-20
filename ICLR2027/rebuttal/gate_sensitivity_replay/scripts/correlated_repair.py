"""Fixed source-scale trust gate against common-mode branch errors."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from replay_diagnostics import OUT, cells, regress, write_json
from algorithm_repairs import REPAIR

DEST=REPAIR/'correlated_gate'
MAGNITUDES=[1.,5.,10.,100.]

def trust(base,bank,weights,alpha,source_scale):
    aggregate=(weights*bank).sum(0)
    disagreement=np.abs(aggregate-base)/source_scale
    multiplier=np.minimum(1.,(3./np.maximum(disagreement,1e-12))**2)
    output=base+alpha*multiplier*(aggregate-base)
    assert np.all(np.abs(output-base)/source_scale <= 3.+1e-8)
    return output, multiplier

def prepare():
    write_json(DEST/'frozen_plan.json',dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=0,
        formula='alpha_trust=alpha_original*min(1,(3*source_std/abs(aggregate-Base))^2)',
        width_source_sd=3.,evaluation_labels_used_for_fitting=False,hyperparameter_search=False,
        stress='Add a common +/-k source_std bias to 3/5 or 5/5 alternatives; cycle all subsets and signs',
        selection='Per-query worst absolute output displacement, without evaluation labels',
        magnitudes=MAGNITUDES,contexts_and_original_weights_unchanged=True,
        benchmark_unchanged=True,original_predictions_not_overwritten=True))

def run(backbone,track):
    scores=[];counts=[];stress=[];pool=defaultdict(list)
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        original,w,a=regress(b,k,t,sm,ss)
        repaired,multiplier=trust(b,k,w,a,ss)
        scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        assert abs(np.sqrt(np.mean((original-y)**2))/scale-ref['nrmse']) < 1e-8*max(1,ref['nrmse'])
        for method,p in [('Original',original),('Trust_gate',repaired)]:
            scores.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                method=method,n=len(y),r2=1-float(np.sum((p-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                nrmse=float(np.sqrt(np.mean((p-y)**2))/scale)))
        counts.append(dict(data_id=m['data_id'],family=m['family'],rho=m['rho'],n=len(y),
            trust_activated=int((multiplier<1).sum()),changed=int((np.abs(original-repaired)>1e-10*ss).sum())))
        for number in [3,5]:
            for magnitude in MAGNITUDES:
                worst={name:np.zeros(len(y)) for name in ['Original','Trust_gate']}
                for subset in combinations(range(5),number):
                    for sign in [-1.,1.]:
                        shifted=k.copy();shifted[list(subset)]+=sign*magnitude*ss
                        old,sw,sa=regress(b,shifted,t,sm,ss)
                        new,_=trust(b,shifted,sw,sa,ss)
                        for name,p,p0 in [('Original',old,original),('Trust_gate',new,repaired)]:
                            worst[name]=np.maximum(worst[name],np.abs(p-p0)/ss)
                for name,d in worst.items():
                    pool[(number,magnitude,name)].append(d)
                    stress.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                        method=name,biased_branches=number,magnitude_source_sd=magnitude,n=len(y),
                        mean_displacement_source_sd=float(d.mean()),p95_displacement_source_sd=float(np.quantile(d,.95))))
        if i%100==0:print('CORRELATED',backbone,track,i,flush=True)
    dest=DEST/backbone/track;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(scores).to_csv(dest/'performance.csv',index=False)
    pd.DataFrame(counts).to_csv(dest/'activation_counts.csv',index=False)
    pd.DataFrame(stress).to_csv(dest/'stress_by_condition.csv',index=False)
    pooled=[]
    for (number,magnitude,name),parts in pool.items():
        d=np.concatenate(parts)
        pooled.append(dict(backbone=backbone,track=track,method=name,biased_branches=number,
            magnitude_source_sd=magnitude,n_query_conditions=len(d),mean_displacement_source_sd=float(d.mean()),
            p95_displacement_source_sd=float(np.quantile(d,.95)),max_displacement_source_sd=float(d.max())))
    pd.DataFrame(pooled).to_csv(dest/'stress_pooled.csv',index=False)
    write_json(dest/'audit.json',dict(original_policy_replay_verified=True,
        source_scale_only=True,evaluation_labels_used_for_gate=False,maximum_update_source_sd=3.,
        no_branch_prediction_clipping=True,n_datasets=len({s['data_id'] for s in scores})))
    print('CORRELATED_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--backbone');p.add_argument('--track')
    a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.backbone,a.track)
