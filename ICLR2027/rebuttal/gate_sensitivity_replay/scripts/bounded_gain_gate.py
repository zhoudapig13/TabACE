"""Third gate iteration: retain small original updates and calibrate bounded gain.

The 12-sigma redescending radius is fixed before this evaluation. All four cases
are reported; no outer-test-based selection between gate iterations is performed.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from replay_diagnostics import OUT,ROOT,cells,regress,write_json
from calibrated_regression_repairs import components

DEST=OUT/'algorithm_repairs_v3'
RADIUS=12.

def predict(b,k,theta,beta):
    ag,a,f=components(b,k,theta)
    gate=np.clip(a*(1+.5*np.tanh(f@beta)),0,1)
    gate*=np.minimum(1.,(RADIUS/np.maximum(np.abs(ag-b),1e-12))**2)
    out=b+gate*(ag-b)
    assert np.all(np.abs(out-b)<=RADIUS+1e-7)
    return out

def fit(train,theta,track):
    groups=defaultdict(float)
    for c in train:groups[c['id']]+=(5. if track=='five' else 7.) if c['meta']['rho']==0 else 1.
    parts=[]
    for c in train:
        ag,a,f=components(c['b'],c['k'],theta)
        clean_weight=(5. if track=='five' else 7.) if c['meta']['rho']==0 else 1.
        w=clean_weight/(len(groups)*groups[c['id']]*len(c['y'])*max(float(c['y'].var()),1e-12))
        parts.append((c['b'],ag,a,f,c['y'],np.full(len(a),w)))
    B,AG,A,F,Y,W=[torch.as_tensor(np.concatenate([v[i] for v in parts]),dtype=torch.float64) for i in range(6)]
    factor=torch.minimum(torch.ones_like(B),(RADIUS/torch.abs(AG-B).clamp(min=1e-12))**2)
    def objective(x):
        beta=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        alpha=torch.clamp(A*(1+.5*torch.tanh(F@beta)),0,1)*factor
        pred=B+alpha*(AG-B)
        loss=(W*(pred-Y)**2).sum()+.001*(beta**2).mean()
        loss.backward();return float(loss.detach()),beta.grad.detach().numpy()
    initial=np.zeros(F.shape[1]);before=objective(initial)[0]
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(-4,4)]*len(initial),
        options={'maxiter':160,'ftol':1e-10,'gtol':1e-6})
    assert np.isfinite(result.x).all()
    return result.x,dict(success=bool(result.success),iterations=int(result.nit),message=str(result.message),
        initial_training_loss=before,final_training_loss=float(result.fun))

def prepare():
    p=DEST/'frozen_plan.json'
    if p.exists():raise RuntimeError('Do not overwrite frozen plan')
    write_json(p,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='Preserve naturally small updates while learning modest query-dependent gain',
        hypothesis='V2 logit shifts can greatly amplify tiny original gates; fixed 3-sigma radius also suppresses legitimate large updates',
        formula='alpha_new=clip(alpha_original*(1+0.5*tanh(features@beta)),0,1)*min(1,(12/d)^2)',
        radius_source_sd=RADIUS,regularization=.001,head_dimension=6,
        calibration='Training dataset folds only; equal-dataset normalized MSE with clean weight 5/7',
        held_out_labels_used_for_training=False,hyperparameter_search=False,
        all_four_cases_reported=True,previous_iterations_retained=True,backbone_calls=0,markdown_written=False))

def run(backbone,track):
    dataset=[]
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        dataset.append(dict(id=int(m['data_id']),meta=m,b=(b-sm)/ss,k=((k-sm)/ss).T,y=(y-sm)/ss,
            raw_b=b,raw_k=k,raw_y=y,sm=sm,ss=ss,fold=fold,ref=ref))
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    case=next(c for c in plan['cases'] if (c['backbone'],c['track'],c['task'])==(backbone,track,'regression'))
    rows=[]
    for record in case['policies']:
        p=json.loads((ROOT/record['path']).read_text());theta=np.asarray(p['parameters'])
        train_ids=set(p['train_data_ids']);test_ids=set(p['test_data_ids']);assert train_ids.isdisjoint(test_ids)
        train=[c for c in dataset if c['id'] in train_ids];test=[c for c in dataset if c['id'] in test_ids]
        beta,audit=fit(train,theta,track)
        write_json(DEST/backbone/track/f'fold_{p["fold"]}.json',dict(parameters=beta.tolist(),outer_fold=p['fold'],
            source_data_ids=sorted(train_ids),evaluation_data_ids=sorted(test_ids),original_policy_sha256=record['sha256'],
            optimization=audit,held_out_labels_used_for_training=False))
        print('BOUNDED_GAIN_FIT',backbone,track,p['fold'],audit,flush=True)
        for c in test:
            m,y=c['meta'],c['raw_y'];old=regress(c['raw_b'],c['raw_k'],theta,c['sm'],c['ss'])[0]
            new=predict(c['b'],c['k'],theta,beta)*c['ss']+c['sm']
            scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
            assert abs(float(np.sqrt(np.mean((old-y)**2))/scale)-c['ref']['nrmse'])<1e-8*max(1,c['ref']['nrmse'])
            for method,pred in [('Original',old),('A_bounded_gain',new)]:
                rows.append(dict(backbone=backbone,track=track,data_id=c['id'],family=m['family'],rho=m['rho'],
                    outer_fold=p['fold'],method=method,n=len(y),r2=1-float(np.sum((pred-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                    nrmse=float(np.sqrt(np.mean((pred-y)**2))/scale)))
    dest=DEST/backbone/track;pd.DataFrame(rows).to_csv(dest/'performance.csv',index=False)
    write_json(dest/'audit.json',dict(original_policy_replay_verified=True,source_test_disjoint=True,
        held_out_labels_used_for_training=False,radius_source_sd=RADIUS))
    print('BOUNDED_GAIN_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--backbone');p.add_argument('--track')
    a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.backbone,a.track)
