"""Training-fold-calibrated regression safeguards, evaluated on held-out datasets.

A: learn when to use the aggregate, while retaining a redescending trust gate.
C: recalibrate a bounded, residual-penalized branch fusion after median clipping.
All original banks and policies remain read-only.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import torch
from replay_diagnostics import OUT, ROOT, cells, regress, softmax, write_json

DEST=OUT/'algorithm_repairs_v2'
REGULARIZATION=.001

def prepare():
    record=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='Improve natural regression accuracy while retaining protections against correlated and single-branch errors',
        source_only=True,held_out_labels_used_for_training=False,hyperparameter_search=False,
        regularization=REGULARIZATION,optimizer='L-BFGS-B, maximum 160 iterations',
        objective='Equal-dataset mean normalized squared error (1-R2), clean conditions weighted 5/7',
        A=dict(head_features=['intercept','log1p(Base-aggregate distance)','original_support',
                'log1p(branch_MAD)','log1p(abs(standardized_Base))','signed_aggregate_Base_distance'],
            gate='sigmoid(logit(original_alpha)+features @ beta) * min(1,(3/d)^2)',
            original_aggregation_frozen=True),
        C=dict(guard='query median +/-3 source label SD before fusion',
            weighting='softmax(role_prior+eta*support-lambda*log1p(squared median residual))',
            recalibration='13 parameters: original 12 fusion parameters plus residual penalty lambda'),
        contexts_unchanged=True,backbone_calls=0,input_scope='Appendix-rule reconstructed caches',
        original_results_preserved=True,markdown_written=False)
    path=DEST/'frozen_plan.json'
    if path.exists():raise RuntimeError('Plan already exists: do not overwrite an evaluated experiment')
    write_json(path,record)

def components(b,k,t):
    median=np.median(k,axis=1)
    residual=np.abs(k-median[:,None])
    agreement=np.exp(-residual/np.exp(t[-2]))
    w=softmax(t[:5]+agreement*t[5:10],1)
    aggregate=(w*k).sum(1)
    support=(w*agreement).sum(1)
    difference=aggregate-b;distance=np.abs(difference)
    alpha=t[-1]*(1-np.exp(-distance/np.exp(t[-2])))*support
    features=np.c_[np.ones(len(b)),np.log1p(distance),support,
        np.log1p(np.median(residual,axis=1)),np.log1p(np.abs(b)),np.tanh(difference/3.)]
    return aggregate,alpha,features

def numpy_predict(variant,b,k,original_theta,parameters):
    if variant=='A_learned_trust':
        aggregate,alpha,features=components(b,k,original_theta)
        alpha_new=expit(np.log(np.clip(alpha,1e-6,1-1e-6)/(1-np.clip(alpha,1e-6,1-1e-6)))+features@parameters)
        alpha_new=np.where(alpha==0,0.,alpha_new)
        d=np.abs(aggregate-b)
        alpha_new*=np.minimum(1.,(3./np.maximum(d,1e-12))**2)
        out=b+alpha_new*(aggregate-b)
        assert np.all(np.abs(out-b)<=3.+1e-7)
        return out
    median=np.median(k,axis=1)
    bounded=median[:,None]+np.clip(k-median[:,None],-3.,3.)
    residual=np.abs(bounded-median[:,None])
    agreement=np.exp(-residual/np.exp(parameters[-3]))
    weights=softmax(parameters[:5]+agreement*parameters[5:10]-parameters[-1]*np.log1p(residual**2),1)
    aggregate=(weights*bounded).sum(1)
    support=(weights*agreement).sum(1)
    alpha=parameters[-2]*(1-np.exp(-np.abs(aggregate-b)/np.exp(parameters[-3])))*support
    out=b+alpha*(aggregate-b)
    assert np.all(out>=np.minimum(b,bounded.min(1))-1e-7)
    assert np.all(out<=np.maximum(b,bounded.max(1))+1e-7)
    return out

def fit(variant,train,theta,track):
    groups=defaultdict(float)
    for c in train:groups[c['id']]+=5. if track=='five' and c['meta']['rho']==0 else 7. if c['meta']['rho']==0 else 1.
    B=np.concatenate([c['b'] for c in train]);K=np.concatenate([c['k'] for c in train]);Y=np.concatenate([c['y'] for c in train])
    weights=[]
    for c in train:
        clean_weight=(5. if track=='five' else 7.) if c['meta']['rho']==0 else 1.
        w=clean_weight/(len(groups)*groups[c['id']]*len(c['y'])*max(float(c['y'].var()),1e-12))
        weights.append(np.full(len(c['y']),w))
    B,K,Y,W=[torch.as_tensor(a,dtype=torch.float64) for a in [B,K,Y,np.concatenate(weights)]]
    if variant=='A_learned_trust':
        ag,old_alpha,features=components(B.numpy(),K.numpy(),theta)
        AG,AA,F=[torch.as_tensor(v,dtype=torch.float64) for v in [ag,old_alpha,features]]
        LOGIT=torch.logit(AA.clamp(1e-6,1-1e-6))
        factor=torch.minimum(torch.ones_like(AG),(3./torch.abs(AG-B).clamp(min=1e-12))**2)
        initial=np.zeros(F.shape[1]);bounds=[(-6,6)]*len(initial)
        def mapping(p):
            alpha=torch.sigmoid(LOGIT+F@p)
            alpha=torch.where(AA==0,torch.zeros_like(alpha),alpha)*factor
            return B+alpha*(AG-B)
    else:
        median=torch.median(K,dim=1).values
        bounded=median[:,None]+torch.clamp(K-median[:,None],-3.,3.)
        residual=torch.abs(bounded-median[:,None])
        initial=np.r_[theta,0.]
        bounds=[(-8,8)]*5+[(0,10)]*5+[(-4,4),(0,1),(0,10)]
        def mapping(p):
            tau=torch.exp(p[-3]);agreement=torch.exp(-residual/tau)
            weight=torch.softmax(p[:5]+agreement*p[5:10]-p[-1]*torch.log1p(residual**2),dim=1)
            ag=(weight*bounded).sum(1);e=(weight*agreement).sum(1)
            alpha=p[-2]*(1-torch.exp(-torch.abs(ag-B)/tau))*e
            return B+alpha*(ag-B)
    prior=torch.as_tensor(initial,dtype=torch.float64)
    def objective(x):
        p=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        loss=(W*(mapping(p)-Y)**2).sum()+REGULARIZATION*((p-prior)**2).mean()
        loss.backward()
        return float(loss.detach()),p.grad.detach().numpy()
    initial_loss=objective(initial)[0]
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
        options={'maxiter':160,'ftol':1e-10,'gtol':1e-6})
    assert np.isfinite(result.x).all()
    return result.x,dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
        initial_training_loss=initial_loss,final_training_loss=float(result.fun),training_queries=len(Y))

def run(backbone,track):
    dataset=[]
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        dataset.append(dict(id=int(m['data_id']),meta=m,b=(b-sm)/ss,k=((k-sm)/ss).T,y=(y-sm)/ss,
            raw_base=b,raw_bank=k,raw_y=y,sm=sm,ss=ss,theta=t,fold=fold,ref=ref))
    parent=json.loads((OUT/'frozen_plan.json').read_text())
    case=next(c for c in parent['cases'] if (c['backbone'],c['track'],c['task'])==(backbone,track,'regression'))
    rows=[];fits=[]
    for record in case['policies']:
        policy=json.loads((ROOT/record['path']).read_text());theta=np.asarray(policy['parameters'])
        train_ids=set(policy['train_data_ids']);test_ids=set(policy['test_data_ids'])
        assert train_ids.isdisjoint(test_ids)
        train=[c for c in dataset if c['id'] in train_ids];test=[c for c in dataset if c['id'] in test_ids]
        heads={}
        for variant in ['A_learned_trust','C_calibrated_guard']:
            parameters,audit=fit(variant,train,theta,track);heads[variant]=parameters
            head=dict(backbone=backbone,track=track,variant=variant,outer_fold=policy['fold'],parameters=parameters.tolist(),
                source_data_ids=sorted(train_ids),evaluation_data_ids=sorted(test_ids),original_policy_sha256=record['sha256'],
                held_out_labels_used_for_training=False,optimization=audit)
            fits.append(head)
            write_json(DEST/backbone/track/variant/f'fold_{policy["fold"]}.json',head)
            print('FITTED',backbone,track,policy['fold'],variant,audit,flush=True)
        for c in test:
            m,y=c['meta'],c['raw_y'];original=regress(c['raw_base'],c['raw_bank'],theta,c['sm'],c['ss'])[0]
            scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
            assert abs(float(np.sqrt(np.mean((original-y)**2))/scale)-c['ref']['nrmse'])<1e-8*max(1,c['ref']['nrmse'])
            predictions={'Original':original}
            predictions.update({v:numpy_predict(v,c['b'],c['k'],theta,p)*c['ss']+c['sm'] for v,p in heads.items()})
            for method,pred in predictions.items():
                rows.append(dict(backbone=backbone,track=track,data_id=c['id'],family=m['family'],rho=m['rho'],
                    outer_fold=policy['fold'],method=method,n=len(y),
                    r2=1-float(np.sum((pred-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                    nrmse=float(np.sqrt(np.mean((pred-y)**2))/scale)))
    dest=DEST/backbone/track;dest.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(dest/'performance.csv',index=False)
    write_json(dest/'audit.json',dict(original_policy_replay_verified=True,outer_training_evaluation_ids_disjoint=True,
        held_out_labels_used_for_training=False,datasets=len({c['id'] for c in dataset}),episodes=len(dataset),fits=fits))
    print('CALIBRATED_REPAIRS_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--backbone');p.add_argument('--track')
    a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.backbone,a.track)
