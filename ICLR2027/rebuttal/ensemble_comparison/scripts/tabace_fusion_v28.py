"""Train-only calibrated confidence/agreement/Base-gate fusion for v28 banks."""
import argparse
import json
import os
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
import tabace_context_reproduction_v28 as main
from prepare_data import atomic_json
from run_tabicl import normalize,metrics
GAINS=[1.,2.,4.]
GATE_VARIANT='entropy'


def load(role,budget,task,block):
    cells=[]
    dataset_ids={int(m['task_id']):int(m['data_id']) for m in main.frozen.cohort(task,block)}
    root=main.OUT/'banks'/role/str(budget)/task/block
    for path in sorted(root.glob('*/*.npz')):
        if '.tmp.' in path.name:continue
        meta=json.loads(path.with_suffix('.json').read_text())
        meta['data_id']=dataset_ids[int(meta['task_id'])]
        with np.load(path) as z:
            c=dict(base=z['base'],bank=z['members'],y=z['y_target'],mu=float(z['source_mean']),sigma=float(z['source_scale']))
        cells.append(c|dict(meta=meta,task=task,role=role,path=path))
    expected=len(main.frozen.cohort(task,block))*len(main.case_grid(block))
    assert len(cells)==expected,(role,budget,task,block,len(cells),expected)
    return cells


def features(c,m,gate_variant='entropy'):
    bank=c['bank'][:m];base=c['base']
    if c['task']=='classification':
        p=np.clip(bank,1e-12,1.);K=base.shape[1]
        Q=(1+np.sum(p*np.log(p),axis=2)/np.log(max(2,K))).T
        S=(1-.5*np.sum(np.abs(bank-bank.mean(0)[None]),axis=2)).T
        q=np.clip(base,1e-12,1.);U=-np.sum(q*np.log(q),axis=1)/np.log(max(2,K))
        if gate_variant=='entropy_or_disagreement':
            U=np.maximum(U,.5*np.sum(np.abs(bank-base[None]),axis=2).max(0))
        return dict(Q=Q,S=S,U=U)
    M=(bank-c['mu'])/c['sigma'];B=(base-c['mu'])/c['sigma']
    return dict(M=M.T,B=B,D=np.abs(M-np.median(M,axis=0)[None]).T)


def training_tensors(cells,m):
    assert all(c['role']=='validation' for c in cells)
    dataset_counts={}
    for c in cells:dataset_counts[c['meta']['task_id']]=dataset_counts.get(c['meta']['task_id'],0)+1
    values={};weights=[]
    for c in cells:
        f=features(c,m,GATE_VARIANT)
        if c['task']=='classification':
            truth=c['y'].astype(int);f['true_members']=c['bank'][:m,np.arange(len(truth)),truth].T
            f['true_base']=c['base'][np.arange(len(truth)),truth]
            factor=1.
        else:
            f['Y']=(c['y']-c['mu'])/c['sigma'];factor=(c['sigma']/c['meta']['scale'])**2
        for k,v in f.items():values.setdefault(k,[]).append(v)
        weights.append(np.full(len(c['y']),factor/(len(dataset_counts)*dataset_counts[c['meta']['task_id']]*len(c['y']))))
    return {k:torch.as_tensor(np.concatenate(v),dtype=torch.float64) for k,v in values.items()}|dict(weights=torch.as_tensor(np.concatenate(weights),dtype=torch.float64))


def mapping(p,data,task,m,gain):
    if task=='classification':
        W=torch.softmax(p[:m]+data['Q']*p[m:2*m]+data['S']*p[2*m:3*m],dim=1)
        mixed=(W*data['true_members']).sum(1);support=(W*data['Q']*data['S']).sum(1)
        alpha=(gain*p[-1]*data['U']*support).clamp(0,1)
        return (1-alpha)*data['true_base']+alpha*mixed
    tau=torch.exp(p[-2]);agreement=torch.exp(-data['D']/tau)
    W=torch.softmax(p[:m]+agreement*p[m:2*m],dim=1);mixed=(W*data['M']).sum(1)
    alpha=(gain*p[-1]*(1-torch.exp(-torch.abs(mixed-data['B'])/tau))*(W*agreement).sum(1)).clamp(0,1)
    return (1-alpha)*data['B']+alpha*mixed


def fit(cells,task,m,gain,reg):
    data=training_tensors(cells,m)
    initial=np.r_[np.zeros(m),np.ones(2*m),.5] if task=='classification' else np.r_[np.zeros(m),np.ones(m),0.,.5]
    bounds=[(-8,8)]*m+([(0,10)]*(2*m)+[(0,1)] if task=='classification' else [(0,10)]*m+[(-4,4),(0,1)])
    def objective(x):
        p=torch.tensor(x,dtype=torch.float64,requires_grad=True);prediction=mapping(p,data,task,m,gain)
        raw=-torch.log(prediction.clamp(1e-12,1.)) if task=='classification' else (prediction-data['Y'])**2
        scores=p[m:3*m] if task=='classification' else p[m:2*m]
        loss=(raw*data['weights']).sum()+reg*((p[:m]-p[:m].mean()).square().mean()+(scores-1).square().mean())
        loss.backward();return float(loss.detach()),p.grad.detach().numpy()
    chunks=[];current=initial
    for budget in [80,200]:
        result=minimize(objective,current,method='L-BFGS-B',jac=True,bounds=bounds,options=dict(maxiter=budget,ftol=1e-9,gtol=1e-6))
        chunks.append(dict(iterations=int(result.nit),success=bool(result.success),loss=float(result.fun)))
        if result.success:break
        current=result.x
    assert np.isfinite(result.x).all()
    return dict(kind='gated_fusion',m=m,gain=gain,regularization=reg,parameters=result.x.tolist(),optimization=chunks,gate_variant=GATE_VARIANT)


def predict(c,policy):
    if policy['kind']=='base':return c['base']
    p=np.asarray(policy['parameters']);m=policy['m'];gain=policy['gain'];f=features(c,m,policy.get('gate_variant','entropy'));bank=c['bank'][:m]
    if c['task']=='classification':
        z=p[:m]+f['Q']*p[m:2*m]+f['S']*p[2*m:3*m];W=np.exp(z-z.max(1,keepdims=True));W/=W.sum(1,keepdims=True)
        mixed=np.einsum('nm,mnc->nc',W,bank)
        alpha=np.clip(gain*p[-1]*f['U']*(W*f['Q']*f['S']).sum(1),0,1)
        return normalize((1-alpha[:,None])*c['base']+alpha[:,None]*mixed)
    tau=np.exp(p[-2]);agreement=np.exp(-f['D']/tau);z=p[:m]+agreement*p[m:2*m]
    W=np.exp(z-z.max(1,keepdims=True));W/=W.sum(1,keepdims=True);mixed=(W*f['M']).sum(1)
    alpha=np.clip(gain*p[-1]*(1-np.exp(-np.abs(mixed-f['B'])/tau))*(W*agreement).sum(1),0,1)
    return ((1-alpha)*f['B']+alpha*mixed)*c['sigma']+c['mu']


def evaluate(cells,policy):
    values=[]
    for c in cells:values.append(metrics(c['y'],predict(c,policy),c['task'],c['meta']['scale']))
    keys=['accuracy','auc'] if cells[0]['task']=='classification' else ['r2','nrmse']
    avg={}
    for k in keys:
        defined=[r[k] for r in values if r[k] is not None]
        avg[k]=float(np.mean(defined)) if defined else None
        avg[k+'_defined_conditions']=len(defined)
    if cells[0]['task']=='classification':
        utility=avg['accuracy'] if avg['auc'] is None else .5*(avg['accuracy']+avg['auc'])
    else:utility=.5*(avg['r2']-avg['nrmse'])
    return avg,utility


def inner_ids(cells):
    ids=sorted({int(c['meta']['data_id']) for c in cells});return set(ids[::5])


def select(cells,task,m,fallback):
    valid_ids=inner_ids(cells)
    train=[c for c in cells if c['meta']['data_id'] not in valid_ids];validation=[c for c in cells if c['meta']['data_id'] in valid_ids]
    assert train and validation
    fallback_scores,fallback_value=evaluate(validation,fallback)
    best=(fallback_value,None,None,fallback);trials=[dict(kind='fallback',validation_scores=fallback_scores,validation_utility=fallback_value)]
    for gain in GAINS:
        for reg in [.0001,.01]:
            model=fit(train,task,m,gain,reg);scores,value=evaluate(validation,model)
            trials.append(dict(gain=gain,regularization=reg,validation_scores=scores,validation_utility=value,optimization=model['optimization']))
            if value>best[0]+1e-10:best=(value,gain,reg,model)
    _,gain,reg,inner_model=best
    final_model=fallback if gain is None else fit(cells,task,m,gain,reg)
    return final_model,inner_model,trials,sorted(valid_ids)


def run(budget,task,block,shared_members=5):
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    calibration=load('validation',budget,task,block)
    root=main.OUT/'policies'/str(budget)/task/block;shared_file=root/'Shared.json'
    shared_model,shared_inner,trials,valid_ids=select(calibration,task,shared_members,dict(kind='base'))
    atomic_json(shared_file,dict(status='FROZEN_BEFORE_QUERY_EVALUATION',method=f'Shared_R{shared_members}_reconstructed',model=shared_model,
        inner_model=shared_inner,trials=trials,inner_validation_data_ids=valid_ids,
        training_task_ids=sorted({c['meta']['task_id'] for c in calibration}),
        held_out_TRAIN_labels_used=True,clean_TRAIN_calibration_supervision=True,original_query_labels_used=False,seed=0,Shared_family_blind=True))
    specific_models={};family_groups=sorted({c['meta']['family'] for c in calibration})
    for family in family_groups:
        group=[c for c in calibration if c['meta']['family']==family]
        # Shared's inner-fit model excludes the same held-out dataset IDs and
        # therefore provides a label-safe lower-tier candidate for selection.
        specific_members=group[0]['bank'].shape[0]
        assert all(c['bank'].shape[0]==specific_members for c in group)
        model,_,attempts,inner=select(group,task,specific_members,shared_inner)
        if model is shared_inner:model=shared_model
        specific_models[family]=model
        atomic_json(root/f'Specific_{family}.json',dict(status='FROZEN_BEFORE_QUERY_EVALUATION',
            method=f'Specific_Shared_union_R{specific_members}_reconstructed',model=model,trials=attempts,
            inner_validation_data_ids=inner,training_task_ids=sorted({c['meta']['task_id'] for c in group}),
            Shared_candidate_bank_preserved=True,Shared_policy_fallback_available=True,
            clean_TRAIN_calibration_supervision=True,
            family_visible=True,strength_used_for_fit_features=False,original_query_labels_used=False,seed=0))
    # Final query labels are opened only after all parameters above are frozen.
    queries=load('query',budget,task,block);rows=[]
    for c in queries:
        predictions=dict(Base=c['base'],Shared=predict(c,shared_model),Specific=predict(c,specific_models[c['meta']['family']]))
        out=main.OUT/'predictions'/str(budget)/task/block/str(c['meta']['task_id'])/c['path'].name
        main.scaling.npwrite(out,**predictions,y_target=c['y'])
        summary=dict(task=task,block=block,task_id=c['meta']['task_id'],dataset=c['meta']['dataset'],family=c['meta']['family'],rho=c['meta']['rho'],
            requested_context=budget,actual_context=c['meta']['actual_context'],scale=c['meta']['scale'],
            input_hashes=c['meta']['source_input_hashes'],seed=0,parameters_fit_only_on_held_out_TRAIN=True,
            clean_TRAIN_calibration_supervision=True,Base_uses_clean_calibration_labels=False,
            provisional_Base_calibration=c['meta']['provisional_Base_calibration'],development_only=True,
            methods={name:metrics(c['y'],p,task,c['meta']['scale']) for name,p in predictions.items()})
        atomic_json(out.with_suffix('.json'),summary)
        for name,values in summary['methods'].items():rows.append({k:x for k,x in summary.items() if k not in ['methods','input_hashes']}|dict(method=name)|values)
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=len(queries),methods=['Base','Shared','Specific'],
        parameter_selection_uses_query_labels=False,all_fusion_policies_source_calibrated=True))
    print('FUSION_COMPLETE',budget,task,block,len(queries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);p.add_argument('--task',choices=['classification','regression'],required=True);p.add_argument('--block',choices=['clean','five','seven'],required=True);p.add_argument('--shared-members',type=int,default=5);a=p.parse_args()
    run(a.budget,a.task,a.block,a.shared_members)
