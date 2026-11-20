"""Train-calibrated reliability adaptation; reuse immutable v29 TFM banks.

No additional TFM calls or query-label fitting. Clean's generic transform scale
is selected using inner TRAIN dataset IDs. Classification calibrates member
temperatures. Regression measures agreement about a learned reliable center,
so a majority of weak branches cannot automatically suppress a useful repair.
"""
import argparse
import json
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import torch
from scipy.special import logsumexp
import tabace_fusion_v28 as fusion
import tabace_context_adaptation_v29 as banks
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabace_reliability_fusion_v33'
TEMPERATURES=[.7,1.,1.4,2.]
_LOAD=fusion.load
_FIT=fusion.fit
_PREDICT=fusion.predict
_MAPPING=fusion.mapping
_TEMPERATURE_CACHE={}
_PERMUTATIONS={}


def plan():
    path=OUT/'plan.json';assert not path.exists()
    atomic_json(path,dict(status='DECLARED_BEFORE_NEW_POLICY_FITS',seed=0,source_banks=str(banks.OUT),
        context_budgets=[256,1024,4096,8192],shared_members=10,specific_total_members=21,
        no_additional_TFM_calls=True,no_query_label_parameter_fit=True,
        clean_generic_scale_candidates=dict(classification_indices=[9,15,16],regression_indices=[9,19,20]),
        scale_selection='inner-training dataset IDs only; inner held-out dataset IDs remain untouched',
        classification_temperature_grid=TEMPERATURES,
        regression_agreement='distance to learned prior-weighted center rather than all-member median',
        Shared_and_Specific_use_same_calibration_information=True,Specific_inherits_all_Shared=True,
        adaptation_marker='dagger',original_paper_replication=False,development_only=True,
        prior_query_results_have_been_viewed=True,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((banks.OUT/'paper_targets.json').read_text())


def load(role,budget,task,block):
    previous=fusion.main.OUT
    try:
        fusion.main.OUT=banks.OUT
        cells=_LOAD(role,budget,task,block)
    finally:fusion.main.OUT=previous
    key=(budget,task,block)
    if block=='clean' and key not in _PERMUTATIONS:
        assert role=='validation','Fit the source-only role choice before loading queries'
        held=fusion.inner_ids(cells)
        train=[c for c in cells if c['meta']['data_id'] not in held]
        candidates=[9,15,16] if task=='classification' else [9,19,20]
        trials=[]
        for index in candidates:
            values=[fusion.metrics(c['y'],c['bank'][index],task,c['meta']['scale']) for c in train]
            names=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
            means={name:float(np.mean([v[name] for v in values if v[name] is not None])) for name in names}
            utility=.5*(means['accuracy']+means['auc']) if task=='classification' else .5*(means['r2']-means['nrmse'])
            trials.append(dict(index=index,source_inner_training_metrics=means,utility=utility))
        chosen=max(trials,key=lambda r:r['utility'])['index'];order=list(range(21));order[9],order[chosen]=order[chosen],order[9]
        _PERMUTATIONS[key]=order
        atomic_json(OUT/'policies'/str(budget)/task/block/'generic_scale_selection.json',
            dict(selected_original_index=chosen,permutation=order,trials=trials,
                 inner_heldout_dataset_ids=sorted(held),query_labels_used=False))
    order=_PERMUTATIONS.get(key,list(range(21)))
    for c in cells:
        assert c['bank'].shape[0]==21,'Unexpected candidate registry; do not truncate it'
        c['bank']=c['bank'][order]
        names=c['meta']['shared_roles']+c['meta']['specific_additional_roles']
        names=[names[j] for j in order]
        c['meta']=c['meta']|dict(shared_roles=names[:10],specific_additional_roles=names[10:],
                                  reliability_adaptation='v33',role_permutation=order)
    return cells


def temperatures(cells,m):
    key=(m,tuple(str(c['path']) for c in cells))
    if key in _TEMPERATURE_CACHE:return _TEMPERATURE_CACHE[key]
    counts={}
    for c in cells:counts[c['meta']['task_id']]=counts.get(c['meta']['task_id'],0)+1
    losses=np.zeros((len(TEMPERATURES),m))
    for c in cells:
        assert c['role']=='validation'
        logs=np.log(np.clip(c['bank'][:m],1e-12,1.));truth=c['y'].astype(int)
        for i,t in enumerate(TEMPERATURES):
            z=logs/t
            nll=logsumexp(z,axis=2)-z[:,np.arange(len(truth)),truth]
            losses[i]+=nll.mean(1)/(len(counts)*counts[c['meta']['task_id']])
    result=np.asarray(TEMPERATURES)[losses.argmin(0)].tolist();_TEMPERATURE_CACHE[key]=result
    return result


def calibrated(c,values):
    bank=c['bank'].copy();m=len(values)
    logp=np.log(np.clip(bank[:m],1e-12,1.))/np.asarray(values)[:,None,None]
    bank[:m]=np.exp(logp-logsumexp(logp,axis=2,keepdims=True))
    return c|dict(bank=bank)


def mapping(p,data,task,m,gain):
    if task=='classification':return _MAPPING(p,data,task,m,gain)
    tau=torch.exp(p[-2]);prior=torch.softmax(p[:m],dim=0)
    center=(data['M']*prior).sum(1)
    agreement=torch.exp(-torch.abs(data['M']-center[:,None])/tau)
    weight=torch.softmax(p[:m]+agreement*p[m:2*m],dim=1)
    mixed=(weight*data['M']).sum(1)
    alpha=(gain*p[-1]*(1-torch.exp(-torch.abs(mixed-data['B'])/tau))*(weight*agreement).sum(1)).clamp(0,1)
    return (1-alpha)*data['B']+alpha*mixed


def fit(cells,task,m,gain,reg):
    if task=='classification':
        values=temperatures(cells,m);prepared=[calibrated(c,values) for c in cells]
        model=_FIT(prepared,task,m,gain,reg);model['member_temperatures']=values
    else:
        model=_FIT(cells,task,m,gain,reg);model['agreement_reference']='learned_prior_center'
    model['reliability_adaptation']='v33';return model


def predict(c,model):
    if model['kind']=='base':return c['base']
    if c['task']=='classification':
        return _PREDICT(calibrated(c,model['member_temperatures']),model)
    if model.get('agreement_reference')!='learned_prior_center':return _PREDICT(c,model)
    p=np.asarray(model['parameters']);m=model['m'];gain=model['gain'];f=fusion.features(c,m)
    prior=np.exp(p[:m]-p[:m].max());prior/=prior.sum();tau=np.exp(p[-2])
    center=(f['M']*prior).sum(1);agreement=np.exp(-np.abs(f['M']-center[:,None])/tau)
    z=p[:m]+agreement*p[m:2*m];w=np.exp(z-z.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    mixed=(w*f['M']).sum(1)
    alpha=np.clip(gain*p[-1]*(1-np.exp(-np.abs(mixed-f['B'])/tau))*(w*agreement).sum(1),0,1)
    return ((1-alpha)*f['B']+alpha*mixed)*c['sigma']+c['mu']


def configure():
    fusion.main.OUT=OUT;fusion.GAINS=[1.,2.,4.,8.];fusion.GATE_VARIANT='entropy_or_disagreement'
    fusion.load=load;fusion.fit=fit;fusion.predict=predict;fusion.mapping=mapping


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);p.add_argument('--task',choices=['classification','regression']);p.add_argument('--block',choices=['clean','five','seven']);a=p.parse_args()
    if a.phase=='plan':plan()
    else:configure();fusion.run(a.budget,a.task,a.block,shared_members=10)
