"""Grouped-TRAIN reliability weighting of the existing classifier TFM banks.

This candidate changes only fusion: no auxiliary feature classifier, new TFM
calls, backbone training, or query-label parameter selection. Source IDs are
grouped across all perturbations and prior resamples, using the checked v51
mapping. Specific preserves Shared as a complete fallback. Query outcomes are
reported even when a source-selected policy fails the requested inequalities.
"""
import argparse
import json
import os
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import probability_head_v51 as head
from prepare_data import RUN,atomic_json
from run_tabicl import normalize,metrics

OUT=RUN/'results/tabace_classification_risk_v58'
PROFILES=[dict(loss=loss,shrinkage=shrink,power=power,agreement=agree,confidence=conf)
          for loss in ['brier','nll'] for shrink in [32.,128.] for power in [1.,2.,4.]
          for agree in [0.,1.] for conf in [0.,.5]]
BLENDS=[.25,.5,1.]


def prepare(cells,m):
    groups={}
    for c in cells:groups.setdefault(str(c['meta']['task_id']),[]).append(c)
    stats={};counts={};trials={'brier':[],'nll':[]}
    for task,group in groups.items():
        folds=np.concatenate([c['folds'] for c in group]);ids=np.concatenate([c['ids'] for c in group])
        losses={'brier':[],'nll':[]};weights=[]
        for c in group:
            P=np.concatenate([c['base'][None],c['bank'][:m]],axis=0).transpose(1,0,2)
            truth=c['y'].astype(int);classes=P.shape[2]
            truep=P[np.arange(len(truth)),:,truth]
            losses['brier'].append((np.square(P).sum(2)-2*truep+1)/max(1.-1./classes,1e-6))
            losses['nll'].append(-np.log(np.clip(truep,1e-12,1.))/np.log(max(2,classes)))
            weights.append(np.full(len(truth),1./(len(group)*len(truth))))
        weights=np.concatenate(weights);stats[task]={};counts[task]=[]
        for split in range(5):
            mask=folds!=split if split<4 else np.ones(len(folds),bool)
            assert mask.any();counts[task].append(int(len(np.unique(ids[mask]))))
        for loss in losses:
            values=np.concatenate(losses[loss]);risks=[]
            for split in range(5):
                mask=folds!=split if split<4 else np.ones(len(folds),bool)
                risks.append(np.average(values[mask],axis=0,weights=weights[mask]))
            stats[task][loss]=np.asarray(risks);trials[loss].append(risks)
    priors={loss:np.median(np.asarray(values),axis=0) for loss,values in trials.items()}
    return dict(stats=stats,counts=counts,priors=priors,m=m)


def predict(c,model,split=4):
    m=model['m'];profile=model['profile'];task=str(c['meta']['task_id'])
    p=np.concatenate([c['base'][None],c['bank'][:m]],axis=0).transpose(1,0,2)
    n,_,classes=p.shape;stats=model['statistics'][task]
    risk=np.asarray(stats['risk'])[split];prior=np.asarray(model['prior'])[split]
    count=stats['count'][split];a=count/(count+profile['shrinkage'])
    mixed=a*risk+(1-a)*prior;logw=-profile['power']*np.log(mixed+.02)
    logw=np.broadcast_to(logw,(n,m+1)).copy()
    logw-=profile['confidence']*(-np.sum(p*np.log(np.clip(p,1e-12,1.)),axis=2))/np.log(max(2,classes))
    if profile['agreement']:
        initial=np.exp(logw-logw.max(1,keepdims=True));initial/=initial.sum(1,keepdims=True)
        center=(initial[:,:,None]*p).sum(1)
        logw-=profile['agreement']*.5*np.abs(p-center[:,None]).sum(2)
    weight=np.exp(logw-logw.max(1,keepdims=True));weight/=weight.sum(1,keepdims=True)
    return normalize((weight[:,:,None]*p).sum(1))


def policy(data,profile):
    return dict(m=data['m'],profile=profile,prior=data['priors'][profile['loss']].tolist(),
        statistics={task:dict(risk=values[profile['loss']].tolist(),count=data['counts'][task]) for task,values in data['stats'].items()})


def oof(cells,model):
    result={}
    for c in cells:
        value=np.empty_like(c['base'],dtype=np.float64)
        for fold in range(4):
            mask=c['folds']==fold
            if mask.any():value[mask]=predict(c,model,fold)[mask]
        result[c['key']]=normalize(value)
    return result


def fit(cells,m,baseline,folder):
    data=prepare(cells,m);initial,best_value=head.score(cells,baseline);best=None;best_pred=baseline;trials=[]
    for profile in PROFILES:
        model=policy(data,profile);values=oof(cells,model)
        for blend in BLENDS:
            pred={c['key']:normalize((1-blend)*baseline[c['key']]+blend*values[c['key']]) for c in cells}
            scores,utility=head.score(cells,pred)
            eligible=scores['accuracy']>=initial['accuracy']-1e-10 and (scores['auc'] is None or scores['auc']>=initial['auc']-1e-10)
            trials.append(dict(profile=profile,blend=blend,source_OOF_scores=scores,utility=utility,eligible=eligible))
            if eligible and utility>best_value+1e-10:
                best_value=utility;best=dict(model=model,blend=blend);best_pred=pred
    atomic_json(folder/'policy.json',dict(status='FROZEN_BEFORE_QUERY',chosen=best,source_baseline=initial,trials=trials,
        original_row_group_folds=4,own_fold_excluded_from_risks_and_prior=True,query_labels_used=False,seed=0))
    return best,best_pred


def apply(c,model,anchor):
    if model is None:return anchor
    value=predict(c,model['model']);a=model['blend'];return normalize((1-a)*anchor+a*value)


def run(budget,block):
    head.source.verify_registry();cells=head.load('validation',budget,block)
    root=OUT/'policies'/str(budget)/'classification'/block
    shared,sh_oof=fit(cells,10,{c['key']:c['base'] for c in cells},root/'Shared')
    specific={}
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family]
        specific[family],_=fit(group,21,sh_oof,root/f'Specific_{family}')
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY_BANKS_LOADED',Shared=shared,Specific=specific,
        Shared_family_blind=True,Specific_family_visible=True,actual_strength_input=False,original_row_group_folds=4,
        extra_auxiliary_feature_models=False,TFM_banks_unchanged=True,Shared_members=10,Specific_members=21,Base_additional=True,
        clean_TRAIN_calibration_supervision=True,checkpoint_finetuned=False,original_query_labels_used=False,seed=0))
    rows=[]
    for c in head.load('query',budget,block):
        m=c['meta'];sh=apply(c,shared,c['base']);sp=apply(c,specific[m['family']],sh)
        out=OUT/'predictions'/str(budget)/c['key'];head.source.adaptation.raw.scaling.npwrite(out,Base=c['base'],Shared=sh,Specific=sp,y_target=c['y'])
        record={key:m[key] for key in ['task','block','task_id','dataset','family','rho','requested_context','actual_context','scale']}
        record.update(input_hashes=m['source_input_hashes'],source_bank=str(c['path']),seed=0,provisional_Base_calibration=False,
            role_permutation=c['order'],adaptation_version='v58',clean_TRAIN_calibration_supervision=True,
            methods={name:metrics(c['y'],p,'classification',1.) for name,p in [('Base',c['base']),('Shared',sh),('Specific',sp)]},
            query_labels_used_for_fit=False,query_feedback_used_for_development_design=True,
            Specific_family_visible=True,actual_strength_input=False,checkpoint_finetuned=False,
            extra_auxiliary_feature_models=False,development_only=True,original_paper_replication=False,independent_test_claim=False)
        atomic_json(out.with_suffix('.json'),record)
        for name,values in record['methods'].items():rows.append(dict(context=budget,block=block,method=name,task_id=m['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=len(rows)//3,source_calibrated=True))
    print('CLASSIFICATION_RISK_COMPLETE',budget,block,len(rows)//3,flush=True)


def plan():
    assert not (OUT/'plan.json').exists();head.source.verify_registry()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_NEW_FUSION_FITS',seed=0,budgets=[256,1024,4096,8192],
        refit=['classification clean','classification seven'],full_goal_checks=48,profiles=PROFILES,blends=BLENDS,
        source_banks=str(head.source.OUT),Shared_members=10,Specific_members=21,Base_additional=True,
        all_TFM_banks_unchanged=True,extra_TFM_calls=0,auxiliary_feature_models=0,
        calibration='four grouped original-TRAIN-ID folds, same fold across all conditions and label-prior resamples',
        loss='condition-normalized member Brier/NLL, dataset shrinkage and confidence/agreement weighting',
        Specific_complete_Shared_fallback=True,Shared_family_blind=True,Specific_family_visible=True,actual_strength_input=False,
        query_labels_used_for_fit_or_hyperparameters=False,query_feedback_used_for_development_design=True,
        development_only=True,original_paper_replication=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((head.OUT/'paper_targets.json').read_text())
    for b in [256,1024,4096,8192]:
        for task,blocks,parent in [('classification',['five'],head.OUT),('regression',['clean','five','seven'],RUN/'results/tabace_bagged_specific_v56')]:
            for block in blocks:
                for name in ['policies','predictions']:
                    dest=OUT/name/str(b)/task/block;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(parent/name/str(b)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);p.add_argument('--block',choices=['clean','seven']);a=p.parse_args()
    plan() if a.phase=='plan' else run(a.budget,a.block)
