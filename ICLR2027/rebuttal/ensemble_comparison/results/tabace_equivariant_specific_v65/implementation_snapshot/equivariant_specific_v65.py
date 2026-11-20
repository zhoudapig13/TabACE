"""Strongly regularized per-task log-linear residuals above frozen Shared.

One coefficient per expert is shared across classes, avoiding the large
class-specific parameter count of v51. Fit using grouped TRAIN-row folds and
core-prior weights. Parent source predictions exclude the entire target
dataset. Native TFM banks and Shared remain unchanged; all queried metrics
are recomputed from one probability array.
"""
import argparse
import json
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
import hierarchical_specific_v64 as prior
from prepare_data import RUN,atomic_json
from run_tabicl import normalize,metrics

OUT=RUN/'results/tabace_equivariant_specific_v65'
PROFILES=[dict(member_prior_power=p,regularization=r) for p in [0.,1.] for r in [.1,1.,10.,100.]]
BLENDS=[.1,.25,.5,.75,1.]


def features(c,anchor,extra,power):
    bank=prior.bind(c,anchor,power)['bank']
    bank=np.concatenate([bank,np.asarray(extra)],axis=0)
    delta=np.log(np.clip(bank,1e-12,1.))-np.log(np.clip(anchor,1e-12,1.))[None]
    delta-=delta.mean(2,keepdims=True)
    return np.clip(delta.transpose(1,0,2),-12,12)


def fit(D,y,anchor,core_prior,regularization):
    counts=np.bincount(y,minlength=anchor.shape[1]).astype(float)+.5
    empirical=counts/counts.sum();weights=core_prior[y]/empirical[y];weights/=weights.sum()
    offset=np.log(np.clip(anchor,1e-12,1.));truth=np.eye(anchor.shape[1])[y]
    def objective(w):
        z=offset+np.einsum('m,nmk->nk',w,D);norm=logsumexp(z,axis=1,keepdims=True)
        p=np.exp(z-norm)
        loss=float(np.sum(weights*(norm[:,0]-z[np.arange(len(y)),y]))+regularization*np.dot(w,w))
        grad=np.einsum('n,nk,nmk->m',weights,p-truth,D)+2*regularization*w
        return loss,grad
    result=minimize(objective,np.zeros(D.shape[1]),jac=True,method='L-BFGS-B',
        options=dict(maxiter=300,ftol=1e-10,gtol=1e-7))
    assert np.isfinite(result.x).all()
    return dict(coefficients=result.x.tolist(),optimization=dict(success=bool(result.success),iterations=int(result.nit),loss=float(result.fun)),
        shared_across_classes=True,regularization=regularization)


def predict(D,anchor,model):
    z=np.log(np.clip(anchor,1e-12,1.))+np.einsum('m,nmk->nk',np.asarray(model['coefficients']),D)
    return normalize(np.exp(z-z.max(1,keepdims=True)))


def source_extras(cells,budget,policy,folds,baseline):
    root=prior.parent.OUT/'policies'/str(budget)/'classification/clean';nested=prior.OUT/'policies'/str(budget)/'classification/clean'
    hierarchy=json.loads((nested/'all_policies_frozen.json').read_text());extra={};old_models={};new_models={}
    for fold in range(4):
        old_models[fold]=json.loads((root/'Specific_original'/f'fold_{fold}.json').read_text())['model']
        if hierarchy['Specific'] is not None:
            index=hierarchy['Specific']['recipe_index'];new_models[fold]=json.loads((nested/'cv'/f'{index}_{fold}.json').read_text())['model']
    for c in cells:
        fold=folds[int(c['meta']['data_id'])];anchor=baseline[c['key']]
        old=prior.parent.apply(c,old_models[fold],policy['Specific'],anchor)
        current=anchor
        if hierarchy['Specific'] is not None:
            choice=hierarchy['Specific'];recipe=prior.RECIPES[choice['recipe_index']]
            raw=prior.parent.gate.predict(prior.bind(c,anchor,recipe['member_prior_power']),new_models[fold])
            a=choice['blend'];current=normalize((1-a)*anchor+a*prior.sharpen(raw,choice['temperature']))
        extra[c['key']]=[old,current]
    return extra


def run(budget):
    assert budget in [1024,4096,8192]
    cells=prior.load('validation',budget);policy,folds,_,baseline=prior.shared_source(cells,budget)
    extras=source_extras(cells,budget,policy,folds,baseline);initial,best_value=prior.parent.prior.score(cells,baseline)
    best=None;trials=[];root=OUT/'policies'/str(budget)/'classification/clean'
    for index,profile in enumerate(PROFILES):
        predictions={}
        for c in cells:
            anchor=baseline[c['key']];D=features(c,anchor,extras[c['key']],profile['member_prior_power'])
            value=np.empty_like(anchor)
            for fold in range(4):
                train=c['folds']!=fold;valid=~train
                if not valid.any():continue
                assert train.any() and not np.intersect1d(c['ids'][train],c['ids'][valid]).size
                model=fit(D[train],c['y'][train],anchor[train],c['metric_core_prior'],profile['regularization'])
                value[valid]=predict(D[valid],anchor[valid],model)
            predictions[c['key']]=value
        for blend in BLENDS:
            combined={c['key']:normalize((1-blend)*baseline[c['key']]+blend*predictions[c['key']]) for c in cells}
            scores,utility=prior.parent.prior.score(cells,combined)
            eligible=scores['accuracy']>=initial['accuracy']-1e-10 and scores['auc']>=initial['auc']-1e-10
            trials.append(dict(profile_index=index,blend=blend,source_OOF_scores=scores,utility=utility,eligible=eligible))
            if eligible and utility>best_value+1e-10:best_value=utility;best=dict(profile_index=index,blend=blend)
        print('EQUIVARIANT_PROFILE_COMPLETE',budget,index,flush=True)
    models={}
    if best is not None:
        profile=PROFILES[best['profile_index']]
        for c in cells:
            anchor=baseline[c['key']];D=features(c,anchor,extras[c['key']],profile['member_prior_power'])
            models[str(c['meta']['task_id'])]=fit(D,c['y'],anchor,c['metric_core_prior'],profile['regularization'])
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Specific=best,models=models,profiles=PROFILES,
        source_baseline=initial,trials=trials,Shared_unchanged_from_v62=True,grouped_original_row_folds=4,
        parent_predictions_exclude_entire_source_dataset=True,source_loss_core_prior_weighted=True,
        clean_TRAIN_calibration_supervision=True,query_labels_used_for_fit_or_selection=False,seed=0))
    rows=[]
    for c in prior.load('query',budget):
        original=prior.parent.OUT/'predictions'/str(budget)/c['key']
        with np.load(original) as z:base=z['Base'];shared=z['Shared'];truth=z['y_target'];old=z['Specific']
        with np.load(prior.OUT/'predictions'/str(budget)/c['key']) as z:new=z['Specific']
        specific=shared
        if best is not None:
            profile=PROFILES[best['profile_index']];D=features(c,shared,[old,new],profile['member_prior_power'])
            p=predict(D,shared,models[str(c['meta']['task_id'])]);a=best['blend'];specific=normalize((1-a)*shared+a*p)
        out=OUT/'predictions'/str(budget)/c['key'];prior.parent.prior.risk.head.source.adaptation.raw.scaling.npwrite(out,Base=base,Shared=shared,Specific=specific,y_target=truth)
        record=json.loads(original.with_suffix('.json').read_text());record.update(adaptation_version='v65',Shared_bitwise_unchanged_from_v62=True,
            local_class_equivariant_loglinear_adapter=True,parameters_per_task=23,query_labels_used_for_fit=False,
            methods={name:metrics(truth,p,'classification',1.) for name,p in [('Base',base),('Shared',shared),('Specific',specific)]})
        atomic_json(out.with_suffix('.json'),record)
        for name,values in record['methods'].items():rows.append(dict(context=budget,block='clean',method=name,task_id=c['meta']['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=72,Shared_unchanged=True))
    print('EQUIVARIANT_SPECIFIC_COMPLETE',budget,flush=True)


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_LOG_LINEAR_FITS',seed=0,refit='Specific Clean1024/4096/8192 only',
        full_goal_checks=48,Shared_frozen_from_v62=True,Clean256_frozen_from_v62=True,profiles=PROFILES,blends=BLENDS,
        features='21 frozen TFM context log-probability contrasts, plus dataset-OOF Specific v62/v64 contrasts, versus complete Shared',
        coefficients_shared_across_classes=True,parameters_per_task=23,intercepts=False,
        source_validation='four grouped original-TRAIN-row folds, parent gates exclude entire target dataset',
        exact_member_priors='hash-verified v63',core_weight_estimation='TRAIN fold only',
        no_additional_TFM_calls=True,clean_TRAIN_calibration_supervision=True,
        query_labels_used_for_fit_or_selection=False,query_feedback_used_for_development_design=True,
        development_only=True,original_paper_replication=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((prior.OUT/'paper_targets.json').read_text())
    for b in [256,1024,4096,8192]:
        for task in ['classification','regression']:
            for block in ['clean','five','seven']:
                if task=='classification' and block=='clean' and b!=256:continue
                origin=prior.parent.OUT if task=='classification' and block=='clean' else RUN/'results/tabace_augmented_classifier_v54' if task=='classification' else RUN/'results/tabace_clean_calibration_v59'
                for name in ['policies','predictions']:
                    dest=OUT/name/str(b)/task/block;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(origin/name/str(b)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);a=p.parse_args();plan() if a.phase=='plan' else run(a.budget)
