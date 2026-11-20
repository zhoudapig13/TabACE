"""Prior-aware Clean fusion using only known TRAIN core/context priors.

The frozen benchmark samples a balanced capped population before its split.
Expanded raw TRAIN calibration has a different class prior. Correct this
source-validation objective without using query labels, and test canonical
context-prior correction as a shared proxy for TFM member posteriors. Missing
calibration-class coverage is recorded, never silently removed from the cohort.
"""
import argparse
import json
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import classification_risk_v58 as risk
import diagnose_class_prior_v60 as diagnostic
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import normalize,metrics

OUT=RUN/'results/tabace_prior_fusion_v61'
POWERS=[0.,.25,.5,.75,1.]
PROFILES=[dict(loss='brier',shrinkage=s,power=p,agreement=1.,confidence=0.) for s in [32.,128.] for p in [1.,2.,4.]]


def load(role,budget):
    cells=risk.head.load(role,budget,'clean');raw=risk.head.source.adaptation.raw
    for c in cells:
        meta=c['meta'];k=c['base'].shape[1];pool=raw.scaling.prepare('classification','five',meta)
        if role=='validation':
            path=raw.OUT/'validation_splits/classification/five'/f'{meta["task_id"]}.npz'
            with np.load(path) as z:positions=z['train_positions'];held=z['validation_original_ids']
            assert not np.intersect1d(pool['source_ids'][:256],held).size
        else:positions=np.arange(len(pool['y']))
        y=pool['y'][positions][:budget]
        assert protocol.array_sha256(y)==meta['source_input_hashes']['y_source']
        core=np.bincount(pool['y'][:256].astype(int),minlength=k).astype(float)+.5
        context=np.bincount(y.astype(int),minlength=k).astype(float)+.5
        c['core_prior']=core/core.sum();c['context_prior']=context/context.sum()
        c['metric_core_prior']=diagnostic.core_prior(meta,k)
    return cells


def correct(c,power):
    ratio=np.power(c['core_prior']/c['context_prior'],power)
    return c|dict(base=normalize(c['base']*ratio),bank=np.asarray([normalize(p*ratio) for p in c['bank']]))


def prepare(cells,m):
    assert all(c['meta']['block']=='clean' for c in cells)
    data=risk.prepare(cells,m)
    for c in cells:
        task=str(c['meta']['task_id']);p=np.concatenate([c['base'][None],c['bank'][:m]],0).transpose(1,0,2)
        y=c['y'].astype(int);k=p.shape[2];loss=(np.square(p).sum(2)-2*p[np.arange(len(y)),:,y]+1)/max(1.-1./k,1e-6)
        values=[]
        for split in range(5):
            keep=c['folds']!=split if split<4 else np.ones(len(y),bool)
            train=y[keep];count=np.bincount(train,minlength=k).astype(float)+.5
            empirical=count/count.sum();weight=c['metric_core_prior'][train]/empirical[train]
            assert np.isfinite(weight).all() and weight.sum()>0
            values.append(np.average(loss[keep],axis=0,weights=weight))
        data['stats'][task]['brier']=np.asarray(values)
    data['priors']['brier']=np.median([v['brier'] for v in data['stats'].values()],axis=0)
    return data


def score(cells,predictions):
    priors={str(c['meta']['task_id']):c['metric_core_prior'] for c in cells}
    values=diagnostic.weighted_score(cells,predictions,priors)
    return values,.5*(values['accuracy']+values['auc'])


def fit(cells,m,baseline,folder):
    initial,best_value=score(cells,baseline);best=None;best_predictions=baseline;trials=[]
    for power in POWERS:
        transformed=[correct(c,power) for c in cells];data=prepare(transformed,m)
        for profile in PROFILES:
            model=risk.policy(data,profile);values=risk.oof(transformed,model)
            for blend in [.25,.5,1.]:
                pred={c['key']:normalize((1-blend)*baseline[c['key']]+blend*values[c['key']]) for c in cells}
                scores,utility=score(cells,pred)
                eligible=scores['accuracy']>=initial['accuracy']-1e-10 and scores['auc']>=initial['auc']-1e-10
                trials.append(dict(prior_power=power,profile=profile,blend=blend,weighted_source_scores=scores,eligible=eligible,utility=utility))
                if eligible and utility>best_value+1e-10:
                    best_value=utility;best=dict(prior_power=power,model=model,blend=blend);best_predictions=pred
    atomic_json(folder/'policy.json',dict(status='FROZEN_BEFORE_QUERY',chosen=best,source_baseline=initial,trials=trials,
        source_prior_estimation='protected TRAIN core256 only',risk_label_frequency_estimation='own TRAIN fold excluded',
        canonical_context_prior_used_as_member_proxy=True,query_labels_used=False))
    return best,best_predictions


def apply(c,policy,anchor):
    if policy is None:return anchor
    p=risk.predict(correct(c,policy['prior_power']),policy['model']);a=policy['blend']
    return normalize((1-a)*anchor+a*p)


def run(budget):
    cells=load('validation',budget);root=OUT/'policies'/str(budget)/'classification/clean'
    sh,source_sh=fit(cells,10,{c['key']:c['base'] for c in cells},root/'Shared')
    sp,_=fit(cells,21,source_sh,root/'Specific_original')
    coverage={str(c['meta']['task_id']):float(c['metric_core_prior'][np.bincount(c['y'],minlength=len(c['metric_core_prior']))>0].sum()) for c in cells}
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Shared=sh,Specific=sp,seed=0,
        source_calibration_class_coverage=coverage,missing_calibration_classes_not_removed=True,
        no_query_labels_in_prior_estimation_or_selection=True,TFM_banks_unchanged=True,extra_TFM_calls=0,
        Shared_members=10,Specific_members=21,Base_additional=True,clean_TRAIN_calibration_supervision=True))
    rows=[]
    for c in load('query',budget):
        meta=c['meta'];shared=apply(c,sh,c['base']);specific=apply(c,sp,shared)
        out=OUT/'predictions'/str(budget)/c['key'];risk.head.source.adaptation.raw.scaling.npwrite(out,Base=c['base'],Shared=shared,Specific=specific,y_target=c['y'])
        record={key:meta[key] for key in ['task','block','task_id','dataset','family','rho','requested_context','actual_context','scale']}
        record.update(input_hashes=meta['source_input_hashes'],source_bank=str(c['path']),role_permutation=c['order'],
            seed=0,provisional_Base_calibration=False,adaptation_version='v61',
            methods={name:metrics(c['y'],p,'classification',1.) for name,p in [('Base',c['base']),('Shared',shared),('Specific',specific)]},
            query_labels_used_for_fit=False,prior_estimation_uses_TRAIN_only=True,context_prior_is_member_proxy=True,
            query_feedback_used_for_development_design=True,checkpoint_finetuned=False,
            development_only=True,original_paper_replication=False,independent_test_claim=False)
        atomic_json(out.with_suffix('.json'),record)
        for name,values in record['methods'].items():rows.append(dict(context=budget,block='clean',method=name,task_id=meta['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=72,canonical_priors_use_TRAIN_only=True))
    print('PRIOR_FUSION_COMPLETE',budget,flush=True)


def plan():
    assert not (OUT/'plan.json').exists();risk.head.source.verify_registry()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_PRIOR_AWARE_FITS',seed=0,budgets=[256,1024,4096,8192],
        refit='classification Clean only',full_goal_checks=48,prior_powers=POWERS,profiles=PROFILES,
        motivation=str(diagnostic.OUT/'diagnosis.json'),
        target_prior_proxy='original protected TRAIN core256, Jeffreys smoothing0.5',
        native_prior='actual canonical TRAIN context labels, Jeffreys smoothing0.5',
        member_prior_proxy='same canonical prior for all existing TFM member posteriors, explicitly approximate',
        fitting_objective='source metrics importance-weighted toward TRAIN core prior; own fold excluded from training risk weights',
        query_labels_used_for_prior_estimation_fit_or_selection=False,query_tasks_and_Base_unchanged=True,
        missing_calibration_class_coverage='record all72 datasets; retain all rows and cohorts',
        extra_TFM_calls=0,Shared_members=10,Specific_members=21,Specific_complete_Shared_fallback=True,
        query_feedback_used_for_development_design=True,development_only=True,original_paper_replication=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((risk.OUT/'paper_targets.json').read_text())
    for b in [256,1024,4096,8192]:
        for task,blocks,parent in [('classification',['five','seven'],RUN/'results/tabace_augmented_classifier_v54'),('regression',['clean','five','seven'],RUN/'results/tabace_clean_calibration_v59')]:
            for block in blocks:
                for name in ['policies','predictions']:
                    dest=OUT/name/str(b)/task/block;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(parent/name/str(b)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);a=p.parse_args();plan() if a.phase=='plan' else run(a.budget)
