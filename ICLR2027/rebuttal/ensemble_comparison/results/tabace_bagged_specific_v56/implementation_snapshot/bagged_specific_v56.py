"""Expand Specific with TRAIN-only bags; preserve the complete v53 Shared.

Each auxiliary learner sees a distinct subset of at most the requested row
budget. Different members may cover additional authorized clean TRAIN rows.
The run seed is zero; member sampling streams are derived deterministically.
Extra supervision, extra learners, and family-visible source selection are
explicit. No query label or score selects any ensemble recipe.
"""
import argparse
import json
import os
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesRegressor
import source_expert_v53 as parent
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabace_bagged_specific_v56'
PROFILES=[dict(kind='ridge',alpha=1.),dict(kind='lasso',alpha=.001),dict(kind='lasso',alpha=.005),
          dict(kind='catboost',iterations=256,depth=6),dict(kind='extra_trees',trees=128,leaf=4,max_depth=12)]
MEMBERS=8


def subset(pool,partition,budget,member,tid,role):
    positions=np.arange(len(pool['y']))
    if role=='validation':positions=positions[~np.isin(pool['source_ids'],partition['validation_original_ids'])]
    n=min(budget,max(16,int(.8*len(positions))),len(positions))
    chosen=np.random.default_rng(np.random.SeedSequence([0,int(tid),5600,member])).permutation(positions)[:n]
    ids=pool['source_ids'][chosen];assert len(np.unique(ids))==len(ids)<=budget
    assert not np.intersect1d(ids,pool['query_ids']).size
    if role=='validation':assert not np.intersect1d(ids,partition['validation_original_ids']).size
    return chosen


def fit(X,y,profile):
    if profile['kind'] not in ['lasso','extra_trees']:return parent.fit(X,y,profile)
    mean=X.mean(0);scale=X.std(0);scale=np.where(scale>1e-8,scale,1.)
    Z=np.clip((X-mean)/scale,-10,10);mu=float(y.mean());sigma=max(float(y.std()),1e-8)
    if profile['kind']=='lasso':model=Lasso(alpha=profile['alpha'],max_iter=30000,tol=1e-5)
    else:model=ExtraTreesRegressor(n_estimators=profile['trees'],min_samples_leaf=profile['leaf'],max_depth=profile['max_depth'],random_state=0,n_jobs=2)
    model.fit(Z,(y-mu)/sigma)
    return dict(center=mean,scale=scale,mu=mu,sigma=sigma,estimator=model)


def make_banks(budget,block):
    for meta in parent.h.source.adaptation.raw.frozen.cohort('regression',block):
        tid=str(meta['task_id']);out=OUT/'experts'/str(budget)/block/f'{tid}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        pool,cal,_=parent.source_data(meta,block,budget);predictions=[];records=[]
        for i,profile in enumerate(PROFILES):
            values=[]
            for member in range(MEMBERS):
                models=out.parent/'models'/tid;models.mkdir(parents=True,exist_ok=True)
                source_pos=subset(pool,cal,budget,member,tid,'validation')
                model=fit(pool['X'][source_pos],pool['y'][source_pos],profile)
                values.append(parent.h.head_predict(cal['X_validation'],model))
                joblib.dump(model,models/f'validation_{i}_{member}.joblib',compress=3)
                final_pos=subset(pool,cal,budget,member,tid,'query')
                final=fit(pool['X'][final_pos],pool['y'][final_pos],profile)
                path=models/f'query_{i}_{member}.joblib';joblib.dump(final,path,compress=3)
                records.append(dict(profile_index=i,member=member,source_rows=len(source_pos),query_rows=len(final_pos),
                    source_original_ids=pool['source_ids'][source_pos].tolist(),query_original_ids=pool['source_ids'][final_pos].tolist(),
                    source_X_sha256=protocol.array_sha256(pool['X'][source_pos]),source_y_sha256=protocol.array_sha256(pool['y'][source_pos]),
                    query_model_path=str(path),calibration_overlap=0))
            predictions.append(values)
        parent.h.source.adaptation.raw.scaling.npwrite(out,predictions=np.asarray(predictions),y_validation=cal['y_validation'])
        atomic_json(out.with_suffix('.json'),dict(status='BAG_BANK_COMPLETE',budget=budget,task_id=meta['task_id'],records=records,
            profiles=PROFILES,members=MEMBERS,seed=0,extra_clean_TRAIN_supervision=True,query_labels_used=False))
        print('SPECIFIC_BAGS_COMPLETE',budget,block,tid,flush=True)


def parent_source(cells,budget,block):
    experts={}
    for c in cells:
        tid=str(c['meta']['task_id'])
        if tid not in experts:
            with np.load(parent.OUT/'experts'/str(budget)/block/f'{tid}.npz') as z:experts[tid]=z['predictions']
    saved=json.loads((parent.OUT/'policies'/str(budget)/'regression'/block/'all_policies_frozen.json').read_text())
    sh,sh_oof=parent.select(cells,{c['key']:c['base'] for c in cells},experts,range(4))
    assert sh['chosen']==saved['Shared']['chosen'];specific={}
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family]
        policy,pred=parent.select(group,sh_oof,experts,range(len(parent.PROFILES)))
        assert policy['chosen']==saved['Specific'][family]['chosen'];specific.update(pred)
    return specific


def select(cells,baseline,banks):
    initial,best_value=parent.score(cells,baseline);best=None;trials=[]
    for index in range(len(PROFILES)):
        for members in [4,8]:
            for blend in parent.BLENDS:
                pred={c['key']:(1-blend)*baseline[c['key']]+blend*banks[str(c['meta']['task_id'])][index,:members].mean(0) for c in cells}
                scores,utility=parent.score(cells,pred)
                eligible=scores['r2']>=initial['r2']-1e-10 and scores['nrmse']<=initial['nrmse']+1e-10
                trials.append(dict(profile_index=index,members=members,blend=blend,source_scores=scores,utility=utility,eligible=eligible))
                if eligible and utility>best_value+1e-10:best_value=utility;best=dict(profile_index=index,members=members,blend=blend)
    return dict(chosen=best,source_baseline=initial,trials=trials)


def run(budget,block):
    parent.h.source.verify_registry();make_banks(budget,block)
    cells=parent.h.risk.load(parent.h.source.OUT,'validation',budget,block);baseline=parent_source(cells,budget,block);banks={}
    for c in cells:
        tid=str(c['meta']['task_id'])
        if tid not in banks:
            with np.load(OUT/'experts'/str(budget)/block/f'{tid}.npz') as z:
                assert np.array_equal(z['y_validation'],c['y']);banks[tid]=z['predictions']
    policies={}
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family];policies[family]=select(group,baseline,banks)
    root=OUT/'policies'/str(budget)/'regression'/block
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Specific=policies,Shared_unchanged=True,
        parent=str(parent.OUT),seed=0,Specific_inherits_full_Shared_and_previous_Specific=True,extra_clean_TRAIN_supervision=True,
        maximum_additional_auxiliary_learners=8,per_learner_unique_row_cap=budget,query_labels_used_for_fit_or_selection=False))
    queries=parent.h.risk.load(parent.h.source.OUT,'query',budget,block);cache={};rows=[]
    for c in queries:
        meta=c['meta'];tid=str(meta['task_id']);original=parent.OUT/'predictions'/str(budget)/c['key']
        with np.load(original) as z:base=z['Base'];shared=z['Shared'];specific=z['Specific'];truth=z['y_target']
        chosen=policies[meta['family']]['chosen']
        if chosen is not None:
            data,_=parent.h.source.adaptation.raw.scaling.bank('regression',block,meta,meta['family'],meta['rho']);X=data['X_target']
            assert protocol.array_sha256(X)==meta['source_input_hashes']['X_target']
            values=[]
            for member in range(chosen['members']):
                path=OUT/'experts'/str(budget)/block/'models'/tid/f'query_{chosen["profile_index"]}_{member}.joblib'
                if str(path) not in cache:cache[str(path)]=joblib.load(path)
                values.append(parent.h.head_predict(X,cache[str(path)]))
            a=chosen['blend'];specific=(1-a)*specific+a*np.mean(values,axis=0)
        out=OUT/'predictions'/str(budget)/c['key'];parent.h.source.adaptation.raw.scaling.npwrite(out,Base=base,Shared=shared,Specific=specific,y_target=truth)
        record=json.loads(original.with_suffix('.json').read_text());record.update(adaptation_version='v56',Shared_bitwise_unchanged_from_v53=True,
            specific_extra_auxiliary_ensemble=chosen,maximum_additional_auxiliary_learners=8,extra_clean_TRAIN_supervision=True,
            methods={n:parent.h.risk.metrics(truth,p,'regression',meta['scale']) for n,p in [('Base',base),('Shared',shared),('Specific',specific)]})
        atomic_json(out.with_suffix('.json'),record)
        for name,values in record['methods'].items():rows.append(dict(context=budget,block=block,method=name,task_id=meta['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=len(queries),Shared_unchanged=True,adaptation_marker='dagger'))
    print('BAGGED_SPECIFIC_GROUP_COMPLETE',budget,block,len(queries),flush=True)


def plan():
    assert not (OUT/'plan.json').exists();parent.h.source.verify_registry()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_BAG_TRAINING',seed=0,budgets=[256,1024,4096,8192],
        refit=['Specific regression clean','Specific regression seven'],full_goal_checks=48,Shared_unchanged_from_v53=True,
        previous_Specific_fallback=True,profiles=PROFILES,members=MEMBERS,ensemble_count_candidates=[4,8],blends=parent.BLENDS,
        sampling='without replacement; min(B, floor(.8N)) unique original TRAIN IDs per member',
        additional_clean_TRAIN_supervision=True,Specific_may_cover_more_unique_TRAIN_rows_across_members=True,
        per_learner_context_cap=True,maximum_extra_auxiliary_inference_learners=8,
        original_query_labels_used_for_fit_or_selection=False,actual_strength_given=False,Specific_family_visible=True,
        checkpoint_finetuned=False,Base_and_query_conditions_unchanged=True,query_feedback_used_for_development_design=True,
        development_only=True,original_paper_replication=False,pure_TabACE_claim=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((parent.OUT/'paper_targets.json').read_text())
    for budget in [256,1024,4096,8192]:
        for task,blocks,origin in [('classification',['clean','five','seven'],RUN/'results/tabace_augmented_classifier_v54'),('regression',['five'],parent.OUT)]:
            for block in blocks:
                for name in ['policies','predictions']:
                    dest=OUT/name/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.symlink_to(origin/name/str(budget)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);p.add_argument('--block',choices=['clean','seven']);a=p.parse_args()
    plan() if a.phase=='plan' else run(a.budget,a.block)
