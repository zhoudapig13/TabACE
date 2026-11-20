"""Explicit extra-supervision development adapters on capped clean TRAIN rows.

The official TFM banks and paired Base remain immutable. Auxiliary feature
experts train on at most B unique pre-corruption TRAIN observations. Their
selection predictions exclude the entire TRAIN calibration partition. After
source-only selection, the chosen recipes are refit on the capped TRAIN pool;
no final-query feature or label is read until policies are frozen. These are
adapted hybrids, not an unmodified TabACE reproduction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,Lasso
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
import calibration_head_v45 as h
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabace_source_expert_v53'
PARENT=RUN/'results/tabace_meta_calibration_v52'
PROFILES=([dict(kind='ridge',alpha=a) for a in [1.,10.,100.]]+
          [dict(kind='catboost',iterations=256,depth=6)]+
          [dict(kind='lasso',alpha=a) for a in [.001,.005,.01,.025]]+
          [dict(kind='catboost',iterations=512,depth=8),dict(kind='extra_trees',trees=256,leaf=2)])
BLENDS=[.1,.25,.5,.75,1.]


def fit(X,y,profile):
    mean=X.mean(0);scale=X.std(0);scale=np.where(scale>1e-8,scale,1.)
    Z=np.clip((X-mean)/scale,-10,10);mu=float(y.mean());sd=max(float(y.std()),1e-8)
    if profile['kind']=='ridge':model=Ridge(alpha=profile['alpha'],solver='lsqr',tol=1e-6)
    elif profile['kind']=='lasso':model=Lasso(alpha=profile['alpha'],max_iter=10000,tol=1e-7)
    elif profile['kind']=='catboost':
        model=CatBoostRegressor(iterations=profile['iterations'],depth=profile['depth'],learning_rate=.05,
            loss_function='RMSE',l2_leaf_reg=5.,random_seed=0,thread_count=2,verbose=False,allow_writing_files=False)
    else:model=ExtraTreesRegressor(n_estimators=profile['trees'],min_samples_leaf=profile['leaf'],max_features=1.,random_state=0,n_jobs=2)
    model.fit(Z,(y-mu)/sd)
    return dict(center=mean,scale=scale,mu=mu,sigma=sd,estimator=model)


def source_data(meta,block,budget):
    track='five' if block=='clean' else block
    pool=h.source.adaptation.raw.scaling.prepare('regression',track,meta)
    partition=h.calibration(meta,block);held=partition['validation_original_ids']
    keep=~np.isin(pool['source_ids'],held);positions=np.flatnonzero(keep)[:budget]
    fit_ids=pool['source_ids'][positions]
    assert len(np.unique(fit_ids))==len(fit_ids)<=budget
    assert not np.intersect1d(fit_ids,held).size
    assert not np.intersect1d(pool['source_ids'],pool['query_ids']).size
    return pool,partition,positions


def make_experts(budget,block):
    folder=OUT/'experts'/str(budget)/block
    for meta in h.source.adaptation.raw.frozen.cohort('regression',block):
        tid=str(meta['task_id']);out=folder/f'{tid}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        pool,partition,pos=source_data(meta,block,budget)
        values=[];paths=[];models=folder/'models'/tid;models.mkdir(parents=True,exist_ok=True)
        for index,profile in enumerate(PROFILES):
            validation_model=fit(pool['X'][pos],pool['y'][pos],profile)
            values.append(h.head_predict(partition['X_validation'],validation_model))
            model_path=models/f'validation_{index}.joblib';joblib.dump(validation_model,model_path,compress=3)
            # This refit uses TRAIN only and never receives query features.
            n=min(budget,len(pool['y']));model=fit(pool['X'][:n],pool['y'][:n],profile)
            p=models/f'query_{index}.joblib';joblib.dump(model,p,compress=3);paths.append(str(p))
        h.source.adaptation.raw.scaling.npwrite(out,predictions=np.asarray(values),y_validation=partition['y_validation'],
            training_original_ids=pool['source_ids'][pos],validation_original_ids=partition['validation_original_ids'])
        atomic_json(out.with_suffix('.json'),dict(status='SOURCE_EXPERTS_COMPLETE',task_id=meta['task_id'],block=block,
            budget=budget,validation_training_rows=len(pos),query_refit_rows=n,validation_and_training_disjoint=True,
            query_overlap=0,source_X_sha256=protocol.array_sha256(pool['X'][pos]),source_y_sha256=protocol.array_sha256(pool['y'][pos]),
            validation_X_sha256=protocol.array_sha256(partition['X_validation']),validation_y_sha256=protocol.array_sha256(partition['y_validation']),
            query_model_paths=paths,profiles=PROFILES,extra_clean_TRAIN_supervision=True,checkpoint_finetuned=False))
        print('CLEAN_SOURCE_EXPERTS_COMPLETE',budget,block,tid,flush=True)


def score(cells,predictions):
    values=[h.risk.metrics(c['y'],predictions[c['key']],'regression',c['meta']['scale']) for c in cells]
    means={key:float(np.mean([v[key] for v in values])) for key in ['r2','nrmse']}
    return means,.5*(means['r2']-means['nrmse'])


def select(cells,baseline,experts,indices):
    initial,best_value=score(cells,baseline);best=None;best_predictions=baseline;trials=[]
    for index in indices:
        for blend in BLENDS:
            predictions={c['key']:(1-blend)*baseline[c['key']]+blend*experts[str(c['meta']['task_id'])][index] for c in cells}
            values,utility=score(cells,predictions)
            eligible=values['r2']>=initial['r2']-1e-10 and values['nrmse']<=initial['nrmse']+1e-10
            trials.append(dict(profile_index=index,blend=blend,source_scores=values,utility=utility,eligible=eligible))
            if eligible and utility>best_value+1e-10:
                best=dict(profile_index=index,blend=blend);best_value=utility;best_predictions=predictions
    return dict(chosen=best,baseline_scores=initial,trials=trials),best_predictions


def run(budget,block):
    h.source.verify_registry();make_experts(budget,block)
    cells=h.risk.load(h.source.OUT,'validation',budget,block);experts={}
    for c in cells:
        tid=str(c['meta']['task_id'])
        if tid not in experts:
            with np.load(OUT/'experts'/str(budget)/block/f'{tid}.npz') as z:
                assert np.array_equal(z['y_validation'],c['y']);experts[tid]=z['predictions']
    sh,sh_source=select(cells,{c['key']:c['base'] for c in cells},experts,range(4));specific={}
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family]
        specific[family],_=select(group,sh_source,experts,range(len(PROFILES)))
    root=OUT/'policies'/str(budget)/'regression'/block
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Shared=sh,Specific=specific,
        profiles=PROFILES,seed=0,Shared_family_blind=True,Specific_family_visible=True,actual_strength_input=False,
        extra_clean_TRAIN_supervision=True,hybrid_auxiliary_heads=True,unchanged_TFM_banks=True,
        unique_auxiliary_TRAIN_rows_cap=budget,original_query_labels_used_for_fit=False,original_paper_replication=False))
    query=h.risk.load(h.source.OUT,'query',budget,block);cache={};rows=[]
    for c in query:
        meta=c['meta'];tid=str(meta['task_id']);raw,_=h.source.adaptation.raw.scaling.bank('regression',block,meta,meta['family'],meta['rho'])
        X=raw['X_target'];assert protocol.array_sha256(X)==meta['source_input_hashes']['X_target']
        info=json.loads((OUT/'experts'/str(budget)/block/f'{tid}.json').read_text())
        def predict(policy,anchor):
            chosen=policy['chosen']
            if chosen is None:return anchor
            index=chosen['profile_index'];key=(tid,index)
            if key not in cache:cache[key]=joblib.load(info['query_model_paths'][index])
            value=h.head_predict(X,cache[key]);a=chosen['blend'];return (1-a)*anchor+a*value
        shared=predict(sh,c['base']);sp=predict(specific[meta['family']],shared)
        out=OUT/'predictions'/str(budget)/c['key'];h.source.adaptation.raw.scaling.npwrite(out,Base=c['base'],Shared=shared,Specific=sp,y_target=c['y'])
        record={key:meta[key] for key in ['task','block','task_id','dataset','family','rho','requested_context','actual_context','scale']}
        record.update(input_hashes=meta['source_input_hashes'],source_bank=str(c['path']),seed=0,provisional_Base_calibration=False,
            methods={n:h.risk.metrics(c['y'],p,'regression',meta['scale']) for n,p in [('Base',c['base']),('Shared',shared),('Specific',sp)]},
            adaptation_version='v53',extra_clean_TRAIN_supervision=True,auxiliary_source_rows=info['query_refit_rows'],
            original_query_labels_used_for_fit=False,checkpoint_finetuned=False,development_only=True,
            query_feedback_used_for_benchmark_design=True,original_paper_replication=False,independent_test_claim=False)
        atomic_json(out.with_suffix('.json'),record)
        for name,values in record['methods'].items():rows.append(dict(method=name,context=budget,block=block,task_id=meta['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=len(query),Base_unchanged=True,adaptation_marker='dagger'))
    print('SOURCE_EXPERT_GROUP_COMPLETE',budget,block,len(query),flush=True)


def plan():
    h.source.verify_registry();assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_EXPERT_TRAINING',seed=0,budgets=[256,1024,4096,8192],
        refit=['regression clean','regression seven'],full_goal_checks=48,profiles=PROFILES,Shared_profile_indices=list(range(4)),
        Specific_profile_indices=list(range(len(PROFILES))),blends=BLENDS,source_partition='existing held-out TRAIN, all labels excluded from source expert fits',
        refit_after_selection='same recipes on capped original clean TRAIN pool',additional_clean_TRAIN_supervision=True,
        auxiliary_training_row_cap='min(requested context, distinct eligible TRAIN rows)',
        original_TFM_banks_and_Base_unchanged=True,query_arrays_and_strengths_unchanged=True,
        query_labels_used_for_training_or_hyperparameters=False,query_feedback_used_for_development_design=True,
        development_only=True,original_paper_replication=False,pure_TabACE_claim=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((PARENT/'paper_targets.json').read_text())
    for budget in [256,1024,4096,8192]:
        for task,blocks in [('classification',['clean','five','seven']),('regression',['five'])]:
            for block in blocks:
                for name in ['policies','predictions']:
                    dest=OUT/name/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.symlink_to(PARENT/name/str(budget)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);p.add_argument('--block',choices=['clean','seven']);a=p.parse_args()
    plan() if a.phase=='plan' else run(a.budget,a.block)
