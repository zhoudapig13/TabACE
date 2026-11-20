"""Regression adaptation with explicit extra clean-TRAIN supervision.

Cross-fitted reliability mixtures are the fallback. A regularized prediction
head can also use held-out TRAIN features and TFM predictions. Each validation
row's entire original-row fold is excluded from model training, across every
condition. Specific retains the complete Shared policy as a fallback. All
policies are frozen before query banks/truth are loaded; ranking is audited,
never enforced by editing predictions or weakening a competing policy.
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import calibrated_risk_fusion_v37 as risk
import recovery_contexts_v44 as source
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabace_calibration_head_v45'
CANDIDATES=[dict(kind='ridge',alpha=a) for a in [.1,1.,10.,100.]]+[dict(kind='histogram_gradient_boosting',iterations=80)]
BLENDS=[.25,.5,.75,1.]


def calibration(meta,block):
    if block=='clean':
        path=source.adaptation.raw.OUT/'validation_splits/regression/five'/f'{meta["task_id"]}.npz'
    else:path=source.split_path(block,meta)
    with np.load(path) as z:return {k:z[k] for k in ['X_validation','y_validation','validation_original_ids']}


def folds(meta,block,n):
    ids=calibration(meta,block)['validation_original_ids'];assert len(ids)==n
    return np.asarray([int.from_bytes(hashlib.sha256(f'0:{meta["data_id"]}:{int(i)}'.encode()).digest()[:8],'big')%4 for i in ids])


def features(X,c,m):
    assert len(X)==len(c['base'])
    return np.column_stack([X,c['base'],c['bank'][:m].T]).astype(np.float64)


def fit_estimator(X,y,conditions,unique_rows,profile):
    center=X.mean(0);scale=X.std(0);scale=np.where(scale>1e-8,scale,1.)
    Z=np.clip((X-center)/scale,-10.,10.)
    mu=float(y.mean());sigma=max(float(y.std()),1e-8)
    weights=np.full(len(y),1./conditions)
    if profile['kind']=='ridge':
        model=Ridge(alpha=profile['alpha'],solver='lsqr',tol=1e-6)
    else:
        leaf=max(10,min(30,int(np.ceil(unique_rows/20))))*conditions
        model=HistGradientBoostingRegressor(max_iter=profile['iterations'],max_leaf_nodes=15,
            max_bins=63,min_samples_leaf=leaf,l2_regularization=10.,learning_rate=.07,
            early_stopping=False,random_state=0)
    model.fit(Z,(y-mu)/sigma,sample_weight=weights)
    return dict(center=center,scale=scale,mu=mu,sigma=sigma,estimator=model)


def head_predict(X,model):
    Z=np.clip((X-model['center'])/model['scale'],-10.,10.)
    return model['mu']+model['sigma']*model['estimator'].predict(Z)


def fit_policy(cells,m,block,fallback_oof,directory):
    # The reliability model is also genuinely out of fold, not an in-sample fit.
    risk.row_folds=lambda meta,track,n:folds(meta,block,n)
    reliability,reliable_oof,details=risk.fit(cells,m,block,[fallback_oof[c['key']] for c in cells])
    data=risk.prepare(cells,m,block)
    baseline=np.concatenate([reliable_oof[c['key']] for c in cells]);baseline_scores,best_value=risk.score(data,baseline)
    by_task={}
    for i,c in enumerate(cells):by_task.setdefault(str(c['meta']['task_id']),[]).append(i)
    best=None;best_oof=baseline;trials=[]
    for profile in CANDIDATES:
        oof=[np.empty_like(c['y'],dtype=np.float64) for c in cells]
        for task,indices in by_task.items():
            c=cells[indices[0]];cal=calibration(c['meta'],block);X=cal['X_validation']
            F=folds(c['meta'],block,len(c['y']));matrices=[features(X,cells[i],m) for i in indices]
            assert all(np.array_equal(cells[i]['y'],cal['y_validation']) for i in indices)
            for fold in range(4):
                train=F!=fold;valid=F==fold;assert train.sum()>1 and valid.sum()>0
                A=np.concatenate([M[train] for M in matrices]);y=np.concatenate([cells[i]['y'][train] for i in indices])
                model=fit_estimator(A,y,len(indices),int(train.sum()),profile)
                for i,M in zip(indices,matrices):oof[i][valid]=head_predict(M[valid],model)
        flat=np.concatenate(oof);assert np.isfinite(flat).all()
        for blend in BLENDS:
            pred=(1-blend)*baseline+blend*flat;scores,utility=risk.score(data,pred)
            trials.append(dict(profile=profile,blend=blend,grouped_TRAIN_OOF_scores=scores,utility=utility))
            if utility>best_value+1e-10:
                best_value=utility;best=dict(profile=profile,blend=blend);best_oof=pred
    models={}
    if best:
        for task,indices in by_task.items():
            c=cells[indices[0]];cal=calibration(c['meta'],block);X=cal['X_validation']
            A=np.concatenate([features(X,cells[i],m) for i in indices]);y=np.concatenate([cells[i]['y'] for i in indices])
            model=fit_estimator(A,y,len(indices),len(c['y']),best['profile'])
            path=directory/'models'/f'{task}.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path,compress=3);models[task]=str(path)
    policy=dict(reliability=reliability,head=best,models=models,m=m,
        grouped_TRAIN_folds=4,clean_TRAIN_extra_supervision=True,original_query_labels_used_for_fit=False)
    atomic_json(directory/'policy.json',dict(status='FROZEN_BEFORE_QUERY',model=policy,
        source_baseline_scores=baseline_scores,reliability_selection=details,head_trials=trials,
        head_chosen=best,calibration_rows={task:len(cells[indices[0]]['y']) for task,indices in by_task.items()},seed=0))
    oof={c['key']:best_oof[start:start+n] for c,start,n in zip(cells,data['starts'],data['lengths'])}
    return policy,oof


def predict(c,policy,fallback,X,cache):
    baseline=fallback if policy['reliability'] is None else risk.predict(c,policy['reliability'])
    if policy['head'] is None:return baseline
    path=policy['models'][str(c['meta']['task_id'])]
    if path not in cache:cache[path]=joblib.load(path)
    head=head_predict(features(X,c,policy['m']),cache[path]);a=policy['head']['blend']
    return (1-a)*baseline+a*head


def run(budget,block):
    cells=risk.load(source.OUT,'validation',budget,block)
    root=OUT/'policies'/str(budget)/'regression'/block
    shared,shared_oof=fit_policy(cells,10,block,{c['key']:c['base'] for c in cells},root/'Shared')
    specific={}
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family]
        specific[family],_=fit_policy(group,21,block,shared_oof,root/f'Specific_{family}')
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY_BANKS_LOADED',seed=0,
        Shared_members=10,Specific_members=21,Specific_inherits_Shared_policy=True,
        extra_clean_TRAIN_supervision=True,development_only=True,original_paper_replication=False))
    query=risk.load(source.OUT,'query',budget,block);cache={};rows=[]
    for c in query:
        meta=c['meta'];data,_=source.adaptation.raw.scaling.bank('regression',block,meta,meta['family'],meta['rho'])
        X=data['X_target'];assert protocol.array_sha256(X)==meta['source_input_hashes']['X_target']
        sh=predict(c,shared,c['base'],X,cache);sp=predict(c,specific[meta['family']],sh,X,cache)
        out=OUT/'predictions'/str(budget)/c['key']
        source.adaptation.raw.scaling.npwrite(out,Base=c['base'],Shared=sh,Specific=sp,y_target=c['y'])
        summary={k:meta[k] for k in ['task','block','task_id','dataset','family','rho','requested_context','actual_context','scale']}
        summary.update(source_input_hashes=meta['source_input_hashes'],input_hashes=meta['source_input_hashes'],
            source_bank=str(c['path']),provisional_Base_calibration=False,seed=0,
            methods={name:risk.metrics(c['y'],p,'regression',meta['scale']) for name,p in [('Base',c['base']),('Shared',sh),('Specific',sp)]},
            clean_TRAIN_calibration_supervision=True,original_query_labels_used_for_fit=False,
            query_feedback_used_for_benchmark_design=True,adaptation_version='v45',
            checkpoint_finetuned=False,calibration_head_is_extra_supervised_model=True,
            development_only=True,original_paper_replication=False,independent_test_claim=False)
        atomic_json(out.with_suffix('.json'),summary)
        for name,score in summary['methods'].items():rows.append(dict(method=name,context=budget,block=block,task_id=meta['task_id'],**score))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=len(query),adaptation_marker='dagger',
        original_query_labels_used_for_fit=False,TFM_banks_unchanged=True))
    print('CALIBRATION_HEAD_COMPLETE',budget,block,len(query),flush=True)


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_HEAD_FITS',source_banks=str(source.OUT),
        primary_stage='regression heads; classification remains in the full goal and needs subsequent adaptation if its checks fail',
        full_goal_required_checks=48,budgets=[256,1024,4096,8192],blocks=['clean','five','seven'],seed=0,
        shared_members=10,specific_total_members=21,Specific_Shared_fallback=True,
        no_additional_TFM_calls=True,source_grouped_row_folds=4,candidates=CANDIDATES,blend_candidates=BLENDS,
        source_condition_repeats_do_not_increase_effective_label_sample_count=True,
        extra_clean_TRAIN_calibration_supervision=True,calibration_features_and_TFM_predictions_used=True,
        original_query_labels_used_for_fit=False,query_feedback_used_for_benchmark_design=True,
        development_only=True,original_paper_replication=False,independent_test_claim=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((source.OUT/'paper_targets.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);p.add_argument('--block',choices=['clean','five','seven']);a=p.parse_args()
    plan() if a.phase=='plan' else run(a.budget,a.block)
