"""Expand Specific with source-validated sparse experts; Shared is immutable.

Additional noisy prediction features can worsen generalization. Sparse heads
and features-only heads test that hypothesis with exactly the same TRAIN
labels. No query score chooses profiles, blending weights, or fallback rules.
"""
import argparse
import json
import os
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Lasso
import calibration_head_v45 as h
from replay_head_oof_v47 import replay
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabace_sparse_specific_v47'
PROFILES=([dict(kind='ridge',alpha=a,scope='features_only') for a in [.1,1.,10.,100.]]+
          [dict(kind='lasso',alpha=a,scope=s) for s in ['features_only','features_and_members'] for a in [.001,.005,.01,.025]])


def matrix(X,c,profile):return np.asarray(X,dtype=np.float64) if profile['scope']=='features_only' else h.features(X,c,21)


def train(A,y,conditions,unique,profile):
    if profile['kind']=='ridge':return h.fit_estimator(A,y,conditions,unique,profile)
    mean=A.mean(0);scale=A.std(0);scale=np.where(scale>1e-8,scale,1.)
    mu=float(y.mean());sigma=max(float(y.std()),1e-8);Z=np.clip((A-mean)/scale,-10,10)
    model=Lasso(alpha=profile['alpha'],max_iter=5000,tol=1e-6,selection='cyclic').fit(Z,(y-mu)/sigma,sample_weight=np.full(len(y),1./conditions))
    return dict(center=mean,scale=scale,mu=mu,sigma=sigma,estimator=model)


def fit(cells,block,baseline,directory):
    d=h.risk.prepare(cells,21,block);initial=np.concatenate([baseline[c['key']] for c in cells]);scores,best_value=h.risk.score(d,initial)
    best=None;trials=[];groups={}
    for i,c in enumerate(cells):groups.setdefault(str(c['meta']['task_id']),[]).append(i)
    for profile in PROFILES:
        oof=[np.empty_like(c['y'],dtype=np.float64) for c in cells]
        for task,indices in groups.items():
            c=cells[indices[0]];cal=h.calibration(c['meta'],block);X=cal['X_validation'];F=h.folds(c['meta'],block,len(c['y']))
            matrices=[matrix(X,cells[i],profile) for i in indices]
            for fold in range(4):
                train_mask=F!=fold;valid=F==fold
                A=np.concatenate([M[train_mask] for M in matrices]);y=np.concatenate([cells[i]['y'][train_mask] for i in indices])
                fitted=train(A,y,len(indices),int(train_mask.sum()),profile)
                for i,M in zip(indices,matrices):oof[i][valid]=h.head_predict(M[valid],fitted)
        flat=np.concatenate(oof)
        for blend in h.BLENDS:
            prediction=(1-blend)*initial+blend*flat;values,utility=h.risk.score(d,prediction)
            trials.append(dict(profile=profile,blend=blend,source_OOF_scores=values,utility=utility))
            if utility>best_value+1e-10:best_value=utility;best=dict(profile=profile,blend=blend)
    paths={}
    if best:
        for task,indices in groups.items():
            c=cells[indices[0]];X=h.calibration(c['meta'],block)['X_validation'];A=np.concatenate([matrix(X,cells[i],best['profile']) for i in indices]);y=np.concatenate([cells[i]['y'] for i in indices])
            model=train(A,y,len(indices),len(c['y']),best['profile']);path=directory/'models'/f'{task}.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,path,compress=3);paths[task]=str(path)
    policy=dict(chosen=best,models=paths,source_baseline_scores=scores)
    atomic_json(directory/'policy.json',dict(status='FROZEN_BEFORE_QUERY',model=policy,trials=trials,Shared_changed=False,query_labels_used_for_fit=False))
    return policy


def run(budget,block):
    cells=h.risk.load(h.source.OUT,'validation',budget,block)
    old=h.OUT/'policies'/str(budget)/'regression'/block;root=OUT/'policies'/str(budget)/'regression'/block
    shared=json.loads((old/'Shared/policy.json').read_text())['model']
    shared_oof,check=replay(cells,10,block,shared,{c['key']:c['base'] for c in cells})
    policies={};originals={};replay_checks=dict(Shared=check)
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family]
        original=json.loads((old/f'Specific_{family}/policy.json').read_text())['model'];originals[family]=original
        baseline,score=replay(group,21,block,original,shared_oof);replay_checks[family]=score
        policies[family]=fit(group,block,baseline,root/f'Specific_{family}')
    atomic_json(root/'source_replay.json',dict(scores=replay_checks,query_labels_used=False))
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Shared_source=str(old/'Shared/policy.json'),
        source_bank=str(h.source.OUT),Shared_unchanged=True,no_additional_TFM_calls=True,extra_clean_TRAIN_supervision=True))
    query=h.risk.load(h.source.OUT,'query',budget,block);cache={};sparse_cache={};rows=[]
    for c in query:
        meta=c['meta'];data,_=h.source.adaptation.raw.scaling.bank('regression',block,meta,meta['family'],meta['rho']);X=data['X_target']
        assert protocol.array_sha256(X)==meta['source_input_hashes']['X_target']
        sh=h.predict(c,shared,c['base'],X,cache);baseline=h.predict(c,originals[meta['family']],sh,X,cache)
        original_path=h.OUT/'predictions'/str(budget)/c['key']
        with np.load(original_path) as z:assert np.allclose(sh,z['Shared'],atol=1e-12,rtol=0) and np.allclose(baseline,z['Specific'],atol=1e-12,rtol=0)
        policy=policies[meta['family']];best=policy['chosen'];sp=baseline
        if best:
            path=policy['models'][str(meta['task_id'])]
            if path not in sparse_cache:sparse_cache[path]=joblib.load(path)
            pred=h.head_predict(matrix(X,c,best['profile']),sparse_cache[path]);a=best['blend'];sp=(1-a)*baseline+a*pred
        out=OUT/'predictions'/str(budget)/c['key'];h.source.adaptation.raw.scaling.npwrite(out,Base=c['base'],Shared=sh,Specific=sp,y_target=c['y'])
        summary=json.loads(original_path.with_suffix('.json').read_text());summary.update(adaptation_version='v47',Shared_unchanged_from_v45=True,
            methods={name:h.risk.metrics(c['y'],p,'regression',meta['scale']) for name,p in [('Base',c['base']),('Shared',sh),('Specific',sp)]})
        atomic_json(out.with_suffix('.json'),summary)
        for name,score in summary['methods'].items():rows.append(dict(method=name,context=budget,block=block,task_id=meta['task_id'],**score))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=len(query),Shared_bitwise_or_1e12_equivalent=True,query_labels_used_for_fit=False))
    print('SPARSE_SPECIFIC_COMPLETE',budget,block,len(query),flush=True)


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_SPARSE_FITS',full_goal_checks=48,source_banks=str(h.source.OUT),
        budgets=[256,1024,4096,8192],refit='Specific regression five/seven only',
        Shared_unchanged=True,Base_unchanged=True,no_additional_TFM_calls=True,
        new_experts=PROFILES,blend_candidates=h.BLENDS,original_Specific_fallback=True,
        selection='Grouped four-fold TRAIN validation; original Shared and Specific policies replayed without new HPO',
        query_labels_used_for_fit=False,extra_clean_TRAIN_supervision=True,
        query_feedback_used_for_benchmark_design=True,development_only=True,
        original_paper_replication=False,independent_test_claim=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((h.OUT/'paper_targets.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);p.add_argument('--block',choices=['five','seven']);a=p.parse_args();plan() if a.phase=='plan' else run(a.budget,a.block)
