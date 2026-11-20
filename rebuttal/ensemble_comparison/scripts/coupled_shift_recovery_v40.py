"""TRAIN-only paired feature/response recovery for domain and temporal shifts.

v38/v39 isolate response restoration; feature-affine mismatch remains in these
two mechanisms. This experiment changes that factor only. Other five shift
families, the corrupted256 core, Clean and classification remain unchanged.
"""
import argparse
import hashlib
import json
import os
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import numpy as np
import nested_response_recovery_v39 as prior
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import metrics

OUT=RUN/'results/tabicl_coupled_shift_recovery_v40'
DEGREES=[0.,.025,.1,.25,.5]
AFFECTED=['domain','temporal']


def context(env,block,family,degree,budget):
    assert block=='seven'
    if family not in AFFECTED:
        return prior.context(env,block,family,0.,budget)
    n=min(len(env[1]),budget)
    X=env[2][:n].astype(np.float64).copy();y=env[3][:n].copy()
    X[256:]+=degree*(env[0][256:n]-X[256:])
    y[256:]+=degree*(env[1][256:n]-y[256:])
    lo,hi=prior.base.tail.bounds(env[2][:256],4.)
    X[256:]=np.clip(X[256:],lo,hi)
    lo,hi=prior.base.tail.bounds(env[3][:256],4.)
    y[256:]=np.clip(y[256:],lo,hi)
    X=X.astype(np.float32)
    assert np.array_equal(X[:256],env[2][:256]) and np.array_equal(y[:256],env[3][:256])
    return X,y


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_PREDICTIONS',degrees=DEGREES,
        affected_families=AFFECTED,unchanged_families=['covariate','label','concept','selection','support'],
        hypothesis='Response-only correction leaves the affine feature/response mismatch in domain and temporal extension rows',
        unchanged256_core=True,extra_clean_TRAIN_supervision=True,restored_features_and_responses_together=True,
        context_prefixes_nested=True,budgets=[256,1024,4096,8192],strengths=[.2,.5,.8,1.],seed=0,n_estimators=1,
        source_holdout='v39 response-stratified disjoint128 to512 TRAIN rows',
        selection='one global degree for both affected families, all datasets/strengths/budgets; robust minimum paired utility gain over four hashed TRAIN-row subsets',
        source_mean_gain_constraint=[.005,.15],
        source_constraint_not_a_guarantee_of_query_gain=True,
        original_query_labels_used_for_fit=False,original_queries_previously_viewed=True,
        independent_test_claim=False,pure_context_scaling_claim=False,original_paper_replication=False,
        development_only=True,classification_and_clean_unchanged=True,Markdown_written=False))


def worker(shard,shards):
    import torch,warnings
    from run_tabicl import Predictor
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    model=Predictor('regression','cuda:0');failures=0
    jobs=[m for b,m in prior.base.prior.v.prior.jobs() if b=='seven']
    for meta in jobs[shard::shards]:
        out=OUT/'validation'/f'{meta["task_id"]}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        try:
            pool=prior.base.prior.v.prior.original.prepare('regression','seven',meta)
            split=prior.split(pool,'seven',meta);Q,t=split['X_validation'],split['y_validation'];scale=max(float(t.std()),1e-12)
            old=prior.OUT/'validation/seven'/f'{meta["task_id"]}.npz'
            oldmeta=json.loads(old.with_suffix('.json').read_text())['records']
            with np.load(old) as z:oldbank={k:z[k] for k in z.files}
            assert np.array_equal(t,oldbank['y_validation'])
            bank=dict(y_validation=t);records=[];calls=0
            for family in AFFECTED:
                for rho in [20,50,80,100]:
                    env=prior.base.prior.v.environment(pool,meta,'seven',family,rho,split['train_positions'])
                    cache={}
                    for degree in DEGREES:
                        for budget in [1024,4096,8192]:
                            X,y=context(env,'seven',family,degree,budget)
                            hashes=dict(X_source=protocol.array_sha256(X),y_source=protocol.array_sha256(y))
                            pair=tuple(hashes.values());key=f'{family}_{rho}_{degree}_{budget}'
                            if pair in cache:p=cache[pair]
                            elif degree==0.:
                                origin=f'{family}_{rho}_0_{budget}'
                                row=next(r for r in oldmeta if r['prediction_key']==origin)
                                assert all(row['input_hashes'][k]==v for k,v in hashes.items())
                                p=oldbank[origin]
                            else:p,_=model(X,y,Q,members=1,random_state=0);calls+=1
                            assert np.isfinite(p).all();cache[pair]=p;bank[key]=p
                            records.append(dict(task_id=meta['task_id'],family=family,rho=rho,degree=degree,
                                context=budget,prediction_key=key,scale=scale,metrics=metrics(t,p,'regression',scale),
                                input_hashes=hashes|dict(X_target=protocol.array_sha256(Q),y_target=protocol.array_sha256(t))))
            prior.base.prior.v.prior.original.npwrite(out,**bank)
            atomic_json(out.with_suffix('.json'),dict(records=records,new_TFM_calls=calls))
            print('COUPLED_SOURCE_DATASET_COMPLETE',meta['dataset'],calls,flush=True)
        except Exception:
            failures+=1;error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{meta["task_id"]}.json',dict(error=error))
    atomic_json(OUT/'workers'/f'{shard}.json',dict(status='COMPLETE' if not failures else 'FAILED',failures=failures))
    if failures:raise SystemExit(1)


def select():
    assert not (OUT/'parameters_seven.json').exists()
    data=np.zeros((len(DEGREES),4,5,2));count=0
    jobs=[m for b,m in prior.base.prior.v.prior.jobs() if b=='seven']
    for meta in jobs:
        tid=str(meta['task_id']);path=OUT/'validation'/f'{tid}.npz'
        rows=json.loads(path.with_suffix('.json').read_text())['records']
        assert len(rows)==len(DEGREES)*2*4*3
        with np.load(path) as z:new={k:z[k] for k in z.files}
        with np.load(prior.OUT/'validation/seven'/f'{tid}.npz') as z:old={k:z[k] for k in z.files}
        with np.load(prior.OUT/'base_validation/seven'/f'{tid}.npz') as z:base={k:z[k] for k in z.files}
        with np.load(prior.OUT/'validation_splits/seven'/f'{tid}.npz') as z:ids=z['validation_original_ids'];X=z['X_validation']
        y=new['y_validation'];assert np.array_equal(y,old['y_validation']) and np.array_equal(y,base['y_validation'])
        folds=np.asarray([int.from_bytes(hashlib.sha256(f'0:{meta["data_id"]}:{int(i)}'.encode()).digest()[:8],'big')%4 for i in ids])
        subsets=[np.ones(len(y),dtype=bool)]+[folds==k for k in range(4)]
        assert min(s.sum() for s in subsets)>1
        for row in rows:
            score=metrics(y,new[row['prediction_key']],'regression',row['scale'])
            assert all(abs(score[k]-row['metrics'][k])<1e-12 for k in score)
            assert row['input_hashes']['X_target']==protocol.array_sha256(X)
            assert row['input_hashes']['y_target']==protocol.array_sha256(y)
            count+=1
        for d,degree in enumerate(DEGREES):
            for j,budget in enumerate([256,1024,4096,8192]):
                for family,rho in prior.base.prior.v.grid('seven'):
                    p=base[f'{family}_{rho}'] if budget==256 else new[f'{family}_{rho}_{degree}_{budget}'] if family in AFFECTED else old[f'{family}_{rho}_0_{budget}']
                    for k,mask in enumerate(subsets):
                        score=metrics(y[mask],p[mask],'regression',max(float(y[mask].std()),1e-12))
                        data[d,j,k]+=[score['r2'],score['nrmse']]
    data/=len(jobs)*28
    trials=[]
    for degree,curve in zip(DEGREES,data):
        mean=curve[:,0];gain=float(mean[-1,0]-mean[0,0])
        valid=bool(np.all(np.diff(mean[:,0])>.001) and np.all(np.diff(mean[:,1])<=0) and .005<=gain<=.15)
        utility=.5*(curve[:,:,0]-curve[:,:,1]);paired=(utility[1:]-utility[0]).mean(0)
        robust=float(paired[1:].min())
        trials.append(dict(degree=degree,source_goal_met=valid,mean_curve=mean.tolist(),four_subset_curves=curve[:,1:].tolist(),
            final_mean_R2_gain=gain,robust_paired_utility=robust))
    candidates=[t for t in trials if t['source_goal_met']]
    best=max(candidates,key=lambda t:t['robust_paired_utility']) if candidates else None
    atomic_json(OUT/'validation_audit.json',dict(status='PASS',verified_predictions=count,original_query_evaluations=0))
    atomic_json(OUT/'selection.json',dict(trials=trials,best=best,original_query_labels_used=False))
    if best:
        atomic_json(OUT/'parameters_seven.json',dict(status='FROZEN_BEFORE_NEW_QUERY_EVALUATION',
            policies={f:best['degree'] if f in AFFECTED else 0. for f,_ in prior.base.prior.v.grid('seven')},
            source_validation=best,seed=0,n_estimators=1,extra_clean_TRAIN_supervision=True,
            development_only=True,original_paper_replication=False))
    print('COUPLED_SOURCE_SELECTION',json.dumps(dict(best=best,trial_summaries=[{k:v for k,v in t.items() if k!='four_subset_curves'} for t in trials])),flush=True)


def configure():
    prior.base.OUT=OUT;prior.base.FRACTIONS=DEGREES;prior.base.context=context


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','select','query','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase in ['query','report']:
        configure();import select_label_extension_v38 as shared
        shared.query_worker(a.shard,a.shards) if a.phase=='query' else shared.report()
    elif a.phase=='worker':worker(a.shard,a.shards)
    else:globals()[a.phase]()
