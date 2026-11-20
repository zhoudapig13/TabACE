"""TRAIN-only extension-tail diagnostic for the seven-shift development set.

v30 diagnosis isolates large-context failures in domain/temporal, especially
cps88wages. This new protocol caps only ADDITIONAL observed source rows using
the corrupted256 core's median/IQR. It changes the benchmark; it is not an
original-paper reproduction or a checkpoint fine-tune. No query labels enter
construction or parameter selection.
"""
import argparse
import json
import os
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import continuous_recovery_v30 as source
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabicl_tail_recovery_v32'
CAPS=[4.,8.,12.,16.,24.,32.,64.]
VARIANTS=[('base',0.)]+[(kind,cap) for kind in ['y','x','xy'] for cap in CAPS]
AFFECTED=['domain','temporal']


def plan():
    path=OUT/'plan.json';assert not path.exists()
    atomic_json(path,dict(status='DECLARED_BEFORE_NEW_VALIDATION',seed=0,n_estimators=1,
        evidence='v30 source-only diagnosis: domain/temporal cps88wages R2 deteriorates below -50 at8192; other five mechanisms improve on average',
        hypothesis='Rare high-leverage additional rows interact with feature-domain mismatch; robust extension bounds may stabilize the curve',
        caps_in_corrupted_core_IQR_units=CAPS,profiles=['response_only','features_only','both'],
        affected_families=AFFECTED,all_other_families='unchanged zero-repair fixed-core v26 environment',
        original_corrupted256_core_unchanged=True,context_budgets=[256,1024,4096,8192],nominal_strengths=[.2,.5,.8,1.],
        one_shared_rule_across_datasets_strengths_and_budgets=True,context_specific_parameters=False,
        prefer_same_rule_for_both_affected_families=True,otherwise_two_family_rules_allowed=True,
        clean_and_classification_unchanged=True,extra_clean_labels_used=False,
        selection='same disjoint128 TRAIN holdout as v26; no original query labels/scores',
        desired_validation_R2_deltas=[.02,.035,.04],minimum_validation_R2_step=.001,
        final_validation_R2_gain_bounds=[.005,.08],nrmse_nonincreasing=True,
        new_development_protocol=True,original_paper_replication=False,Markdown_written=False))


def bounds(values,cap):
    center=np.median(values,axis=0)
    scale=np.percentile(values,75,axis=0)-np.percentile(values,25,axis=0)
    scale=np.where(scale>1e-8,scale,np.std(values,axis=0))
    scale=np.where(scale>1e-8,scale,np.inf)
    return center-cap*scale,center+cap*scale


def context(env,kind,cap,budget):
    X,y=source.context(env,0.,budget)
    if kind=='base':return X,y
    if 'y' in kind:
        lo,hi=bounds(env[3][:256],cap);y[256:]=np.clip(y[256:],lo,hi)
    if 'x' in kind:
        lo,hi=bounds(env[2][:256],cap);X[256:]=np.clip(X[256:],lo,hi)
    return X,y


def worker(shard,shards):
    import warnings,torch
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0');failures=0
    jobs=[m for block,m in source.v.prior.jobs() if block=='seven']
    for meta in jobs[shard::shards]:
        try:
            out=OUT/'validation'/f'{meta["task_id"]}.npz'
            if out.exists() and out.with_suffix('.json').exists():continue
            pool=source.v.prior.original.prepare('regression','seven',meta);split=source.v.split(meta,'seven',pool)
            Q,t=split['X_validation'],split['y_validation'];scale=max(float(t.std()),1e-12)
            bank=dict(y_validation=t);records=[];new_calls=0
            for name,rho in source.v.grid('seven'):
                env=source.v.environment(pool,meta,'seven',name,rho,split['train_positions'])
                cache={}
                for kind,cap in VARIANTS if name in AFFECTED else VARIANTS[:1]:
                    for budget in [1024,4096,8192]:
                        X,y=context(env,kind,cap,budget)
                        assert np.array_equal(X[:256],env[2][:256]) and np.array_equal(y[:256],env[3][:256])
                        key=(protocol.array_sha256(X),protocol.array_sha256(y));alias=key in cache
                        if alias:p=cache[key]
                        elif kind=='base':
                            old=source.v.OUT/'validation/seven'/str(meta['task_id'])/f'{name}_{rho}'/'0'/f'{budget}.npz'
                            oldmeta=json.loads(old.with_suffix('.json').read_text())
                            assert oldmeta['input_hashes']['X_source']==key[0] and oldmeta['input_hashes']['y_source']==key[1]
                            with np.load(old) as z:
                                assert np.array_equal(t,z['y_target']);p=z['prediction']
                        else:p,_=predictor(X,y,Q,members=1,random_state=0);new_calls+=1
                        assert np.isfinite(p).all();cache[key]=p
                        namekey=f'{name}_{rho}_{kind}_{int(cap)}_{budget}';bank[namekey]=p
                        records.append(dict(task_id=meta['task_id'],dataset=meta['dataset'],family=name,rho=rho,
                            kind=kind,cap=cap,requested_context=budget,actual_context=len(y),prediction_key=namekey,
                            scale=scale,metrics=metrics(t,p,'regression',scale),
                            input_hashes=dict(X_source=key[0],y_source=key[1],X_target=protocol.array_sha256(Q),y_target=protocol.array_sha256(t)),
                            original_core_preserved=True,source_only_tail_bounds=True,clean_TRAIN_labels_used=False,
                            seed=0,n_estimators=1,identical_input_prediction_reuse=alias))
            source.v.prior.original.npwrite(out,**bank);atomic_json(out.with_suffix('.json'),dict(records=records,new_TFM_calls=new_calls))
            print('TAIL_VALIDATION_DATASET_COMPLETE',meta['dataset'],'new_TFM_calls',new_calls,flush=True)
        except Exception:
            failures+=1;error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{meta["task_id"]}.json',dict(error=error))
    if failures:raise SystemExit(1)


def audit():
    from run_tabicl import metrics
    errors=[];count=0;reconstructed=0
    for meta in [m for b,m in source.v.prior.jobs() if b=='seven']:
        path=OUT/'validation'/f'{meta["task_id"]}.npz'
        if not path.exists() or not path.with_suffix('.json').exists():errors.append([str(path),'missing']);continue
        pool=source.v.prior.original.prepare('regression','seven',meta);split=source.v.split(meta,'seven',pool)
        if np.intersect1d(split['source_original_ids'],split['validation_original_ids']).size:errors.append([str(path),'source_overlap'])
        records=json.loads(path.with_suffix('.json').read_text())['records'];seen=set()
        with np.load(path) as bank:
            truth=bank['y_validation']
            if not np.array_equal(truth,split['y_validation']):errors.append([str(path),'truth'])
            for r in records:
                key=(r['family'],r['rho'],r['kind'],r['cap'],r['requested_context'])
                if key in seen:errors.append([str(path),'duplicate'])
                seen.add(key)
                measured=metrics(truth,bank[r['prediction_key']],'regression',r['scale'])
                if any(not np.isclose(value,r['metrics'][name],atol=1e-12,rtol=0) for name,value in measured.items()):errors.append([str(path),'metric',key])
                if r['input_hashes']['X_target']!=protocol.array_sha256(split['X_validation']) or r['input_hashes']['y_target']!=protocol.array_sha256(truth):errors.append([str(path),'partition',key])
                if r['family'] in AFFECTED and r['rho']==100 and r['kind']=='xy' and r['cap']==16 and r['requested_context']==8192:
                    env=source.v.environment(pool,meta,'seven',r['family'],100,split['train_positions'])
                    X,y=context(env,'xy',16,8192)
                    if any(protocol.array_sha256(a)!=r['input_hashes'][name] for name,a in [('X_source',X),('y_source',y)]):errors.append([str(path),'source',key])
                    reconstructed+=1
                count+=1
        expected={(f,r,k,c,n) for f,r in source.v.grid('seven') for k,c in (VARIANTS if f in AFFECTED else VARIANTS[:1]) for n in [1024,4096,8192]}
        if seen!=expected:errors.append([str(path),'coverage',len(seen),len(expected)])
    if count!=19404:errors.append(['global_coverage',count,19404])
    atomic_json(OUT/'validation_audit.json',dict(status='PASS' if not errors else 'FAIL',verified_predictions=count,
        reconstructed_source_contexts=reconstructed,query_predictions=0,failures=errors))
    print('TAIL_VALIDATION_AUDIT',count,len(errors),errors[:3],flush=True)
    if errors:raise SystemExit(1)


def select():
    assert json.loads((OUT/'validation_audit.json').read_text())['status']=='PASS'
    assert not (OUT/'selection.json').exists(),'Do not repeat selection from query feedback'
    rows=[]
    for path in (OUT/'validation').glob('*.json'):
        rows += [{k:v for k,v in r.items() if k not in ['metrics','input_hashes']}|r['metrics'] for r in json.loads(path.read_text())['records']]
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'validation_detail.csv',index=False)
    grouped=frame.groupby(['family','kind','cap','requested_context'],as_index=False).agg(
        datasets=('task_id','nunique'),conditions=('rho','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    options={}
    for name in AFFECTED:
        result=[]
        for kind,cap in VARIANTS:
            r=grouped[(grouped.family==name)&(grouped.kind==kind)&(grouped.cap==cap)].sort_values('requested_context')
            assert len(r)==3 and (r.datasets==33).all() and (r.conditions==132).all()
            result.append(r[['r2','nrmse']].values)
        options[name]=np.asarray(result)
    rest=grouped[~grouped.family.isin(AFFECTED)].groupby('requested_context')[['r2','nrmse']].mean().sort_index().values
    base=pd.read_csv(source.v.OUT/'validation_base256.csv').set_index('block').loc['seven'][['r2','nrmse']].values.astype(float)
    trials=[]
    for a,(kind_a,cap_a) in enumerate(VARIANTS):
        for b,(kind_b,cap_b) in enumerate(VARIANTS):
            avg=(5*rest+options['domain'][a]+options['temporal'][b])/7
            curve=np.vstack([base,avg]);delta=avg[:,0]-base[0]
            valid=bool(np.all(np.diff(curve[:,0])>=.001) and np.all(np.diff(curve[:,1])<=0) and .005<=delta[-1]<=.08)
            trials.append(dict(domain=dict(kind=kind_a,cap=cap_a),temporal=dict(kind=kind_b,cap=cap_b),
                uniform=a==b,validation_goal_met=valid,means=avg.tolist(),
                loss=float(np.sum(((delta-np.asarray([.02,.035,.04]))/.02)**2))))
    good=[r for r in trials if r['validation_goal_met']];uniform=[r for r in good if r['uniform']]
    best=min(uniform or good,key=lambda r:r['loss']) if good else None
    atomic_json(OUT/'selection.json',dict(status='SOURCE_VALIDATION_GOAL_MET' if best else 'NO_VALID_SOURCE_RULE',
        best=best,checked_rules=len(trials),passing_rules=len(good),trials=trials,query_evaluations=0))
    if best:
        atomic_json(OUT/'parameters.json',dict(status='FROZEN_BEFORE_QUERY_EVALUATION',
            policies={name:best[name] for name in AFFECTED},all_other_families='base',
            seed=0,n_estimators=1,development_only=True,extra_clean_TRAIN_labels_used=False,
            corrupted256_core_unchanged=True,clean_and_classification_unchanged=True))
    print('TAIL_SOURCE_SELECTION',json.dumps(dict(best=best,passing=len(good),checked=len(trials))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','audit','select']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    worker(a.shard,a.shards) if a.phase=='worker' else globals()[a.phase]()
