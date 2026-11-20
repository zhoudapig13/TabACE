"""Fixed-environment regression scaling diagnosis, independently of score targets.

Generate each complete source environment ONCE, freeze its statistics, then take
nested source-stratified prefixes. Importance probabilities are capped at 10/N
to prevent a nominal large context collapsing to repeated copies of one row.
This is a NEW DEVELOPMENT PROTOCOL, not a replacement for original paper data.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import tabicl_context_scaling_v23 as old
from prepare_data import RUN, atomic_json, protocol

OUT=RUN/'results/tabicl_regression_scaling_v24'
BUDGETS=[256,1024,4096,8192]
DENSITY_CAP=10.


def stratified_order(values,task_id,tag):
    """Source response quantile bins; no query labels or outcome-based selection."""
    values=np.asarray(values)
    cuts=np.unique(np.quantile(values,np.linspace(0,1,11)[1:-1]))
    bins=np.searchsorted(cuts,values,side='right')
    rng=np.random.default_rng(np.random.SeedSequence([0,int(task_id),2401,int(tag)]))
    rows=[];priorities=[]
    for label in np.unique(bins):
        ids=rng.permutation(np.flatnonzero(bins==label));count=len(ids)
        rows.extend(ids.tolist());priorities.extend(((np.arange(count)+rng.random())/count).tolist())
    return np.asarray(rows,dtype=int)[np.argsort(priorities,kind='stable')]


def probabilities(score,coefficient):
    logits=np.clip(coefficient*np.asarray(score,dtype=np.float64),-15,15)
    weights=np.exp(logits-logits.max());raw=weights/weights.sum();limit=min(1.,DENSITY_CAP/len(raw))
    lo,hi=0.,1./max(float(weights.min()),1e-30)
    for _ in range(100):
        mid=(lo+hi)/2
        if np.minimum(mid*weights,limit).sum()<1:lo=mid
        else:hi=mid
    prob=np.minimum(hi*weights,limit);prob/=prob.sum()
    assert prob.max()<=limit+1e-12
    return prob,dict(raw_ess=float(1/np.sum(raw**2)),bounded_ess=float(1/np.sum(prob**2)),
                     raw_max_probability=float(raw.max()),bounded_max_probability=float(prob.max()),
                     density_ratio_cap=DENSITY_CAP)


def seven_environment(pool,family,rho,Q):
    X=pool['X'].copy();y=pool['y'].copy();ids=np.arange(len(y));s=rho/100.
    cfg=old.frozen.configuration()['regression_seven']['family_amplitude_multipliers'];a=cfg[family]
    mean=X.mean(0);sd=np.maximum(X.std(0),1e-6);z=(X-mean)/sd;j=int(np.argmax(X.var(0)))
    ystd=max(float(y.std()),1e-6);info={}
    def sample(score,strength):
        p,diagnostics=probabilities(score,3*a*strength)
        return np.random.default_rng(2718).choice(len(y),size=len(y),replace=True,p=p),diagnostics
    if family=='covariate':
        affected=np.random.default_rng(2718).permutation(len(y))[:int(np.ceil(s*len(y)))]
        chosen,info=sample(z[:,j],1.)
        X[affected],y[affected]=X[chosen[affected]],y[chosen[affected]];ids[affected]=chosen[affected]
    elif family=='label':
        chosen,info=sample((y-y.mean())/ystd,s);X,y,ids=X[chosen],y[chosen],ids[chosen]
    elif family in ['selection','support']:
        distance=np.mean(((X-Q.mean(0))/sd)**2,axis=1);score=distance/max(float(distance.std()),1e-6)
        chosen,info=sample(score if family=='selection' else -score,s);X,y,ids=X[chosen],y[chosen],ids[chosen]
    elif family in ['concept','temporal']:
        y=y+2*a*s*ystd*np.tanh(z[:,j])
        if family=='temporal':X=mean+(X-mean)*(1+.75*a*s)+.5*a*s*sd
    elif family=='domain':X=mean+(X-mean)*(1+.75*a*s)+.5*a*s*sd
    else:raise ValueError(family)
    return X.astype(np.float32),y,ids,info


def jobs():
    return [(track,m) for track in ['five','seven'] for m in old.frozen.cohort('regression',track)]


def plan():
    if (OUT/'plan.json').exists():return
    atomic_json(OUT/'plan.json',dict(status='FROZEN_BEFORE_PREDICTIONS',seed=0,n_estimators=1,budgets=BUDGETS,
        development_only=True,original_paper_replication=False,new_protocol=True,
        source_pool='unchanged v23 unique eligible TRAIN rows, capped8192; excludes frozen queries',
        source_statistics='each complete eligible source pool, fixed for all context budgets',
        nominal_strengths=[.2,.5,.8,1.],frozen_family_constants=True,
        corruption_generation='one full environment per dataset/family/rho; never regenerated per context budget',
        importance_density_ratio_cap=DENSITY_CAP,cap_chosen_without_query_metrics=True,
        nested_order='source response decile stratification, fixedseed0; no query label access',
        no_monotonicity_score_selection=True,no_query_metric_hyperparameter_search=True,
        resimulate_256_reference=True,clean_as_control=True,classification_unchanged=True,
        semisynthetic_regression_blocks=['clean','five'],metric_denominators='unchanged v22',
        checkpoint='official TabICL-v2, single estimator, no adaptation or finetuning',
        frozen_v22_sha256=old.sha(old.frozen.CONFIG),script_sha256=old.sha(__file__),Markdown_written=False))


def worker(shard,shards,only=None):
    import torch,warnings
    from run_tabicl import Predictor,metrics
    from seven_shifts import FAMILIES
    from tabicl_semisynthetic_five_v22 import corrupt_data
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0')
    selected=jobs()
    if only:selected=[(k,m) for k,m in selected if str(m['task_id']) in only.split(',')]
    for track,meta in selected[shard::shards]:
        try:
            pool=old.prepare('regression',track,meta);tid=int(meta['task_id'])
            grid=([('clean','original',0)]+[('five',f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]) if track=='five' else [('seven',f,r) for f in FAMILIES for r in [20,50,80,100]]
            shared_order=stratified_order(pool['y'],tid,0)
            for block,family,rho in grid:
                paths=[OUT/'predictions'/str(n)/block/str(tid)/f'{family}_{rho}.npz' for n in BUDGETS]
                if all(p.exists() and p.with_suffix('.json').exists() for p in paths):continue
                base,original=old.bank('regression',block,meta,family,rho)
                if block=='seven':
                    A,b,ids,diagnostics=seven_environment(pool,family,rho,base['X_target'])
                    order=stratified_order(pool['y'][ids],tid,0)
                else:
                    clean=base|dict(X_source=pool['X'].copy(),y_source=pool['y'].copy())
                    changed=clean if block=='clean' else corrupt_data(clean,family,rho,old.frozen.configuration()['regression_five']['families'][family]['parameters'])
                    A,b=changed['X_source'],changed['y_source'];ids=np.arange(len(b));order=shared_order;diagnostics={}
                affected=(np.any(A!=pool['X'][ids],axis=1)|(b!=pool['y'][ids]))
                env_hash={k:protocol.array_sha256(v) for k,v in dict(X=A,y=b,ids=ids,order=order).items()}
                env_record=dict(task_id=tid,dataset=meta['dataset'],block=block,family=family,rho=rho,
                    total_context_rows=len(b),unique_original_source_rows=len(np.unique(ids)),hashes=env_hash,
                    sampling=diagnostics,source_y_min=float(b.min()),source_y_max=float(b.max()))
                atomic_json(OUT/'environments'/block/str(tid)/f'{family}_{rho}.json',env_record)
                for budget,out in zip(BUDGETS,paths):
                    if out.exists() and out.with_suffix('.json').exists():continue
                    selected_ids=order[:budget]
                    data=base|dict(X_source=A[selected_ids],y_source=b[selected_ids])
                    hashes={k:protocol.array_sha256(v) for k,v in data.items()}
                    start=time.monotonic();reused=None
                    for small in BUDGETS:
                        if small>=budget:break
                        prior=OUT/'predictions'/str(small)/block/str(tid)/out.name
                        if prior.with_suffix('.json').exists() and json.loads(prior.with_suffix('.json').read_text())['input_hashes']==hashes:reused=prior;break
                    if reused:
                        with np.load(reused) as z:pred=z['prediction']
                    else:pred,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,random_state=0)
                    assert np.isfinite(pred).all()
                    old.npwrite(out,prediction=pred,y_target=data['y_target'],source_environment_positions=selected_ids,source_original_ids=pool['source_ids'][ids[selected_ids]])
                    atomic_json(out.with_suffix('.json'),dict(task='regression',block=block,task_id=tid,dataset=meta['dataset'],family=family,rho=rho,
                        requested_context=budget,actual_context=len(selected_ids),unique_original_source_rows=len(np.unique(ids[selected_ids])),
                        material_changed_fraction=float(affected[selected_ids].mean()),input_hashes=hashes,environment_hashes=env_hash,
                        scale=original['scale'],metrics=metrics(data['y_target'],pred,'regression',original['scale']),
                        elapsed_seconds=time.monotonic()-start,seed=0,n_estimators=1,semisynthetic=block in ['clean','five'],
                        new_development_protocol=True,adaptation=False,finetuning=False,reused_exact_inputs=str(reused) if reused else None))
            print('DATASET_COMPLETE',track,tid,meta['dataset'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{track}_{meta["task_id"]}.json',dict(error=error))


def report():
    rows=[]
    for p in (OUT/'predictions').glob('*/*/*/*.json'):
        if p.name.endswith('.tmp.json'):continue
        row=json.loads(p.read_text());rows.append({k:v for k,v in row.items() if k not in ['metrics','input_hashes','environment_hashes']}|row['metrics'])
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(OUT/'detail.csv',index=False)
    per=df.groupby(['requested_context','block','task_id','dataset'],as_index=False).agg(
        conditions=('family','size'),actual_context=('actual_context','first'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    per['complete']=per.conditions==per.block.map(dict(clean=1,five=20,seven=28));per.to_csv(OUT/'per_dataset.csv',index=False)
    avg=per[per.complete].groupby(['requested_context','block'],as_index=False).agg(datasets=('task_id','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    avg.to_csv(OUT/'averages.csv',index=False);print(avg.to_string(index=False))
    expected=(35*21+33*28)*4
    atomic_json(OUT/'progress.json',dict(completed_conditions=len(df),expected_conditions=expected,complete=len(df)==expected,
                                      errors=len(list((OUT/'errors').glob('*.json')))))


def audit():
    from run_tabicl import metrics
    from seven_shifts import FAMILIES
    failures=[];missing=[];verified=0
    for track,meta in jobs():
        grid=([('clean','original',0)]+[('five',f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]) if track=='five' else [('seven',f,r) for f in FAMILIES for r in [20,50,80,100]]
        for block,family,rho in grid:
            truth_meta=json.loads((old.frozen.OUT/'input_banks/regression'/block/str(meta['task_id'])/f'{family}_{rho}.json').read_text())
            prefix=None;environment_hashes=None
            for budget in BUDGETS:
                path=OUT/'predictions'/str(budget)/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                if not path.exists():missing.append(str(path));continue
                record=json.loads(path.with_suffix('.json').read_text())
                with np.load(path) as z:
                    p=z['prediction'];t=z['y_target'];positions=z['source_environment_positions'];rowids=z['source_original_ids']
                if prefix is not None and not np.array_equal(positions[:len(prefix)],prefix):failures.append([str(path),'not_nested'])
                prefix=positions
                if environment_hashes is not None and environment_hashes!=record['environment_hashes']:failures.append([str(path),'environment_changed'])
                environment_hashes=record['environment_hashes']
                for key in ['X_target','y_target']:
                    if record['input_hashes'][key]!=truth_meta['input_hashes'][key]:failures.append([str(path),key])
                if protocol.array_sha256(t)!=truth_meta['input_hashes']['y_target']:failures.append([str(path),'saved_truth'])
                if record['scale']!=truth_meta['scale']:failures.append([str(path),'denominator'])
                actual=metrics(t,p,'regression',record['scale'])
                if any(not np.isclose(actual[k],v,atol=1e-12,rtol=0) for k,v in record['metrics'].items()):failures.append([str(path),'metrics'])
                if record['seed']!=0 or record['n_estimators']!=1:failures.append([str(path),'configuration'])
                verified+=1
    atomic_json(OUT/'final_audit.json',dict(status='PASS' if not failures and not missing else 'FAILED_OR_INCOMPLETE',
        verified_conditions=verified,expected_conditions=(35*21+33*28)*4,missing=missing,failures=failures,
        nested_context_verified=True,unchanged_queries_and_metric_denominators=True,
        new_development_protocol=True,original_paper_replication=False,monotonicity_not_used_for_selection=True))
    print('AUDIT',verified,'missing',len(missing),'failures',failures[:5])
    if failures or missing:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report','audit'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--only')
    a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='worker':worker(a.shard,a.shards,a.only)
    elif a.phase=='report':report()
    else:audit()
