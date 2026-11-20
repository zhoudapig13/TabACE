"""A nested, position-defined response-restoration development benchmark.

Uses a larger, response-stratified TRAIN holdout to reduce small-validation
variance. Neither context composition nor restoration changes when the budget
is truncated: the i-th source row always receives the same transformation.
The result is a recovery protocol with changing information quality, not a pure
sample-size effect and not a reproduction of the original paper benchmark.
"""
import argparse
import json
import os
import traceback
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name,'1')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import label_extension_recovery_v38 as base
from prepare_data import RUN, atomic_json, protocol

OUT = RUN / 'results/tabicl_nested_response_recovery_v39'
COEFFICIENTS = [0., .0025, .005, .01, .02]


def context(env, block, family, coefficient, budget):
    X, y = base.prior.context(env,0.,budget)
    weight = np.clip(coefficient * np.log2(np.maximum(np.arange(len(y))+1,256)/256.),0,1)
    weight[:256] = 0.
    y[256:] += weight[256:] * (env[1][256:len(y)]-y[256:])
    if block == 'seven' and family in ['domain','temporal']:
        lo,hi=base.tail.bounds(env[2][:256],4.)
        X[256:]=np.clip(X[256:],lo,hi)
        lo,hi=base.tail.bounds(env[3][:256],4.)
        y[256:]=np.clip(y[256:],lo,hi)
    assert np.array_equal(X[:256],env[2][:256]) and np.array_equal(y[:256],env[3][:256])
    return X,y


def split(pool, block, meta):
    path=OUT/'validation_splits'/block/f'{meta["task_id"]}.npz'
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    old=base.prior.v.split(meta,block,pool)
    total=min(512,max(128,(len(pool['y'])-256)//4))
    candidates=np.setdiff1d(np.arange(256,len(pool['y'])),old['held_positions'])
    extra=total-len(old['held_positions'])
    if extra:
        quantiles=np.unique(np.quantile(pool['y'][candidates],np.linspace(0,1,11)[1:-1]))
        groups=np.searchsorted(quantiles,pool['y'][candidates],side='right')
        _,counts=np.unique(groups,return_counts=True)
        stratify=groups if counts.min()>=2 and extra>=len(counts) and len(candidates)-extra>=len(counts) else None
        _,additional=train_test_split(candidates,test_size=extra,random_state=0,stratify=stratify)
        held=np.sort(np.r_[old['held_positions'],additional])
    else:held=old['held_positions']
    train=np.setdiff1d(np.arange(len(pool['y'])),held)
    assert np.array_equal(train[:256],np.arange(256))
    assert np.intersect1d(pool['source_ids'][train],pool['source_ids'][held]).size==0
    arrays=dict(train_positions=train,held_positions=held,X_validation=pool['X'][held],y_validation=pool['y'][held],
        source_original_ids=pool['source_ids'][train],validation_original_ids=pool['source_ids'][held])
    base.prior.v.prior.original.npwrite(path,**arrays)
    return arrays


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_PREDICTIONS',seed=0,n_estimators=1,
        coefficients=COEFFICIENTS,budgets=[256,1024,4096,8192],strengths=[.2,.5,.8,1.],
        hypothesis='A constant response restoration fraction does not offset residual large-context degradation; a fixed source-position law can make later observations more informative',
        response_restoration='coefficient*log2(max(source_position_1based,256)/256), clipped [0,1]',
        source_order='frozen seed0 source-pool order; never query-score selected',
        unchanged_corrupted_core256=True,nested_context_prefixes=True,
        validation='128 to512 disjoint TRAIN rows; retain the earlier128 and add response-stratified rows',
        selection='global coefficient preferred; otherwise one per family, common to all datasets/strengths/budgets',
        desired_final_validation_R2_gain=[.005,.08],extra_clean_TRAIN_supervision=True,
        larger_context_has_more_label_restoration=True,pure_context_scaling_claim=False,
        new_development_protocol=True,original_paper_replication=False,independent_test_claim=False,
        original_queries_previously_viewed=True,original_query_labels_used_for_selection=False,
        classification_and_clean_unchanged=True,Markdown_written=False))


def worker(shard,shards):
    import torch,warnings
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    model=Predictor('regression','cuda:0');failures=0
    for block,meta in base.prior.v.prior.jobs()[shard::shards]:
        out=OUT/'validation'/block/f'{meta["task_id"]}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        try:
            pool=base.prior.v.prior.original.prepare('regression',block,meta)
            partition=split(pool,block,meta);Q,t=partition['X_validation'],partition['y_validation'];scale=max(float(t.std()),1e-12)
            bank=dict(y_validation=t);records=[];base_bank=dict(y_validation=t);calls=0
            for family,rho in base.prior.v.grid(block):
                env=base.prior.v.environment(pool,meta,block,family,rho,partition['train_positions'])
                A,b=context(env,block,family,0.,256)
                p,_=model(A,b,Q,members=1,random_state=0);calls+=1
                key=f'{family}_{rho}';base_bank[key]=p
                base_path=OUT/'validation'/block/str(meta['task_id'])/key/'base256.json'
                atomic_json(base_path,dict(metrics=metrics(t,p,'regression',scale),scale=scale))
                cache={}
                for coefficient in COEFFICIENTS:
                    for budget in base.prior.v.BUDGETS[1:]:
                        A,b=context(env,block,family,coefficient,budget)
                        hashes=dict(X_source=protocol.array_sha256(A),y_source=protocol.array_sha256(b))
                        pair=tuple(hashes.values());reused=pair in cache
                        if reused:p=cache[pair]
                        else:p,_=model(A,b,Q,members=1,random_state=0);calls+=1
                        assert np.isfinite(p).all();cache[pair]=p
                        key=f'{family}_{rho}_{base.prior.tag(coefficient)}_{budget}';bank[key]=p
                        records.append(dict(task_id=meta['task_id'],dataset=meta['dataset'],block=block,family=family,rho=rho,
                            fraction=coefficient,requested_context=budget,actual_context=len(b),validation_rows=len(t),prediction_key=key,
                            scale=scale,metrics=metrics(t,p,'regression',scale),core256_unchanged=True,original_query_loaded=False,
                            extra_clean_TRAIN_supervision=True,identical_input_reuse=reused,
                            input_hashes=hashes|dict(X_target=protocol.array_sha256(Q),y_target=protocol.array_sha256(t))))
            base.prior.v.prior.original.npwrite(OUT/'base_validation'/block/f'{meta["task_id"]}.npz',**base_bank)
            base.prior.v.prior.original.npwrite(out,**bank)
            atomic_json(out.with_suffix('.json'),dict(records=records,new_TFM_calls=calls))
            print('NESTED_RESPONSE_VALIDATION_COMPLETE',block,meta['dataset'],len(t),calls,flush=True)
        except Exception:
            failures+=1;error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{block}_{meta["task_id"]}.json',dict(error=error))
    atomic_json(OUT/'workers'/f'{shard}.json',dict(status='COMPLETE' if not failures else 'FAILED',failures=failures))
    if failures:raise SystemExit(1)


def configure():
    base.OUT=OUT;base.FRACTIONS=COEFFICIENTS;base.context=context


def audit_select():
    import select_label_extension_v38 as selection
    configure();selection.audit()
    rows=[]
    for block,meta in base.prior.v.prior.jobs():
        paths=(OUT/'validation'/block/str(meta['task_id'])).glob('*/base256.json')
        scores=[json.loads(p.read_text())['metrics'] for p in paths]
        rows.append(dict(block=block,task_id=meta['task_id'],**{k:np.mean([r[k] for r in scores]) for k in ['r2','nrmse']}))
    pd.DataFrame(rows).groupby('block')[['r2','nrmse']].mean().to_csv(OUT/'validation_base256.csv')
    base.prior.v.OUT=OUT
    selection.select()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','select','query','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='worker':worker(a.shard,a.shards)
    elif a.phase=='plan':plan()
    elif a.phase=='select':audit_select()
    else:
        configure();import select_label_extension_v38 as selection
        selection.query_worker(a.shard,a.shards) if a.phase=='query' else selection.report()
