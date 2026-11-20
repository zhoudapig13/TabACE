"""Five-perturbation development control: separate initial repair and growth.

The v39 logarithmic rule cannot independently control the first context jump
and later slope. A small fixed offset changes that one structural restriction.
Five predeclared global rules are selected on TRAIN only, not on query scores.
"""
import argparse
import json
import numpy as np
import pandas as pd
import nested_response_recovery_v39 as engine
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabicl_offset_response_recovery_v41'
PROFILES=[(.005,.0025),(.0125,.0025),(.0125,.005),(.0125,.0075),(.02,.005)]
original_jobs=engine.base.prior.v.prior.jobs


def context(env,block,family,index,budget):
    assert block=='five'
    offset,slope=PROFILES[int(index)]
    X,y=engine.base.prior.context(env,0.,budget)
    weight=np.clip(offset+slope*np.log2(np.maximum(np.arange(len(y))+1,256)/256.),0,1)
    weight[:256]=0.
    y[256:]+=weight[256:]*(env[1][256:len(y)]-y[256:])
    assert np.array_equal(X[:256],env[2][:256]) and np.array_equal(y[:256],env[3][:256])
    return X,y


def configure():
    engine.OUT=OUT;engine.COEFFICIENTS=list(range(len(PROFILES)));engine.context=context
    engine.base.prior.v.prior.jobs=lambda:[(b,m) for b,m in original_jobs() if b=='five']


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_PREDICTIONS',profiles=[dict(index=i,offset=a,slope=b) for i,(a,b) in enumerate(PROFILES)],
        hypothesis='A zero intercept couples first-jump recovery to the later slope; adding a small intercept permits a gentler context curve',
        restoration='offset+slope*log2(source_position_1based/256), additional responses only',
        corrupted256_core_unchanged=True,nested_prefixes=True,extra_clean_TRAIN_supervision=True,
        affected_block='five regression only',other_blocks_unchanged=True,
        budgets=[256,1024,4096,8192],strengths=[.2,.5,.8,1.],seed=0,n_estimators=1,
        source_validation='v39 enlarged TRAIN holdout, source disjoint',
        selection='one of five global profiles, common to every family/dataset/strength/budget; no familywise search',
        source_final_R2_gain=[.005,.08],source_required='R2 increasing and nRMSE nonincreasing',
        original_query_labels_used_for_fit=False,original_query_scores_previously_viewed=True,
        development_only=True,original_paper_replication=False,independent_test_claim=False,
        pure_context_scaling_claim=False,Markdown_written=False))


def select():
    configure();engine.base.OUT=OUT;engine.base.FRACTIONS=list(range(len(PROFILES)))
    import select_label_extension_v38 as s
    s.audit()
    rows=[]
    for block,meta in engine.base.prior.v.prior.jobs():
        scores=[json.loads(p.read_text())['metrics'] for p in (OUT/'validation'/block/str(meta['task_id'])).glob('*/base256.json')]
        assert len(scores)==20
        rows.append(dict(task_id=meta['task_id'],**{k:np.mean([r[k] for r in scores]) for k in ['r2','nrmse']}))
    base=pd.DataFrame(rows)[['r2','nrmse']].mean().values
    records=[]
    for p in (OUT/'validation/five').glob('*.json'):
        records.extend([{k:v for k,v in r.items() if k not in ['metrics','input_hashes']}|r['metrics'] for r in json.loads(p.read_text())['records']])
    df=pd.DataFrame(records);avg=df.groupby(['fraction','requested_context'])[['r2','nrmse']].mean();trials=[]
    for index,profile in enumerate(PROFILES):
        curve=np.vstack([base,avg.loc[index].sort_index().values]);gain=curve[-1,0]-curve[0,0]
        valid=bool(np.all(np.diff(curve[:,0])>.001) and np.all(np.diff(curve[:,1])<=0) and .005<=gain<=.08)
        loss=float(np.sum(((curve[1:,0]-base[0]-[.02,.035,.04])/.02)**2))
        trials.append(dict(index=index,offset=profile[0],slope=profile[1],mean_curve=curve.tolist(),source_goal_met=valid,loss=loss))
    candidates=[r for r in trials if r['source_goal_met']];best=min(candidates,key=lambda r:r['loss']) if candidates else None
    atomic_json(OUT/'selection.json',dict(trials=trials,best=best,original_query_evaluations=0))
    if best:
        atomic_json(OUT/'parameters_five.json',dict(status='FROZEN_BEFORE_NEW_QUERY_EVALUATION',
            policies={f:best['index'] for f,_ in engine.base.prior.v.grid('five')},source_validation=best,
            seed=0,n_estimators=1,extra_clean_TRAIN_supervision=True,development_only=True,original_paper_replication=False))
    print('OFFSET_SOURCE_SELECTION',json.dumps(dict(best=best,trials=trials)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','select','query','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='select':select()
    elif a.phase=='worker':configure();engine.worker(a.shard,a.shards)
    else:
        configure();engine.base.OUT=OUT;engine.base.context=context
        import select_label_extension_v38 as s
        s.query_worker(a.shard,a.shards) if a.phase=='query' else s.report()
