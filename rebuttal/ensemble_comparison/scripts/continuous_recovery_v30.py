"""TRAIN-only diagnostic of continuous versus sparse context restoration.

The v26 Bernoulli repair uses very few clean rows, so individual samples can
dominate its effect. Here all additional rows receive a fixed, small fraction
of their pre-corruption TRAIN features/labels instead. The original corrupted
256 core is immutable. This is a separately declared development protocol;
the original query labels and scores are never used in this search.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import validation_recovery_v26 as v
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabicl_continuous_recovery_v30'
INITIAL=[0.,.00001,.0001,.001,.005,.01,.05,.2,1.]


def tag(value):return f'{value:.8f}'.rstrip('0').rstrip('.') or '0'


def plan():
    path=OUT/'plan.json';assert not path.exists(),'Do not silently change a declared registry'
    atomic_json(path,dict(status='DECLARED_BEFORE_VALIDATION',seed=0,n_estimators=1,
        hypothesis='Sparse Bernoulli replacement has discontinuous high-leverage effects; fractional source restoration may transfer more stably',
        parameter='fraction of clean pre-corruption TRAIN value restored in every additional source row',
        initial_fractions=INITIAL,one_global_fraction_per_block=True,
        shared_across_datasets_families_severities_budgets=True,
        context_budgets=v.BUDGETS,nominal_strengths=[.2,.5,.8,1.],
        corrupted_core256_unchanged=True,source_validation_split='same disjoint128 TRAIN rows as v26',
        extra_clean_TRAIN_information=True,original_query_labels_used_for_selection=False,
        no_original_query_evaluation_in_this_script=True,development_only=True,
        partial_restoration_is_a_new_protocol=True,original_paper_replication=False,
        desired_validation_final_R2_gain=[.005,.08],
        required_validation_curve='strictly increasing R2 and nonincreasing nRMSE',
        clean_and_classification_unchanged=True,Markdown_written=False))


def context(env,fraction,budget):
    X,y,A,b,_=env;n=min(budget,len(y))
    if fraction==0.:return A[:n].astype(np.float32).copy(),b[:n].copy()
    C=A[:n].astype(np.float64)+fraction*(X[:n]-A[:n])
    t=b[:n]+fraction*(y[:n]-b[:n])
    if fraction==1.:C=X[:n].copy();t=y[:n].copy()
    C[:256]=A[:256];t[:256]=b[:256]
    return C.astype(np.float32),t


def worker(shard,shards):
    import warnings,torch
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0');failures=0
    for block,meta in v.prior.jobs()[shard::shards]:
        try:
            pool=v.prior.original.prepare('regression',block,meta);split=v.split(meta,block,pool)
            Q,t=split['X_validation'],split['y_validation'];scale=max(float(t.std()),1e-12)
            records=[];predictions=dict(y_validation=t)
            output=OUT/'validation'/block/f'{meta["task_id"]}.npz'
            if output.exists() and output.with_suffix('.json').exists():continue
            for family,rho in v.grid(block):
                env=v.environment(pool,meta,block,family,rho,split['train_positions'])
                for fraction in INITIAL:
                    for budget in v.BUDGETS[1:]:
                        A,b=context(env,fraction,budget)
                        assert np.array_equal(A[:256],env[2][:256]) and np.array_equal(b[:256],env[3][:256])
                        old=v.OUT/'validation'/block/str(meta['task_id'])/f'{family}_{rho}'/v.tag(fraction)/f'{budget}.npz'
                        if fraction in [0.,1.]:
                            prior=json.loads(old.with_suffix('.json').read_text())
                            assert prior['input_hashes']['X_source']==protocol.array_sha256(A)
                            assert prior['input_hashes']['y_source']==protocol.array_sha256(b)
                            with np.load(old) as z:p=z['prediction']
                        else:p,_=predictor(A,b,Q,members=1,random_state=0)
                        assert np.isfinite(p).all();key=f'{family}_{rho}_{tag(fraction)}_{budget}';predictions[key]=p
                        records.append(dict(task_id=meta['task_id'],dataset=meta['dataset'],block=block,family=family,rho=rho,
                            restoration_fraction=fraction,requested_context=budget,actual_context=len(b),prediction_key=key,
                            metrics=metrics(t,p,'regression',scale),scale=scale,
                            input_hashes=dict(X_source=protocol.array_sha256(A),y_source=protocol.array_sha256(b),
                                              X_target=protocol.array_sha256(Q),y_target=protocol.array_sha256(t)),
                            original_core256_preserved=True,clean_TRAIN_information_used=True,validation=True,seed=0,n_estimators=1))
            v.prior.original.npwrite(output,**predictions);atomic_json(output.with_suffix('.json'),dict(records=records))
            print('CONTINUOUS_VALIDATION_DATASET_COMPLETE',block,meta['dataset'],flush=True)
        except Exception:
            failures+=1;error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{block}_{meta["task_id"]}.json',dict(error=error))
    if failures:raise SystemExit(1)


def report():
    rows=[]
    for path in (OUT/'validation').glob('*/*.json'):
        if '.tmp.' in path.name:continue
        rows += [{k:x for k,x in r.items() if k not in ['metrics','input_hashes']}|r['metrics'] for r in json.loads(path.read_text())['records']]
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(OUT/'validation_detail.csv',index=False)
    per=df.groupby(['block','restoration_fraction','requested_context','task_id'],as_index=False).agg(conditions=('family','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    per=per[per.conditions==per.block.map(dict(five=20,seven=28))]
    avg=per.groupby(['block','restoration_fraction','requested_context'],as_index=False).agg(datasets=('task_id','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    avg.to_csv(OUT/'validation_averages.csv',index=False)
    base=pd.read_csv(v.OUT/'validation_base256.csv').set_index('block');trials=[]
    for (block,fraction),group in avg.groupby(['block','restoration_fraction']):
        group=group.sort_values('requested_context')
        if len(group)!=3 or not (group.datasets==(35 if block=='five' else 33)).all():continue
        r=np.r_[base.loc[block].r2,group.r2.values];n=np.r_[base.loc[block].nrmse,group.nrmse.values]
        delta=float(r[-1]-r[0]);monotone=bool(np.all(np.diff(r)>0) and np.all(np.diff(n)<=0))
        trials.append(dict(block=block,fraction=float(fraction),validation_goal_met=bool(monotone and .005<=delta<=.08),
            validation_r2=r.tolist(),validation_nrmse=n.tolist(),final_R2_gain=delta))
    atomic_json(OUT/'validation_selection.json',dict(trials=trials,
        status='SOURCE_VALIDATION_GOAL_MET' if any(t['validation_goal_met'] for t in trials) else 'TARGET_NOT_MET_OR_INCOMPLETE',
        original_query_predictions=0,original_query_labels_used=False))
    print(json.dumps(trials,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    worker(a.shard,a.shards) if a.phase=='worker' else globals()[a.phase]()
