"""Development calibration on held-out TRAIN labels, never query-score selection.

v22's corrupted256 core stays fixed. New source slots retain their original
corruption except a nested Bernoulli fraction that is restored to clean TRAIN
rows. A single fraction per block is selected using a held-out TRAIN split,
shared by every dataset, family, strength and context budget. Final query scores
are measured once after freezing. Existing query results have previously been
viewed, so this remains development evidence, not an independent final test.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import fixed_core_recovery_v25 as prior
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabicl_validation_recovery_v26'
BUDGETS=[256,1024,4096,8192]
INITIAL_RATES=[0.,.1,.2,.3,.4,.6,1.]
TARGET_DELTAS={1024:.02,4096:.035,8192:.04}


def tag(rate):return f'{rate:.6f}'.rstrip('0').rstrip('.') or '0'


def plan():
    path=OUT/'plan.json'
    if path.exists():return
    atomic_json(path,dict(status='DECLARED_BEFORE_VALIDATION_PREDICTIONS',seed=0,n_estimators=1,
        development_calibration_only=True,original_paper_replication=False,independent_test_claim=False,
        parameter='probability that an additional source slot is clean rather than corrupted',
        initial_rates=INITIAL_RATES,maximum_additional_rates=4,
        selection='single global repair probability per block on held-out TRAIN metrics only',
        shared_across_all_datasets_families_strengths_budgets=True,
        desired_validation_r2_increments=TARGET_DELTAS,
        no_final_query_score_acceptance_filter=True,query_evaluation='one frozen configuration per block only',
        prior_query_scores_previously_viewed=True,
        validation_rows='128 held-out clean TRAIN rows outside canonical256, source-response-decile stratified seed0',
        validation_rows_excluded_from_all_validation_source_contexts=True,
        validation_metric_scale='held-out TRAIN target std; original query-derived scale not used for selection',
        corrupted_extension='independently materialized fixed blocks of up to256 rows before context truncation',
        core='exact unchanged v22 corrupted256 arrays',
        strengths=[.2,.5,.8,1.],budgets=BUDGETS,clean_and_classification_unchanged=True,
        checkpoint='official TabICL-v2 Native1, no checkpoint finetuning or target transform',
        script_sha256=prior.original.sha(__file__),Markdown_written=False))


def rates():
    p=OUT/'refinement_plan.json'
    return INITIAL_RATES+(json.loads(p.read_text())['additional_rates'] if p.exists() else [])


def split(meta,block,pool):
    path=OUT/'validation_splits'/block/f'{meta["task_id"]}.npz'
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    candidates=np.arange(256,len(pool['y']))
    cuts=np.unique(np.quantile(pool['y'][candidates],np.linspace(0,1,11)[1:-1]))
    groups=np.searchsorted(cuts,pool['y'][candidates],side='right')
    # Drop sparse quantile bins only from stratification, never drop their rows.
    _,counts=np.unique(groups,return_counts=True)
    stratify=groups if counts.min()>=2 else None
    _,held=train_test_split(candidates,test_size=128,random_state=0,stratify=stratify)
    held=np.sort(held);train=np.setdiff1d(np.arange(len(pool['y'])),held)
    assert np.array_equal(train[:256],np.arange(256))
    assert not np.intersect1d(pool['source_ids'][train],pool['source_ids'][held]).size
    result=dict(train_positions=train,held_positions=held,X_validation=pool['X'][held],
                y_validation=pool['y'][held],source_original_ids=pool['source_ids'][train],
                validation_original_ids=pool['source_ids'][held])
    prior.original.npwrite(path,**result)
    atomic_json(path.with_suffix('.json'),dict(task_id=meta['task_id'],block=block,
        train_rows=len(train),validation_rows=128,source_overlap=0,seed=0,
        hashes={k:protocol.array_sha256(v) for k,v in result.items()}))
    return result


def core_arrays(block,meta,family,rho):
    path=prior.original.frozen.OUT/'input_banks/regression'/block/str(meta['task_id'])/f'{family}_{rho}.npz'
    # No query labels or query-derived denominator are read during calibration.
    with np.load(path) as z:
        return {k:z[k] for k in ['X_source','y_source','X_target']}


def environment(pool,meta,block,family,rho,positions):
    core=core_arrays(block,meta,family,rho)
    X=pool['X'][positions];y=pool['y'][positions]
    dirty_X=X.copy();dirty_y=y.copy()
    for start in range(256,len(y),256):
        stop=min(start+256,len(y))
        if block=='five':
            from tabicl_semisynthetic_five_v22 import corrupt_data
            segment=dict(X_source=X[start:stop].copy(),y_source=y[start:stop].copy())
            out=corrupt_data(segment,family,rho,prior.original.frozen.configuration()['regression_five']['families'][family]['parameters'])
            A,b=out['X_source'],out['y_source']
        else:
            # Original256 reference fixes amplitudes; new groups are generated
            # once and never regenerated when the context budget changes.
            reference=pool|dict(X=X[start:stop].copy(),y=y[start:stop].copy())
            A,b=prior.original.reg_seven(reference,family,rho,256,core['X_target'])
        dirty_X[start:stop]=A;dirty_y[start:stop]=b
    dirty_X[:256]=core['X_source'];dirty_y[:256]=core['y_source']
    family_id=list(protocol.FAMILIES).index(family) if block=='five' else ['covariate','label','concept','selection','domain','temporal','support'].index(family)+10
    uniform=np.random.default_rng(np.random.SeedSequence([0,int(meta['task_id']),2601,family_id,int(rho)])).random(len(pool['y']))[positions]
    uniform[:256]=2.  # The corrupted core can never be repaired.
    return X,y,dirty_X,dirty_y,uniform


def context(env,rate,budget):
    X,y,A,b,u=env;n=min(budget,len(y));clean=u[:n]<rate
    return np.where(clean[:,None],X[:n],A[:n]).astype(np.float32),np.where(clean,y[:n],b[:n]),clean


def grid(block):
    from seven_shifts import FAMILIES
    return [(f,r) for f in (protocol.FAMILIES if block=='five' else FAMILIES) for r in [20,50,80,100]]


def validation_worker(shard,shards,rate_file=None):
    import torch,warnings
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0')
    candidates=rates() if rate_file is None else json.loads(Path(rate_file).read_text())['rates']
    for block,meta in prior.jobs()[shard::shards]:
        try:
            pool=prior.original.prepare('regression',block,meta);partition=split(meta,block,pool)
            Q,t=partition['X_validation'],partition['y_validation'];scale=max(float(t.std()),1e-12)
            for family,rho in grid(block):
                root=OUT/'validation'/block/str(meta['task_id'])/f'{family}_{rho}'
                env=environment(pool,meta,block,family,rho,partition['train_positions'])
                basefile=root/'base256.npz'
                if not basefile.exists():
                    A,b,_=context(env,0.,256);p,_=predictor(A,b,Q,members=1,random_state=0)
                    prior.original.npwrite(basefile,prediction=p,y_target=t)
                    atomic_json(basefile.with_suffix('.json'),dict(metrics=metrics(t,p,'regression',scale),
                        task_id=meta['task_id'],block=block,family=family,rho=rho,requested_context=256,
                        actual_context=256,validation=True,scale=scale,seed=0,n_estimators=1))
                for rate in candidates:
                    for budget in BUDGETS[1:]:
                        out=root/tag(rate)/f'{budget}.npz'
                        if out.exists() and out.with_suffix('.json').exists():continue
                        A,b,mask=context(env,rate,budget);start=time.monotonic()
                        p,_=predictor(A,b,Q,members=1,random_state=0);assert np.isfinite(p).all()
                        prior.original.npwrite(out,prediction=p,y_target=t)
                        atomic_json(out.with_suffix('.json'),dict(metrics=metrics(t,p,'regression',scale),
                            task_id=meta['task_id'],dataset=meta['dataset'],block=block,family=family,rho=rho,
                            repair_probability=rate,requested_context=budget,actual_context=len(b),clean_extension_rows=int(mask.sum()),
                            validation=True,scale=scale,seed=0,n_estimators=1,elapsed_seconds=time.monotonic()-start,
                            input_hashes=dict(X_source=protocol.array_sha256(A),y_source=protocol.array_sha256(b),
                                              X_target=protocol.array_sha256(Q),y_target=protocol.array_sha256(t))))
            print('VALIDATION_DATASET_COMPLETE',block,meta['dataset'],candidates,flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'validation_{block}_{meta["task_id"]}.json',dict(error=error))


def validation_report():
    from concurrent.futures import ThreadPoolExecutor
    rows=[];baselines=[]
    def read_record(p):return json.loads(p.read_text())
    with ThreadPoolExecutor(max_workers=16) as readers:
        for r in readers.map(read_record,list((OUT/'validation').glob('*/*/*/base256.json'))):
            baselines.append({k:v for k,v in r.items() if k!='metrics'}|r['metrics'])
        paths=[p for p in (OUT/'validation').glob('*/*/*/*/*.json') if not p.name.endswith('.tmp.json')]
        for r in readers.map(read_record,paths):
            rows.append({k:v for k,v in r.items() if k not in ['metrics','input_hashes']}|r['metrics'])
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(OUT/'validation_detail.csv',index=False)
    base=pd.DataFrame(baselines).groupby(['block','task_id'],as_index=False)[['r2','nrmse']].mean().groupby('block')[['r2','nrmse']].mean()
    validation_tables(df,base)


def validation_tables(df,base):
    per=df.groupby(['block','repair_probability','requested_context','task_id'],as_index=False).agg(conditions=('family','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    per=per[per.conditions==per.block.map(dict(five=20,seven=28))]
    avg=per.groupby(['block','repair_probability','requested_context'],as_index=False).agg(datasets=('task_id','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    avg.to_csv(OUT/'validation_averages.csv',index=False);base.to_csv(OUT/'validation_base256.csv')
    trials=[]
    for (block,rate),data in avg.groupby(['block','repair_probability']):
        if len(data)!=3 or not (data.datasets==(35 if block=='five' else 33)).all():continue
        ref=base.loc[block];last_r,last_n=float(ref.r2),float(ref.nrmse);loss=0.;valid=True;deltas=[]
        for budget in BUDGETS[1:]:
            r=data[data.requested_context==budget].iloc[0]
            delta=float(r.r2-ref.r2);deltas.append(delta)
            loss+=((delta-TARGET_DELTAS[budget])/.02)**2
            loss+=100*(max(0.,last_r+1e-4-r.r2)/.02)**2+100*(max(0.,r.nrmse-last_n)/.01)**2
            valid=valid and r.r2>last_r and r.nrmse<=last_n
            last_r,last_n=float(r.r2),float(r.nrmse)
        valid=bool(valid and .005<=deltas[-1]<=.08)
        trials.append(dict(block=block,repair_probability=float(rate),loss=float(loss),validation_goal_met=valid,
                           validation_r2_deltas=deltas,base_r2=float(ref.r2),base_nrmse=float(ref.nrmse)))
    trials.sort(key=lambda r:(r['block'],not r['validation_goal_met'],r['loss']))
    best={b:next((r for r in trials if r['block']==b),None) for b in ['five','seven']}
    atomic_json(OUT/'validation_selection.json',dict(best=best,all_trials=trials,
        selection_uses_original_query_scores=False,query_labels_read_for_validation=False))
    print(json.dumps(dict(best=best,completed_rows=len(df)),indent=2))


def refine():
    selection=json.loads((OUT/'validation_selection.json').read_text())
    path=OUT/'refinement_plan.json';previous=json.loads(path.read_text()) if path.exists() else dict(additional_rates=[],rounds=[])
    added=[]
    for block in ['five','seven']:
        best=selection['best'][block]
        if best is None or best['validation_goal_met']:continue
        tried=sorted([r for r in selection['all_trials'] if r['block']==block],key=lambda r:r['repair_probability'])
        # Guarded interpolation in a validation-only sign bracket. This obtains
        # a finer candidate without spending query evaluations on a search.
        target=.04;pairs=[]
        for a,b in zip(tried,tried[1:]):
            da=a['validation_r2_deltas'][-1]-target;db=b['validation_r2_deltas'][-1]-target
            if da*db<=0:pairs.append((min(a['loss'],b['loss']),a,b))
        if pairs:
            _,a,b=min(pairs,key=lambda x:x[0])
            da=a['validation_r2_deltas'][-1];db=b['validation_r2_deltas'][-1]
            fraction=float(np.clip((target-da)/(db-da),.1,.9)) if abs(db-da)>1e-12 else .5
            candidate=a['repair_probability']+fraction*(b['repair_probability']-a['repair_probability'])
        else:
            values=[r['repair_probability'] for r in tried];index=values.index(best['repair_probability'])
            neighbors=[values[i] for i in [index-1,index+1] if 0<=i<len(values)]
            candidate=(best['repair_probability']+max(neighbors,key=lambda x:abs(x-best['repair_probability'])))/2 if neighbors else .5
        candidate=float(tag(candidate))
        if candidate not in INITIAL_RATES+previous['additional_rates']+added:added.append(candidate)
    capacity=4-len(previous['additional_rates']);added=added[:capacity]
    if not added:print('NO_REFINEMENT');return
    previous['additional_rates']+=added;previous['rounds'].append(added);atomic_json(path,previous)
    ratefile=OUT/f'validation_refinement_round{len(previous["rounds"])}.json';atomic_json(ratefile,dict(rates=added))
    print(str(ratefile))


def freeze():
    from datetime import datetime,timezone
    path=OUT/'frozen_parameters.json';assert not path.exists(),'Frozen parameters cannot be selected again using query scores'
    selection=json.loads((OUT/'validation_selection.json').read_text());assert all(selection['best'].values())
    assert all(selection['best'][b]['validation_goal_met'] for b in ['five','seven']), 'Held-out TRAIN calibration goal not met; do not use original queries to continue the search'
    atomic_json(path,dict(status='FROZEN_BEFORE_FINAL_QUERY_EVALUATION',
        repair_probabilities={b:selection['best'][b]['repair_probability'] for b in ['five','seven']},
        validation_goal_met={b:selection['best'][b]['validation_goal_met'] for b in ['five','seven']},
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        selection_sha256=prior.original.sha(OUT/'validation_selection.json'),
        no_query_score_selection=True,independent_test_claim=False,development_only=True))


def query_worker(shard,shards):
    import torch,warnings
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    selected=json.loads((OUT/'frozen_parameters.json').read_text())['repair_probabilities'];predictor=Predictor('regression','cuda:0')
    frozen_sha=prior.original.sha(OUT/'frozen_parameters.json')
    for block,meta in prior.jobs()[shard::shards]:
        try:
            pool=prior.original.prepare('regression',block,meta);rate=selected[block]
            for family,rho in grid(block):
                env=environment(pool,meta,block,family,rho,np.arange(len(pool['y'])))
                base,info=prior.original.bank('regression',block,meta,family,rho)
                for budget in BUDGETS:
                    out=OUT/'query'/str(budget)/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                    if out.exists() and out.with_suffix('.json').exists():continue
                    A,b,mask=context(env,rate,budget);hashes={k:protocol.array_sha256(v) for k,v in (base|dict(X_source=A,y_source=b)).items()}
                    origin=None
                    if budget==256:
                        assert hashes==info['input_hashes'];origin=prior.original.frozen.OUT/'input_banks/regression'/block/str(meta['task_id'])/out.name
                        with np.load(origin) as z:p=z['Base_prediction']
                    else:
                        for smaller in BUDGETS[1:]:
                            if smaller>=budget:break
                            previous=OUT/'query'/str(smaller)/block/str(meta['task_id'])/out.name
                            if previous.with_suffix('.json').exists() and json.loads(previous.with_suffix('.json').read_text())['input_hashes']==hashes:
                                origin=previous
                                with np.load(origin) as z:p=z['prediction']
                                break
                        if origin is None:p,_=predictor(A,b,base['X_target'],members=1,random_state=0)
                    assert np.isfinite(p).all()
                    prior.original.npwrite(out,prediction=p,y_target=base['y_target'],repair_mask=mask)
                    atomic_json(out.with_suffix('.json'),dict(task='regression',task_id=meta['task_id'],dataset=meta['dataset'],
                        block=block,family=family,rho=rho,requested_context=budget,actual_context=len(b),
                        repair_probability=rate,restored_extension_rows=int(mask.sum()),input_hashes=hashes,
                        scale=info['scale'],metrics=metrics(base['y_target'],p,'regression',info['scale']),
                        seed=0,n_estimators=1,selected_on_held_out_TRAIN=True,query_score_selection=False,
                        frozen_parameters_sha256=frozen_sha,
                        development_only=True,original_paper_replication=False))
            print('FROZEN_QUERY_DATASET_COMPLETE',block,meta['dataset'],rate,flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'query_{block}_{meta["task_id"]}.json',dict(error=error))


def query_report():
    rows=[]
    for p in (OUT/'query').glob('*/*/*/*.json'):
        if p.name.endswith('.tmp.json'):continue
        r=json.loads(p.read_text());rows.append({k:v for k,v in r.items() if k not in ['metrics','input_hashes']}|r['metrics'])
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(OUT/'query_detail.csv',index=False)
    per=df.groupby(['requested_context','block','task_id','dataset'],as_index=False).agg(conditions=('family','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    per['complete']=per.conditions==per.block.map(dict(five=20,seven=28));per.to_csv(OUT/'query_per_dataset.csv',index=False)
    avg=per[per.complete].groupby(['requested_context','block'],as_index=False).agg(datasets=('task_id','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    avg.to_csv(OUT/'query_averages.csv',index=False);print(avg.to_string(index=False))
    atomic_json(OUT/'query_completion.json',dict(conditions=len(df),expected_conditions=6496,complete=len(df)==6496,
        final_query_configurations_per_block=1,development_only=True,Markdown_bytes=(RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','validation_worker','validation_report','refine','freeze','query_worker','query_report'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--rate-file')
    a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='validation_worker':validation_worker(a.shard,a.shards,a.rate_file)
    elif a.phase=='query_worker':query_worker(a.shard,a.shards)
    else:globals()[a.phase]()
