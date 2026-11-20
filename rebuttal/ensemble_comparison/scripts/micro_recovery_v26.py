"""Train-validation-only micro-rate refinement, with bundled resumable artifacts.

Reuses predictions only for byte-identical source inputs. No original query
labels are read and no original query inference runs in this script.
"""
import argparse
import json
import os
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import validation_recovery_v26 as v
from prepare_data import atomic_json,protocol

MICRO_RATES=[.00001,.00004,.00016,.00064,.00002,.00008,.00032,.00128]


def plan():
    path=v.OUT/'micro_refinement_plan.json'
    if path.exists():return
    atomic_json(path,dict(status='DECLARED_BEFORE_MICRO_RATE_PREDICTIONS',
        previous_stage='11 rates on held-out TRAIN, goal unmet, 55216 predictions audited',
        original_queries_not_evaluated=True,additional_stage_budget=8,planned_rates=MICRO_RATES,
        batches=[MICRO_RATES[:4],MICRO_RATES[4:]],objective_unchanged=True,
        shared_rate_across_datasets_families_strengths_budgets=True,
        query_gate='both validation blocks must meet the predeclared small-increment criterion',
        unchanged_core_environment_seed_and_validation_splits=True,
        exact_input_reuse=True,development_calibration_only=True))


def worker(shard,shards,rate_file):
    import warnings,torch
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    rates=json.loads(Path(rate_file).read_text())['rates'];predictor=Predictor('regression','cuda:0')
    for block,meta in v.prior.jobs()[shard::shards]:
        pool=v.prior.original.prepare('regression',block,meta);split=v.split(meta,block,pool)
        Q,t=split['X_validation'],split['y_validation'];scale=max(float(t.std()),1e-12)
        needed=[]
        for rate in rates:
            path=v.OUT/'micro_validation'/block/str(meta['task_id'])/f'{v.tag(rate)}.npz'
            if not (path.exists() and path.with_suffix('.json').exists()):needed.append(rate)
        if not needed:continue
        banks={rate:dict(y_validation=t) for rate in needed};records={rate:[] for rate in needed}
        for family,rho in v.grid(block):
            env=v.environment(pool,meta,block,family,rho,split['train_positions']);cache={}
            for rate in sorted(needed):
                for budget in v.BUDGETS[1:]:
                    A,b,mask=v.context(env,rate,budget)
                    ah=protocol.array_sha256(A);bh=protocol.array_sha256(b);key=(ah,bh)
                    origin=None
                    if key in cache:p=cache[key];origin='identical_inputs_in_this_environment'
                    elif not mask.any():
                        path=v.OUT/'validation'/block/str(meta['task_id'])/f'{family}_{rho}'/'0'/f'{budget}.npz'
                        previous=json.loads(path.with_suffix('.json').read_text())
                        assert previous['input_hashes']['X_source']==ah and previous['input_hashes']['y_source']==bh
                        with np.load(path) as z:p=z['prediction']
                        origin=str(path)
                    else:p,_=predictor(A,b,Q,members=1,random_state=0)
                    cache[key]=p;assert np.isfinite(p).all()
                    bank_key=f'{family}_{rho}_{budget}';banks[rate][bank_key]=p
                    records[rate].append(dict(block=block,task_id=meta['task_id'],dataset=meta['dataset'],family=family,rho=rho,
                        repair_probability=rate,requested_context=budget,actual_context=len(b),validation=True,scale=scale,
                        clean_extension_rows=int(mask.sum()),seed=0,n_estimators=1,prediction_key=bank_key,
                        input_hashes=dict(X_source=ah,y_source=bh,X_target=protocol.array_sha256(Q),y_target=protocol.array_sha256(t)),
                        metrics=metrics(t,p,'regression',scale),exact_input_reuse=origin))
        for rate in needed:
            out=v.OUT/'micro_validation'/block/str(meta['task_id'])/f'{v.tag(rate)}.npz'
            v.prior.original.npwrite(out,**banks[rate]);atomic_json(out.with_suffix('.json'),dict(records=records[rate]))
        print('MICRO_VALIDATION_COMPLETE',block,meta['dataset'],needed,flush=True)


def report():
    original=pd.read_csv(v.OUT/'validation_detail.csv')
    rows=[]
    for path in (v.OUT/'micro_validation').glob('*/*/*.json'):
        if path.name.endswith('.tmp.json'):continue
        for r in json.loads(path.read_text())['records']:
            rows.append({k:x for k,x in r.items() if k not in ['metrics','input_hashes']}|r['metrics'])
    df=pd.concat([original,pd.DataFrame(rows)],ignore_index=True)
    df=df.drop_duplicates(['block','repair_probability','requested_context','task_id','family','rho'])
    df.to_csv(v.OUT/'validation_detail.csv',index=False)
    base=pd.read_csv(v.OUT/'validation_base256.csv').set_index('block')
    v.validation_tables(df,base)


def audit():
    from run_tabicl import metrics
    fail=[];count=0;contexts=0
    for block,meta in v.prior.jobs():
        pool=v.prior.original.prepare('regression',block,meta);split=v.split(meta,block,pool)
        Q,t=split['X_validation'],split['y_validation'];scale=max(float(t.std()),1e-12)
        for path in (v.OUT/'micro_validation'/block/str(meta['task_id'])).glob('*.json'):
            if path.name.endswith('.tmp.json'):continue
            records=json.loads(path.read_text())['records']
            with np.load(path.with_suffix('.npz')) as z:
                if not np.array_equal(z['y_validation'],t):fail.append([str(path),'truth'])
                for r in records:
                    pred=z[r['prediction_key']];measured=metrics(t,pred,'regression',scale)
                    if any(not np.isclose(measured[k],value,atol=1e-12,rtol=0) for k,value in r['metrics'].items()):fail.append([str(path),'metrics'])
                    if r['input_hashes']['X_target']!=protocol.array_sha256(Q) or r['input_hashes']['y_target']!=protocol.array_sha256(t):fail.append([str(path),'partition'])
                    count+=1
            # Verify reconstructed contexts across all budgets for one condition
            # per bundle, in addition to every prediction/metric above.
            first=records[0];env=v.environment(pool,meta,block,first['family'],first['rho'],split['train_positions'])
            for r in records:
                if r['family']!=first['family'] or r['rho']!=first['rho']:continue
                A,b,_=v.context(env,r['repair_probability'],r['requested_context'])
                if protocol.array_sha256(A)!=r['input_hashes']['X_source'] or protocol.array_sha256(b)!=r['input_hashes']['y_source']:fail.append([str(path),'context'])
                contexts+=1
    atomic_json(v.OUT/'micro_validation_audit.json',dict(status='PASS' if not fail else 'FAIL',verified_predictions=count,
        reconstructed_contexts=contexts,failures=fail,selection_uses_query_labels=False,development_only=True))
    print('MICRO_AUDIT',count,'failures',fail[:5])
    if fail:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report','audit']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--rate-file');a=p.parse_args()
    if a.phase=='worker':worker(a.shard,a.shards,a.rate_file)
    else:globals()[a.phase]()
