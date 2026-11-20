"""Source-validation calibration of additional domain/temporal shift amplitude.

Only the new source groups change amplitude; the v22 corrupted256 core is
identical. All other shift families use the unchanged v26 zero-repair source
environment. This is explicitly another DEVELOPMENT benchmark protocol.
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
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabicl_affine_validation_v27'
INITIAL=[1.,.999,.997,.99,.97,.9,0.]


def tag(gamma):return f'{gamma:.8f}'.rstrip('0').rstrip('.') or '0'


def plan():
    if (OUT/'plan.json').exists():return
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_AMPLITUDE_VALIDATION',
        reason='v26 five-corruption validation goal met; no single sparse-repair probability found for all seven-shift budgets',
        hypothesis='Additional domain/temporal contexts introduce increasing affine mismatch on heavy-tail tasks',
        parameter='one amplitude multiplier for domain and temporal ADDITIONAL source groups only',
        initial_factors=INITIAL,maximum_refinements=4,
        shared_across_datasets_strengths_budgets=True,all_other_families_unchanged=True,
        core256_unchanged=True,nominal_strengths=[.2,.5,.8,1.],
        selection='held-out TRAIN only, same128 source-disjoint rows as v26',
        query_labels_for_selection=False,no_original_query_runs_yet=True,
        validation_goal='strictly increasing R2 and nonincreasing nRMSE; final R2 gain0.005..0.08',
        refinement_final_gain_target=.075,
        target_reason='Zero-repair first-step validation gain is already0.0718; maintain the predeclared0.08 total-gain upper limit',
        development_only=True,independent_test_claim=False,prior_query_scores_previously_viewed=True,
        checkpoint='official TabICL-v2 Native1, seed0, no finetuning',Markdown_written=False))


def environment(pool,meta,family,rho,positions,gamma):
    if family not in ['domain','temporal']:return v.environment(pool,meta,'seven',family,rho,positions)
    from calibrated_seven_protocol import amplitude
    from seven_shifts import standardize,affine,PARAMETERS
    core=v.core_arrays('seven',meta,family,rho);X=pool['X'][positions];y=pool['y'][positions]
    A=X.copy();b=y.copy();reference=np.vstack([pool['H0'],pool['R0'],pool['X0']])
    j=int(np.argmax(pool['X0'].var(0)));base=v.prior.original.frozen.configuration()['regression_seven']['family_amplitude_multipliers'][family]
    with amplitude(base*gamma):
        for start in range(256,len(y),256):
            stop=min(start+256,len(y));segment=X[start:stop];s=rho/100.
            A[start:stop]=affine(segment,reference,s).astype(np.float32)
            if family=='temporal':b[start:stop]=y[start:stop]+PARAMETERS['regression_response_drift_sd']*s*pool['y0'].std()*np.tanh(standardize(segment,reference)[:,j])
    A[:256]=core['X_source'];b[:256]=core['y_source']
    return X,y,A,b,np.full(len(y),2.)


def worker(shard,shards,factor_file):
    import warnings,torch
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    factors=json.loads(Path(factor_file).read_text())['factors'];predictor=Predictor('regression','cuda:0')
    jobs=[m for block,m in v.prior.jobs() if block=='seven']
    for meta in jobs[shard::shards]:
        pool=v.prior.original.prepare('regression','seven',meta);split=v.split(meta,'seven',pool)
        Q,t=split['X_validation'],split['y_validation'];scale=max(float(t.std()),1e-12)
        needed=[g for g in factors if not (OUT/'validation'/str(meta['task_id'])/f'{tag(g)}.json').exists()]
        if not needed:continue
        banks={g:dict(y_validation=t) for g in needed};records={g:[] for g in needed}
        for family,rho in v.grid('seven'):
            unchanged_env=None
            if family not in ['domain','temporal']:unchanged_env=v.environment(pool,meta,'seven',family,rho,split['train_positions'])
            cached={};old_predictions={}
            for gamma in needed:
                env=unchanged_env if unchanged_env is not None else environment(pool,meta,family,rho,split['train_positions'],gamma)
                for budget in v.BUDGETS[1:]:
                    A,b,_=v.context(env,0.,budget);ah=protocol.array_sha256(A);bh=protocol.array_sha256(b);key=(ah,bh)
                    origin=None
                    if key in cached:p=cached[key];origin='identical_inputs'
                    elif family not in ['domain','temporal'] or gamma==1.:
                        path=v.OUT/'validation/seven'/str(meta['task_id'])/f'{family}_{rho}'/'0'/f'{budget}.npz'
                        prior_record=json.loads(path.with_suffix('.json').read_text());assert prior_record['input_hashes']['X_source']==ah and prior_record['input_hashes']['y_source']==bh
                        with np.load(path) as z:p=z['prediction']
                        origin=str(path)
                    else:p,_=predictor(A,b,Q,members=1,random_state=0)
                    assert np.isfinite(p).all();cached[key]=p;bk=f'{family}_{rho}_{budget}';banks[gamma][bk]=p
                    records[gamma].append(dict(task_id=meta['task_id'],dataset=meta['dataset'],family=family,rho=rho,
                        requested_context=budget,actual_context=len(b),affine_factor=gamma,repair_probability=0.,prediction_key=bk,
                        scale=scale,metrics=metrics(t,p,'regression',scale),
                        input_hashes=dict(X_source=ah,y_source=bh,X_target=protocol.array_sha256(Q),y_target=protocol.array_sha256(t)),
                        validation=True,seed=0,n_estimators=1,exact_input_reuse=origin))
        for gamma in needed:
            out=OUT/'validation'/str(meta['task_id'])/f'{tag(gamma)}.npz'
            v.prior.original.npwrite(out,**banks[gamma]);atomic_json(out.with_suffix('.json'),dict(records=records[gamma]))
        print('AFFINE_VALIDATION_COMPLETE',meta['dataset'],needed,flush=True)


def report():
    rows=[]
    for path in (OUT/'validation').glob('*/*.json'):
        if path.name.endswith('.tmp.json'):continue
        rows += [{k:x for k,x in r.items() if k not in ['metrics','input_hashes']}|r['metrics'] for r in json.loads(path.read_text())['records']]
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(OUT/'validation_detail.csv',index=False)
    per=df.groupby(['affine_factor','requested_context','task_id'],as_index=False).agg(conditions=('family','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    per=per[per.conditions==28]
    avg=per.groupby(['affine_factor','requested_context'],as_index=False).agg(datasets=('task_id','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    avg.to_csv(OUT/'validation_averages.csv',index=False)
    ref=pd.read_csv(v.OUT/'validation_base256.csv').set_index('block').loc['seven'];trials=[]
    for gamma,data in avg.groupby('affine_factor'):
        if len(data)!=3 or not (data.datasets==33).all():continue
        last_r,last_n=float(ref.r2),float(ref.nrmse);valid=True;loss=0.;deltas=[]
        for budget in v.BUDGETS[1:]:
            r=data[data.requested_context==budget].iloc[0];delta=float(r.r2-ref.r2);deltas.append(delta)
            valid=valid and r.r2>last_r and r.nrmse<=last_n
            loss+=((delta-v.TARGET_DELTAS[budget])/.02)**2+100*(max(0.,last_r+1e-4-r.r2)/.02)**2+100*(max(0.,r.nrmse-last_n)/.01)**2
            last_r,last_n=float(r.r2),float(r.nrmse)
        valid=bool(valid and .005<=deltas[-1]<=.08)
        trials.append(dict(affine_factor=float(gamma),validation_goal_met=valid,loss=float(loss),validation_r2_deltas=deltas,
            base_r2=float(ref.r2),base_nrmse=float(ref.nrmse)))
    trials.sort(key=lambda r:(not r['validation_goal_met'],r['loss']))
    atomic_json(OUT/'validation_selection.json',dict(best=trials[0] if trials else None,all_trials=trials,query_labels_used=False))
    print(json.dumps(dict(best=trials[0] if trials else None,trials=len(trials)),indent=2))


def refine():
    selection=json.loads((OUT/'validation_selection.json').read_text());old=OUT/'refinement_plan.json'
    previous=json.loads(old.read_text()) if old.exists() else dict(factors=[])
    assert len(previous['factors'])<4
    trials=sorted(selection['all_trials'],key=lambda r:r['affine_factor']);pairs=[]
    for a,b in zip(trials,trials[1:]):
        da=a['validation_r2_deltas'][-1]-.075;db=b['validation_r2_deltas'][-1]-.075
        if da*db<=0:pairs.append((a,b))
    if not pairs:raise RuntimeError('No validation sign bracket for desired final increment')
    a,b=min(pairs,key=lambda pair:pair[1]['affine_factor']-pair[0]['affine_factor'])
    da,db=a['validation_r2_deltas'][-1],b['validation_r2_deltas'][-1]
    fraction=float(np.clip((.075-da)/(db-da),.1,.9)) if abs(db-da)>1e-12 else .5
    candidate=float(tag(a['affine_factor']+fraction*(b['affine_factor']-a['affine_factor'])))
    if candidate in INITIAL+previous['factors']:candidate=float(tag((a['affine_factor']+b['affine_factor'])/2))
    previous['factors'].append(candidate);atomic_json(old,previous)
    out=OUT/f'refine{len(previous["factors"])}_factors.json';atomic_json(out,dict(factors=[candidate]));print(str(out))


def audit():
    from run_tabicl import metrics
    failures=[];count=0;reconstructed=0
    for meta in [m for block,m in v.prior.jobs() if block=='seven']:
        pool=v.prior.original.prepare('regression','seven',meta);split=v.split(meta,'seven',pool)
        Q,t=split['X_validation'],split['y_validation'];scale=max(float(t.std()),1e-12)
        for path in (OUT/'validation'/str(meta['task_id'])).glob('*.json'):
            if path.name.endswith('.tmp.json'):continue
            records=json.loads(path.read_text())['records']
            with np.load(path.with_suffix('.npz')) as bank:
                if not np.array_equal(bank['y_validation'],t):failures.append([str(path),'truth'])
                for r in records:
                    measured=metrics(t,bank[r['prediction_key']],'regression',scale)
                    if any(not np.isclose(measured[k],value,atol=1e-12,rtol=0) for k,value in r['metrics'].items()):failures.append([str(path),'metric'])
                    if r['input_hashes']['X_target']!=protocol.array_sha256(Q) or r['input_hashes']['y_target']!=protocol.array_sha256(t):failures.append([str(path),'partition'])
                    count+=1
            changed=[r for r in records if r['family']=='domain' and r['rho']==20]
            if changed:
                env=environment(pool,meta,'domain',20,split['train_positions'],changed[0]['affine_factor'])
                for r in changed:
                    A,b,_=v.context(env,0.,r['requested_context'])
                    if protocol.array_sha256(A)!=r['input_hashes']['X_source'] or protocol.array_sha256(b)!=r['input_hashes']['y_source']:failures.append([str(path),'context'])
                    reconstructed+=1
    atomic_json(OUT/'validation_audit.json',dict(status='PASS' if not failures else 'FAIL',verified_predictions=count,
        reconstructed_contexts=reconstructed,failures=failures,query_evaluations=0,development_only=True))
    print('AFFINE_VALIDATION_AUDIT',count,'failures',failures[:5])
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report','refine','audit']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--factor-file');a=p.parse_args()
    if a.phase=='worker':worker(a.shard,a.shards,a.factor_file)
    else:globals()[a.phase]()
