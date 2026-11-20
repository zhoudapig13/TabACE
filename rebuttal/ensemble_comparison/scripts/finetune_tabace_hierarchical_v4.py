"""Source-held-out DEVELOPMENT selection and residual Specific integration.

All GPU evidence is reused; no final-query scores select any policy. Shared
selects observed-context or explicitly supervised reference banks on held-out
TRAIN rows. Specific refines Shared through a separately calibrated residual
gate, with Shared itself as a genuine held-out selection candidate.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import finetune_tabace_main_v1 as main
import finetune_tabace_reference_v3 as ref
from prepare_data import RUN,atomic_json

OUT=RUN/'results/finetune_tabace_hierarchical_v4'
_base_tensors=main.fusion.training_tensors


def folder(job):return OUT/'jobs'/job['task']/job['track']/str(job['meta']['task_id'])


def tensors(cells,m):
    data=_base_tensors(cells,m)
    if cells[0]['task']=='classification':
        factors=[]
        for c in cells:
            y=c['y'];labels,counts=np.unique(y,return_counts=True)
            lookup={int(k):len(y)/(len(labels)*count) for k,count in zip(labels,counts)}
            factors.extend(lookup[int(k)] for k in y)
        data['weights']*=torch.as_tensor(factors,dtype=torch.float64)
    return data


def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_HELD_OUT_POLICY_SELECTION',seed=0,context_cap=256,
        benchmark_sha256=main.sha(main.frozen.CONFIG),benchmark_unchanged=True,pretrained_Shared_Specific_not_run=True,
        checkpoints='classification:v2, regression:v1; unchanged across all compared combinations',
        adaptation='explicit supervised development extension, not archived paper-method replication',
        Shared_candidates=['original observed-context five roles','clean FT TRAIN reference five roles with canonical reference replacing bootstrap'],
        Shared_calibration_selection='75%/25% disjoint source calibration original row IDs',
        label_shift_resampled_duplicate_ids='reconstruct exact operator resampling IDs, keep copies in one split',
        Shared_grid=dict(gate=['entropy','entropy_or_disagreement'],regularization=[.01,.1]),
        Specific='Shared prediction is the anchor; five original family-specific alternatives; learned residual gate',
        Specific_grid=dict(gain=[.25,.5,1.],regularization=[.01,.1]),Specific_fallback='exact Shared prediction',
        loss='class-balanced NLL or normalized MSE on held-out calibration TRAIN queries',
        final_query_labels_for_policy_selection=False,original_variants_and_control_results_preserved=True,
        incremental_goal='report Base→FT and FT→combination; no score editing or stopping based on benchmark queries',
        nominal_calls=dict(FT=1,FT_Shared=6,FT_Specific=11),
        regression_clean_five_semisynthetic=True,classification_five_query_feature_perturbation=True,
        time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),script_sha256=main.sha(__file__),Markdown_written=False))


def read_cell(path,reference=False):
    record=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as z:a={k:z[k] for k in z.files}
    c=main.cell(a,record)
    if reference:
        c['bank']=c['bank'].copy();c['bank'][0]=a['reference']
    return c


def row_map(job,split,pool,family,rho):
    positions=split['cal'].copy()
    if job['task']=='classification' and job['track']=='seven' and family=='label':
        _,y,_=main.flat(pool);t=y[positions]
        p=main.frozen.configuration()['classification_seven']['generator_parameters'];s=rho/100.
        original=float(t.mean());extreme=.9 if original>=.5 else .1
        if p['label_prior_direction']==-1:extreme=1-extreme
        prior=original+s*p['label_prior_displacement_coefficient']*(extreme-original)
        rng=np.random.default_rng(0);labels=(rng.random(len(t))<prior).astype(int)
        indices=np.asarray([rng.choice(np.flatnonzero(t==label)) for label in labels])
        positions=positions[indices]
    return positions


def calibration_split(job,pool,split):
    _,y,_=main.flat(pool);positions=split['cal'];rng=np.random.default_rng(4401)
    if job['task']=='classification':
        hold=[]
        for label in np.unique(y[positions]):
            group=positions[y[positions]==label]
            n=min(len(group)-1,max(1,int(round(.25*len(group))))) if len(group)>1 else 0
            hold.extend(rng.permutation(group)[:n].tolist())
        hold=np.asarray(hold,dtype=int)
    else:hold=rng.permutation(positions)[:max(8,len(positions)//4)]
    assert len(hold)>0 and len(hold)<len(positions)
    return hold


def subset(c,mask):
    assert mask.any()
    return c|dict(base=c['base'][mask],bank=c['bank'][:,mask],y=c['y'][mask])


def score(cells,policy):
    losses=[]
    for c in cells:
        p=main.fusion.predict(c,policy)
        if c['task']=='classification':
            nll=-np.log(np.clip(p[np.arange(len(c['y'])),c['y'].astype(int)],1e-12,1.))
            losses.append(float(np.mean([nll[c['y']==label].mean() for label in np.unique(c['y'])])))
        else:losses.append(float(np.mean((p-c['y'])**2))/max(c['meta']['scale']**2,1e-12))
    return float(np.mean(losses))


def fit(cells,task,m,gate,reg,gain=1.):
    main.fusion.GATE_VARIANT=gate
    return main.fusion.fit(cells,task,m,gain,reg)


def anchored(cells,shared,m=5):
    return [c|dict(base=main.fusion.predict(c,shared),bank=c['bank'][5:5+m]) for c in cells]


def run(job):
    dest=folder(job);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completion.json').exists():return
    assert (ref.root(job)/'completion.json').exists()
    pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
    with np.load(ref.train_root(job)/'split.npz') as z:split={k:z[k] for k in z.files}
    held=calibration_split(job,pool,split)
    all_cells={};trains={};valids={}
    for kind,source_root in [('observed',ref.train_root(job)),('reference',ref.root(job))]:
        cells=[];train=[];valid=[]
        for path in sorted((source_root/'calibration').glob('*.npz')):
            c=read_cell(path,reference=kind=='reference')
            ids=row_map(job,split,pool,c['meta']['family'],c['meta']['rho'])
            mask=np.isin(ids,held)
            assert mask.any() and (~mask).any()
            train.append(subset(c,~mask));valid.append(subset(c,mask));cells.append(c)
        all_cells[kind]=cells;trains[kind]=train;valids[kind]=valid
    baseline=dict(kind='base');task=job['task'];gates=['entropy','entropy_or_disagreement'] if task=='classification' else ['entropy']
    best=dict(kind='observed',gate='entropy',reg=None,policy=baseline,loss=score(valids['observed'],baseline));trials=[]
    start=time.perf_counter()
    for kind in ['observed','reference']:
        for gate in gates:
            for reg in [.01,.1]:
                policy=fit(trains[kind],task,5,gate,reg)
                value=score(valids[kind],policy)
                trials.append(dict(bank=kind,gate=gate,regularization=reg,held_TRAIN_loss=value))
                if value<best['loss']-1e-5:best=dict(kind=kind,gate=gate,reg=reg,policy=policy,loss=value)
    kind=best['kind'];full=all_cells[kind]
    shared=baseline if best['reg'] is None else fit(full,task,5,best['gate'],best['reg'])
    specific={};specific_trials={}
    for family in ['original']+main.families(job['track']):
        t=[c for c in trains[kind] if c['meta']['family']==family]
        v=[c for c in valids[kind] if c['meta']['family']==family]
        t=anchored(t,best['policy']);v=anchored(v,best['policy'])
        selected=dict(policy=baseline,reg=None,gain=0.,loss=score(v,baseline));attempts=[]
        for gain in [.25,.5,1.]:
            for reg in [.01,.1]:
                policy=fit(t,task,5,best['gate'],reg,gain)
                value=score(v,policy);attempts.append(dict(gain=gain,reg=reg,held_TRAIN_loss=value))
                if value<selected['loss']-1e-5:selected=dict(policy=policy,reg=reg,gain=gain,loss=value)
        group=anchored([c for c in full if c['meta']['family']==family],shared)
        specific[family]=baseline if selected['reg'] is None else fit(group,task,5,best['gate'],selected['reg'],selected['gain'])
        specific_trials[family]=dict(selected={k:v for k,v in selected.items() if k!='policy'},attempts=attempts)
    atomic_json(dest/'policies.json',dict(status='FROZEN_BEFORE_QUERY_SCORING',selected_bank=kind,Shared=shared,Specific=specific,
        shared_trials=trials,specific_trials=specific_trials,selection_original_row_positions=held.tolist(),
        selection_rows_disjoint_fit_rows=True,duplicate_resampled_original_rows_grouped=True,original_query_labels_used=False,
        policy_fit_and_selection_seconds=time.perf_counter()-start,development_revision=True))
    rows=[]
    for block in (['clean','five'] if job['track']=='five' else ['seven']):
        for family,rho in main.contexts.case_grid(block):
            path=(ref.train_root(job) if kind=='observed' else ref.root(job))/'query'/block/f'{family}_{rho}.npz'
            c=read_cell(path,reference=kind=='reference');sp=main.fusion.predict(c,shared)
            special=c|dict(base=sp,bank=c['bank'][5:])
            predictions={'FT':c['base'],'FT+Shared_dev':sp,'FT+Specific_dev':main.fusion.predict(special,specific[family])}
            for method,p in predictions.items():
                rows.append(dict(task=task,block=block,task_id=job['meta']['task_id'],data_id=job['meta']['data_id'],dataset=job['meta']['dataset'],
                    family=family,rho=rho,method=method,selected_bank=kind,gate_variant=best['gate'],
                    **main.metrics(c['y'],p,task,c['meta']['scale'])))
    pd.DataFrame(rows).to_csv(dest/'query_detail.csv',index=False)
    atomic_json(dest/'completion.json',dict(status='COMPLETE',task=task,track=job['track'],task_id=job['meta']['task_id'],
        selected_bank=kind,gate_variant=best['gate'],held_TRAIN_rows=len(held),final_query_labels_for_selection=False))


def worker(shard,shards):
    torch.set_num_threads(1);main.fusion.training_tensors=tensors
    for job in main.jobs()[shard::shards]:
        try:
            print('HIER_START',job['task'],job['track'],job['meta']['task_id'],flush=True)
            run(job);print('HIER_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(folder(job)/'error.json',dict(job=job,error=error))
        gc.collect()


def report():
    files=sorted((OUT/'jobs').glob('*/*/*/query_detail.csv'))
    if not files:return
    df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);base=[]
    for _,r in df[df.method=='FT'].iterrows():
        p=main.frozen.OUT/'input_banks'/r['task']/r['block']/str(r['task_id'])/f"{r['family']}_{r['rho']}.json"
        d=json.loads(p.read_text());base.append({k:r[k] for k in ['task','block','task_id','data_id','dataset','family','rho']}|dict(method='Base (cached, paired)')|d['metrics'])
    df=pd.concat([pd.DataFrame(base),df],ignore_index=True);df.to_csv(OUT/'query_detail.csv',index=False)
    rows=[];deltas=[]
    for (task,block),g in df.groupby(['task','block']):
        fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse'];expected=len(main.frozen.cohort(task,block))
        for method,h in g.groupby('method'):
            rows.append(dict(task=task,block=block,method=method,datasets=h.task_id.nunique(),expected_datasets=expected,
                complete=h.task_id.nunique()==expected,**{f:float(h[f].mean()) for f in fields}))
        for metric in fields:
            table=g.groupby(['task_id','method'])[metric].mean().unstack();direction=-1 if metric=='nrmse' else 1
            one=float((table.FT-table['Base (cached, paired)']).mean())*direction
            for method in ['FT+Shared_dev','FT+Specific_dev']:
                two=float((table[method]-table.FT).mean())*direction
                deltas.append(dict(task=task,block=block,metric=metric,method=method,Base_to_FT_gain=one,
                    FT_to_combination_gain=two,second_increment_at_least_first=bool(two>=one)))
    pd.DataFrame(rows).to_csv(OUT/'averages.csv',index=False);pd.DataFrame(deltas).to_csv(OUT/'increment_comparison.csv',index=False)
    done=len(list((OUT/'jobs').glob('*/*/*/completion.json')))
    atomic_json(OUT/'progress.json',dict(completed_jobs=done,expected_jobs=len(main.jobs()),complete=done==len(main.jobs())))
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print('HIER_PROGRESS',done,'/',len(main.jobs()),flush=True)


def launch_after_v3():
    plan();pending={int(w['gpu']):int(w['pid']) for w in json.loads((ref.OUT/'workers.json').read_text())}
    children=[];(OUT/'logs').mkdir(exist_ok=True)
    while pending:
        for shard,oldpid in list(pending.items()):
            try:os.kill(oldpid,0);continue
            except ProcessLookupError:pass
            log=(OUT/'logs'/f'worker{shard}.log').open('a',buffering=1)
            p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(shard),'--shards','8'],
                env=os.environ,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();children.append(dict(worker=shard,pid=p.pid));atomic_json(OUT/'workers.json',children)
            print('HIER_WORKER_STARTED',shard,p.pid,flush=True);del pending[shard]
        if pending:time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report','launch_after_v3'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='worker':worker(a.shard,a.shards)
    elif a.phase=='report':report()
    else:launch_after_v3()
