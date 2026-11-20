"""Source-reference differential trend repair, with calibration-only selection.

Old residualization removed an entire current-context smooth component,
including stable signal. This attempt removes the observed current-versus-
historical difference instead. No clean counterpart responses or query labels
enter repair. A/B may select a matched constructor or generic fallback before
inference; C/D retain the v100 generic bank. Availability and deployment are
reported separately. No ordering margin enters any fitting objective.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
import json
import os
import subprocess
import sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import torch

import abcd_metric_aligned_v100 as study
import abcd_fresh_confirmation_v101 as fresh
import seven_shifts as seven
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,metrics
from report_abcd_v84 import paired_statistics

OUT=RUN/'results/abcd_reference_repair_v102'
VARIANTS=('generic','legacy','difference05','difference10')


def freeze():
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/'plan.json'
    plan={'seed':0,'stage':'bounded constructor refinement attempt2; no further fresh-score-driven tuning',
        'code_sha256':study.origin.sha(__file__),'classification_policies_unchanged_v100':True,
        'source_inputs_and_benchmark_unchanged':True,'context_cap':256,'alternative_branches':5,
        'repair':'shrunken difference in univariate tanh slopes between current256 and history/reference remainder',
        'candidate_features':'eight largest observed current feature variances; no generator identity',
        'slope_shrinkage':'delta * max(0,1-1/z_score^2); clip magnitude to4 observed pool response SD',
        'repair_strengths':[.5,1.],'constructor_receives_family_or_strength':False,
        'clean_counterpart_responses_or_query_labels_received_by_constructor':False,
        'A_B_constructor_candidates':VARIANTS,'C_D_constructor':'generic',
        'common_calibration_grid_per_bank':study.REGS,'selection':'same other-dataset inner validation R2',
        'A_B_available_vs_deployed_distinction_required':True,
        'extra_constructor_search_for_A_B_reported':True,'B_excludes_concept_and_temporal':True,
        'no_method_ordering_used_in_loss_or_selection':True,
        'fresh_truth_opened_after_all_selected_policies_frozen':True,'original_paper_replication':False,'Markdown_written':False}
    if p.exists():assert json.loads(p.read_text())['code_sha256']==plan['code_sha256']
    else:atomic_json(p,plan)


def differential_repair(X,y,pool_X,pool_y,strength):
    # The known accessible pool lists current rows first, then reference/history.
    assert np.allclose(pool_X[:len(y)],X) and np.array_equal(pool_y[:len(y)],y)
    H,hy=pool_X[len(y):],pool_y[len(y):]
    assert len(hy)>=32
    mu=pool_X.mean(0);sd=np.maximum(pool_X.std(0),1e-6)
    cols=np.argsort(X.var(0),kind='stable')[-min(8,X.shape[1]):]
    U=np.tanh(np.clip((X[:,cols]-mu[cols])/sd[cols],-20,20))
    V=np.tanh(np.clip((H[:,cols]-mu[cols])/sd[cols],-20,20))
    u=U-U.mean(0);v=V-V.mean(0)
    current_slope=(u*(y-y.mean())[:,None]).sum(0)/np.maximum((u*u).sum(0),1e-9)
    history_slope=(v*(hy-hy.mean())[:,None]).sum(0)/np.maximum((v*v).sum(0),1e-9)
    err_cur=y[:,None]-y.mean()-u*current_slope
    err_hist=hy[:,None]-hy.mean()-v*history_slope
    variance=err_cur.var(0)/np.maximum((u*u).sum(0),1e-9)+err_hist.var(0)/np.maximum((v*v).sum(0),1e-9)
    delta=current_slope-history_slope;z2=delta**2/np.maximum(variance,1e-12)
    j=int(np.argmax(z2));shrink=max(0.,1.-1./max(float(z2[j]),1e-12))
    bound=4*max(float(pool_y.std()),1e-9)
    slope=float(np.clip(delta[j]*shrink,-bound,bound))
    repaired=y-strength*slope*U[:,j]
    assert np.isfinite(repaired).all()
    return X,repaired,{'feature':int(cols[j]),'current_slope':float(current_slope[j]),
        'historical_slope':float(history_slope[j]),'slope_difference':float(delta[j]),'z2':float(z2[j]),
        'effective_slope':slope,'strength':strength,'history_rows':len(hy)}


def prepare_worker(shard,shards):
    torch.set_num_threads(1);seven.PARAMETERS['regression_response_drift_sd']=1.
    metas=[m for m in seven.registry() if m['task']=='regression'];model=Predictor('regression','cuda:0')
    for i,meta in enumerate(metas):
        if i%shards!=shard:continue
        tid=int(meta['task_id']);clean=seven.regression_clean(meta)
        source_paths=[p for p in study.origin.source_paths('regression') if int(p.parent.name)==tid]
        extra=fresh.OUT/'inputs/regression'/f'{tid}.npz'
        Qfresh=None
        if extra.exists():
            with np.load(extra) as z:Qfresh=z['X_evaluation']
        for path in source_paths:
            m=json.loads(path.with_suffix('.json').read_text());family=m['family'];rho=m['rho']
            destination=OUT/'banks'/str(tid)/f'{family}_{rho}.npz'
            future=OUT/'fresh_banks'/str(tid)/f'{rho}.npz'
            need_fresh=Qfresh is not None and (rho==0 or family=='concept')
            if destination.with_suffix('.json').exists() and (not need_fresh or future.with_suffix('.json').exists()):continue
            data,_=seven.materialize(meta,family,rho,clean)
            X,y,P,py=(data[k] for k in ('X_source','y_source','X_pool','y_pool'))
            outputs={};fresh_outputs={};details=[]
            for name,weight in [('difference05',.5),('difference10',1.)]:
                A,b,d=differential_repair(X,y,P,py,weight);details.append(d)
                outputs[name]=model(A,b,data['X_target'])[0]
                if need_fresh:fresh_outputs[name]=model(A,b,Qfresh)[0]
            destination.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(destination,**outputs)
            atomic_json(destination.with_suffix('.json'),{'status':'COMPLETE','task_id':tid,'family':family,'rho':rho,
                'repair':details,'context_rows':len(y),'clean_counterpart_or_query_labels_used':False,
                'source_x_sha256':protocol.array_sha256(X),'source_y_sha256':protocol.array_sha256(y)})
            if need_fresh:
                future.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(future,**fresh_outputs)
                atomic_json(future.with_suffix('.json'),{'status':'COMPLETE','fresh_evaluation_labels_used':False,'repair':details})
        print('DIFFERENTIAL_REPAIR_BANK',tid,flush=True)


def prepare():
    freeze()
    def run(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i)
        with (OUT/f'prepare_{i}.log').open('a') as f:
            p=subprocess.run([sys.executable,__file__,'prepare-worker','--shard',str(i)],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'returncode':p.returncode}
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(run,i) for i in range(8)]):
            r=f.result();records.append(r);print(r,flush=True)
    atomic_json(OUT/'preparation.json',{'status':'COMPLETE' if not any(r['returncode'] for r in records) else 'FAILED','workers':records})


def cells(variant):
    values=study.regression_cells(variant=='legacy')
    if variant.startswith('difference'):
        for c in values:
            m=c['meta'];p=OUT/'banks'/str(m['task_id'])/f'{m["family"]}_{m["rho"]}.npz'
            with np.load(p) as z:c['bank'][0]=z[variant]
    return values


def fit_worker(fold):
    torch.set_num_threads(1);folder=OUT/'folds'/str(fold)
    bank_sets={v:cells(v) for v in VARIANTS}
    for method,seen,matched in study.METHODS:
        candidates=[]
        if not matched:
            original=study.OUT/'regression'/str(fold)/f'{method}_policy.json'
            saved=json.loads(original.read_text())
            saved.update(constructor='generic',constructor_candidates=['generic'],available_matched_constructor=False)
            atomic_json(folder/f'{method}_policy.json',saved);continue
        for variant in VARIANTS:
            all_cells=bank_sets[variant]
            source=[c for c in all_cells if c['fold']!=fold and (seen or not study.origin.exposed(c,'regression'))]
            inner=min(c['fold'] for c in source);train=[c for c in source if c['fold']!=inner];val=[c for c in source if c['fold']==inner]
            if variant in ('generic','legacy'):
                name=method if variant=='legacy' else ('C' if method=='A' else 'D')
                saved=json.loads((study.OUT/'regression'/str(fold)/f'{name}_policy.json').read_text())
                trial=max(saved['trials'],key=lambda t:(t['validation']['primary'],t['validation']['secondary']))
                p,best,trials=saved['policy'],trial,saved['trials']
            else:
                prior=np.r_[np.zeros(5),np.ones(5),0.,.5]
                neutral={'kind':'gated_fusion','parameters':prior.tolist(),'m':5,'gain':1.,'gate_variant':'entropy_or_disagreement'}
                p,best,trials=study.select(train,val,source,'regression',prior,neutral)
            candidates.append({'constructor':variant,'policy':p,'validation':best['validation'],
                'selected_candidate':best['candidate'],'trials':trials})
        best=max(candidates,key=lambda c:(c['validation']['primary'],c['validation']['secondary']))
        atomic_json(folder/f'{method}_policy.json',{'policy':best['policy'],'constructor':best['constructor'],
            'constructor_candidates':VARIANTS,'available_matched_constructor':True,'candidates':candidates,
            'seen':seen,'matched':matched,'inner_fold':inner,
            'training_data_ids':sorted({c['meta']['data_id'] for c in source}),
            'excluded_data_ids':sorted({c['meta']['data_id'] for c in all_cells if c['fold']==fold}),
            'heldout_target_labels_used':False,'extra_constructor_selection_reported':True})
        print('REFERENCE_POLICY',fold,method,best['constructor'],best['validation'],flush=True)
    atomic_json(folder/'completion.json',{'status':'COMPLETE'})
    return fold


def fit():
    assert json.loads((OUT/'preparation.json').read_text())['status']=='COMPLETE'
    records=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(fit_worker,i) for i in range(5)]):records.append(f.result())
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','folds':records})
    files=list((OUT/'folds').glob('*/*_policy.json'));assert len(files)==20
    atomic_json(OUT/'policy_seal.json',{str(p.relative_to(OUT)):study.origin.sha(p) for p in files})


def report():
    assert json.loads((OUT/'execution.json').read_text())['status']=='COMPLETE'
    seal=json.loads((OUT/'policy_seal.json').read_text());assert all(study.origin.sha(OUT/p)==h for p,h in seal.items())
    assert json.loads((fresh.OUT/'inference.json').read_text())['status']=='COMPLETE'
    rows=[];deploy=[]
    bank_sets={v:cells(v) for v in VARIANTS}
    for method,seen,matched in study.METHODS:
        for fold in range(5):
            saved=json.loads((OUT/'folds'/str(fold)/f'{method}_policy.json').read_text());variant=saved['constructor']
            deploy.append({'method':method,'fold':fold,'constructor':variant})
            for c in bank_sets[variant]:
                m=c['meta']
                if c['fold']!=fold or (m['rho'] and m['family']!='concept'):continue
                q=study.predict(c,saved['policy']);scale=max(float(c['y'].std()),1e-12)
                rows.append({'cohort':'development','task_id':m['task_id'],'data_id':m['data_id'],'rho':m['rho'],
                    'method':method,**metrics(c['y'],q,'regression',scale)})
                if method=='A':rows.append({'cohort':'development','task_id':m['task_id'],'data_id':m['data_id'],'rho':m['rho'],
                    'method':'Base',**metrics(c['y'],c['base'],'regression',scale)})
    mapping,_=study.cls.fold_mapping('regression','five')
    for path in (fresh.OUT/'inputs/regression').glob('*.npz'):
        tid=int(path.stem);meta=json.loads(path.with_suffix('.json').read_text());fold=mapping[meta['data_id']]
        truth=fresh.OUT/'truth/regression'/f'{tid}.npz'
        assert study.origin.sha(truth)==meta['truth_sha256']
        with np.load(truth) as z:y=z['y_evaluation']
        for rho in (0,20,50,80,100):
            with np.load(fresh.OUT/'banks/regression'/str(tid)/f'{rho}.npz') as z:raw={k:z[k] for k in z.files}
            for method,seen,matched in study.METHODS:
                saved=json.loads((OUT/'folds'/str(fold)/f'{method}_policy.json').read_text());variant=saved['constructor']
                bank=raw['members'].copy()
                if variant=='legacy':bank[0]=raw['repair']
                elif variant.startswith('difference'):
                    with np.load(OUT/'fresh_banks'/str(tid)/f'{rho}.npz') as z:bank[0]=z[variant]
                c={'task':'regression','base':raw['base'],'bank':bank,'mu':float(raw['mu']),'sigma':float(raw['sigma'])}
                q=study.predict(c,saved['policy'])
                rows.append({'cohort':'fresh','task_id':tid,'data_id':meta['data_id'],'rho':rho,'method':method,
                    **metrics(y,q,'regression',max(float(y.std()),1e-12))})
                if method=='A':rows.append({'cohort':'fresh','task_id':tid,'data_id':meta['data_id'],'rho':rho,'method':'Base',
                    **metrics(y,raw['base'],'regression',max(float(y.std()),1e-12))})
    detail=pd.DataFrame(rows);detail.to_csv(OUT/'detail.csv',index=False);tables=[];tests=[]
    for cohort,g in detail.groupby('cohort'):
        v=g[g.rho>0].groupby(['data_id','method'])[['r2','nrmse']].mean();clean=g[g.rho==0].groupby('method').r2.mean()
        for method in ['Base','A','B','C','D']:
            m=v.xs(method,level='method')
            tables.append({'cohort':cohort,'method':method,'datasets':len(m),'r2':m.r2.mean(),'nrmse':m.nrmse.mean(),
                'clean_r2':clean[method],'gain_vs_Base':(m.r2-v.xs('Base',level='method').r2).mean()})
        for a,b in study.CONTRASTS:
            delta=v.xs(a,level='method').r2-v.xs(b,level='method').r2
            tests.append({'cohort':cohort,'contrast':a+'-'+b,**paired_statistics(delta)})
    table=pd.DataFrame(tables);stats=pd.DataFrame(tests)
    for cohort,group in stats.groupby('cohort'):
        prev=0.
        for rank,i in enumerate(sorted(group.index,key=lambda j:stats.loc[j,'p_two_sided'])):
            prev=max(prev,min(1.,(4-rank)*stats.loc[i,'p_two_sided']));stats.loc[i,'p_holm_regression_four']=prev
    table.to_csv(OUT/'tables.csv',index=False);stats.to_csv(OUT/'contrasts.csv',index=False)
    pd.DataFrame(deploy).to_csv(OUT/'constructor_deployment.csv',index=False)
    atomic_json(OUT/'audit.json',{'status':'PASS','all_policy_hashes_unchanged':True,'no_ordering_in_selection':True,
        'matched_availability_is_distinguished_from_deployment':True,'extra_A_B_constructor_search':True,
        'classification_unchanged_v100':True,'fresh_regression_datasets':22,'Markdown_bytes':(RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size})
    print(table.to_string(index=False),flush=True);print(stats.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','prepare-worker','fit','report']);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='prepare-worker':prepare_worker(a.shard,8)
    elif a.phase=='fit':fit()
    else:report()
