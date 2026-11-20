"""Source-heldout reliability adaptation; controlled mechanism development.

Reference64 labels belong to the existing source pool. They are removed from
every branch context before computing prediction errors. Matched and generic
slot0 each select between two candidates using these same labels: strengths
0.5/1 for differential repair, seeds0/1 for bootstrap. All methods deploy
five alternatives, with seven total candidate+Base TFM fits per method.
Global calibration excludes the entire evaluated dataset. B/D exclude
Concept/Temporal offline; source-heldout quality is an inference input, not
a fit on final query labels. Both previously scored query cohorts are now
development cohorts, not independent confirmation.
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
from scipy.optimize import minimize
from scipy.special import softmax

import abcd_metric_aligned_v100 as old
import abcd_fresh_confirmation_v101 as fresh
import abcd_reference_repair_v102 as repair
import seven_shifts as seven
from prepare_data import RUN,atomic_json,protocol
from shared_reference import contexts
from run_tabicl import Predictor,metrics
from report_abcd_v84 import paired_statistics

OUT=RUN/'results/abcd_source_quality_v104'
REGS=(.0001,.001,.01,.1)


def freeze():
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/'plan.json'
    plan={'seed':0,'code_sha256':old.origin.sha(__file__),'context_cap':256,
        'source_heldout_reference_rows':64,'target_query_labels_for_local_adaptation':False,
        'source_reference_exclusion':'pool positions320:384 removed from all constructor fits',
        'source_quality':'-log(1+heldout reference MSE / reference response variance)',
        'global_policy':'six-way convex query-dependent weights from source quality and branch agreement',
        'A_B_slot0_candidates':['difference repair0.5','difference repair1'],
        'C_D_slot0_candidates':['bootstrap seed0','bootstrap seed1'],
        'candidate_selection':'same heldout source reference MSE; two choices each',
        'TFM_fits_per_method':7,'deployed_alternative_contexts':5,
        'A_C_calibration_candidates':['all registered calibration families','Concept plus Clean'],
        'A_C_inner_selection':'Concept plus Clean; explicitly family-specialized calibration adaptation',
        'B_D_calibration':'exclude Concept/Temporal; inner selection uses only other families',
        'inference_receives_family_or_strength':False,'regularizations':REGS,
        'candidate_selection_metric':'other-dataset calibration R2; no method-order constraints',
        'previous_query_cohorts_used_as_development':True,'independent_confirmation':False,
        'new_supervised_reliability_module':True,'new_backbone_training':False,
        'extra_target_labels':0,'original_paper_replication':False,
        'interpretation':'ABCD with source-heldout reliability and family-specialized seen calibration; not original Shared',
        'bounded_attempt':'one source-reliability attempt; report all outcomes including incorrect ordering',
        'Markdown_written':False}
    if p.exists():assert json.loads(p.read_text())['code_sha256']==plan['code_sha256']
    else:atomic_json(p,plan)


def prepare_worker(shard):
    torch.set_num_threads(1);seven.PARAMETERS['regression_response_drift_sd']=1.
    items=[m for m in seven.registry() if m['task']=='regression'];model=Predictor('regression','cuda:0')
    for i,m in enumerate(items):
        if i%8!=shard:continue
        tid=int(m['task_id']);clean=seven.regression_clean(m);Qextra=None
        p=fresh.OUT/'inputs/regression'/f'{tid}.npz'
        if p.exists():
            with np.load(p) as z:Qextra=z['X_evaluation']
        paths=[p for p in old.origin.source_paths('regression') if int(p.parent.name)==tid]
        for path in paths:
            meta=json.loads(path.with_suffix('.json').read_text());family,rho=meta['family'],meta['rho']
            out=OUT/'banks'/str(tid)/f'{family}_{rho}.npz'
            if out.with_suffix('.json').exists():continue
            data,_=seven.materialize(m,family,rho,clean)
            X,y,P,py,Q,truth=(data[k] for k in ('X_source','y_source','X_pool','y_pool','X_target','y_target'))
            held=np.arange(320,384);keep=np.setdiff1d(np.arange(len(py)),held)
            T,t=P[held],py[held];P,py=P[keep],py[keep]
            assert np.allclose(P[:256],X) and np.array_equal(py[:256],y)
            use_extra=Qextra is not None and (rho==0 or family=='concept')
            target=np.vstack([Q,Qextra]) if use_extra else Q
            predict_on=np.vstack([T,target])
            # Geometry sees only unlabeled query covariates; every branch
            # context excludes the heldout source labels used for quality.
            views=contexts(P,py,target,X,y,256,'regression')
            generic=[model(A,b,predict_on)[0] for _,A,b,V in views]
            base=model(X,y,predict_on)[0]
            second_ids=np.random.default_rng(1).choice(len(py),256,replace=True)
            second=model(P[second_ids],py[second_ids],predict_on)[0]
            matched=[];details=[]
            for strength in (.5,1.):
                A,b,d=repair.differential_repair(X,y,P,py,strength)
                matched.append(model(A,b,predict_on)[0]);details.append(d)
            scale=max(float(t.std()),1e-6)
            mse=lambda p:float(np.mean((p[:64]-t)**2))/scale**2
            generic_candidates=[generic[0],second]
            choice_g=int(np.argmin([mse(v) for v in generic_candidates]));choice_m=int(np.argmin([mse(v) for v in matched]))
            bank_g=np.stack(generic);bank_m=bank_g.copy()
            bank_g[0]=generic_candidates[choice_g];bank_m[0]=matched[choice_m]
            pack={}
            for name,b in [('generic',bank_g),('matched',bank_m)]:
                all_predictions=np.concatenate([base[None],b],axis=0)
                errors=((all_predictions[:,:64]-t[None])**2).mean(1)/scale**2
                pack[name+'_quality']=-np.log1p(np.clip(errors,0,1e6))
                pack[name+'_members']=b[:,64:64+len(Q)]
                if use_extra:pack[name+'_extra_members']=b[:,64+len(Q):]
            pack.update(base=base[64:64+len(Q)],y=truth,mu=float(py.mean()),sigma=max(float(py.std()),1e-9))
            if use_extra:pack['extra_base']=base[64+len(Q):]
            out.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(out,**pack)
            atomic_json(out.with_suffix('.json'),{'status':'COMPLETE','task_id':tid,'data_id':int(m['data_id']),
                'family':family,'rho':rho,'dataset':m['dataset'],'heldout_source_rows':64,
                'heldout_source_positions':held.tolist(),'context_pool_rows':len(py),
                'context_rows':[len(v[2]) for v in views],'generic_bootstrap_seed':choice_g,
                'matched_repair_strength':(.5,1.)[choice_m],'source_quality_mse_choices':{
                    'generic':[mse(v) for v in generic_candidates],'matched':[mse(v) for v in matched]},
                'repair_details':details,'query_labels_used_for_quality':False,
                'source_x_sha256':protocol.array_sha256(X),'source_y_sha256':protocol.array_sha256(y),
                'query_x_sha256':protocol.array_sha256(Q),'source_quality_labels_sha256':protocol.array_sha256(t),
                'TFM_fits_per_method':7,'deployed_alternatives':5})
        print('SOURCE_QUALITY_BANK',tid,flush=True)


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


def cells(matched):
    mapping,_=old.cls.fold_mapping('regression','five');values=[]
    name='matched' if matched else 'generic'
    for path in (OUT/'banks').glob('*/*.npz'):
        meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            c={'task':'regression','base':z['base'],'bank':z[name+'_members'],'quality':z[name+'_quality'],
                'y':z['y'],'mu':float(z['mu']),'sigma':float(z['sigma']),'meta':meta,'fold':mapping[meta['data_id']]}
            if 'extra_base' in z.files:c.update(extra_base=z['extra_base'],extra_bank=z[name+'_extra_members'])
        values.append(c)
    return values


def features(c):
    b=np.concatenate([c['base'][None],c['bank']],axis=0)
    m=(b-c['mu'])/c['sigma']
    distances=np.abs(m-np.median(m,axis=0)[None])
    return m.T,distances.T,np.asarray(c['quality'])


def predict(c,p):
    M,D,q=features(c)
    if p['kind']=='base':return c['base']
    if p['kind']=='best_reference_member':return M[:,int(np.argmax(q))]*c['sigma']+c['mu']
    if p['kind']=='quality_fixed':w=softmax(q/p['temperature']);return (M*w).sum(1)*c['sigma']+c['mu']
    a=np.asarray(p['parameters']);agreement=np.exp(-D/np.exp(a[-1]))
    weights=softmax(a[:6]+q*a[6:12]+agreement*a[12:18],axis=1)
    return (weights*M).sum(1)*c['sigma']+c['mu']


def score(values,p):
    groups={}
    for c in values:
        value=metrics(c['y'],predict(c,p),'regression',max(float(c['y'].std()),1e-9))['r2']
        groups.setdefault(c['meta']['data_id'],[]).append(value)
    return float(np.mean([np.mean(v) for v in groups.values()]))


def fit(values,regularization):
    counts={}
    for c in values:counts[c['meta']['data_id']]=counts.get(c['meta']['data_id'],0)+1
    columns={k:[] for k in ('M','D','Q','Y','weights')}
    for c in values:
        m,d,q=features(c);n=len(c['y']);factor=c['sigma']**2/max(float(c['y'].var()),1e-9)
        columns['M'].append(m);columns['D'].append(d);columns['Q'].append(np.broadcast_to(q,m.shape))
        columns['Y'].append((c['y']-c['mu'])/c['sigma'])
        columns['weights'].append(np.full(n,factor/(len(counts)*counts[c['meta']['data_id']]*n)))
    t={k:torch.as_tensor(np.concatenate(v),dtype=torch.float64) for k,v in columns.items()}
    initial=np.r_[np.zeros(6),np.ones(6),np.zeros(6),0.]
    def objective(x):
        p=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        agreement=torch.exp(-t['D']/torch.exp(p[-1]))
        weights=torch.softmax(p[:6]+t['Q']*p[6:12]+agreement*p[12:18],dim=1)
        prediction=(weights*t['M']).sum(1)
        penalty=(p-torch.as_tensor(initial)).square().mean()
        loss=((prediction-t['Y']).square()*t['weights']).sum()+regularization*penalty
        loss.backward();return float(loss.detach()),p.grad.detach().numpy()
    result=minimize(objective,initial,method='L-BFGS-B',jac=True,
        bounds=[(-8,8)]*6+[(0,10)]*6+[(-10,10)]*6+[(-4,4)],options={'maxiter':240,'ftol':1e-9,'gtol':1e-6})
    assert np.isfinite(result.x).all() and np.isfinite(result.fun)
    return {'kind':'source_quality_fusion','parameters':result.x.tolist(),'regularization':regularization,
        'training_objective':float(result.fun),'optimizer_success':bool(result.success)}


def fit_worker(fold):
    torch.set_num_threads(1);folder=OUT/'folds'/str(fold)
    for method,seen,matched in old.METHODS:
        bank=cells(matched)
        source=[c for c in bank if c['fold']!=fold and (seen or c['meta']['rho']==0 or c['meta']['family'] not in ('concept','temporal'))]
        inner=min(c['fold'] for c in source);training=[c for c in source if c['fold']!=inner];validation=[c for c in source if c['fold']==inner]
        if seen:validation=[c for c in validation if c['meta']['rho']==0 or c['meta']['family']=='concept']
        assert not any(c['meta']['family'] in ('concept','temporal') and c['meta']['rho'] for c in source) or seen
        trials=[]
        for p in [{'kind':'base'},{'kind':'best_reference_member'}]+[{'kind':'quality_fixed','temperature':t} for t in (.1,.3,1.)]:
            trials.append({'policy':p,'score':score(validation,p),'scope':'fixed'})
        for scope in (['all','concept_clean'] if seen else ['all']):
            selected=[c for c in training if scope=='all' or c['meta']['rho']==0 or c['meta']['family']=='concept']
            for regularization in REGS:
                p=fit(selected,regularization);trials.append({'policy':p,'score':score(validation,p),'scope':scope})
        best=max(trials,key=lambda t:t['score']);p=best['policy']
        if p['kind']=='source_quality_fusion':
            full=[c for c in source if best['scope']=='all' or c['meta']['rho']==0 or c['meta']['family']=='concept']
            p=fit(full,p['regularization'])
        atomic_json(folder/f'{method}_policy.json',{'policy':p,'seen':seen,'matched':matched,'inner_fold':inner,
            'selected_scope':best['scope'],'trials':trials,'validation_family_specialized':seen,
            'training_data_ids':sorted({c['meta']['data_id'] for c in source}),
            'excluded_data_ids':sorted({c['meta']['data_id'] for c in bank if c['fold']==fold}),
            'offline_concept_temporal_excluded':not seen,'own_query_labels_for_fit':False})
        print('SOURCE_QUALITY_POLICY',fold,method,p['kind'],best['scope'],best['score'],flush=True)
    return fold


def train():
    assert json.loads((OUT/'preparation.json').read_text())['status']=='COMPLETE'
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(fit_worker,i) for i in range(5)]):results.append(f.result())
    files=list((OUT/'folds').glob('*/*_policy.json'));assert len(files)==20
    atomic_json(OUT/'policy_seal.json',{str(p.relative_to(OUT)):old.origin.sha(p) for p in files})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','folds':results})


def report():
    assert json.loads((OUT/'execution.json').read_text())['status']=='COMPLETE'
    seal=json.loads((OUT/'policy_seal.json').read_text());assert all(old.origin.sha(OUT/p)==h for p,h in seal.items())
    rows=[]
    for method,seen,matched in old.METHODS:
        for c in cells(matched):
            m=c['meta']
            if m['rho'] and m['family']!='concept':continue
            saved=json.loads((OUT/'folds'/str(c['fold'])/f'{method}_policy.json').read_text())
            p=saved['policy'];predictions={method:predict(c,p)}
            if method=='A':predictions['Base']=c['base']
            for name,q in predictions.items():rows.append({'cohort':'old_development','task_id':m['task_id'],'data_id':m['data_id'],
                'rho':m['rho'],'method':name,**metrics(c['y'],q,'regression',max(float(c['y'].std()),1e-9))})
            if 'extra_base' in c:
                truth=fresh.OUT/'truth/regression'/f'{m["task_id"]}.npz'
                with np.load(truth) as z:y=z['y_evaluation']
                d=c.copy();d['base'],d['bank']=c['extra_base'],c['extra_bank']
                predictions={method:predict(d,p)}
                if method=='A':predictions['Base']=d['base']
                for name,q in predictions.items():rows.append({'cohort':'additional_development','task_id':m['task_id'],'data_id':m['data_id'],
                    'rho':m['rho'],'method':name,**metrics(y,q,'regression',max(float(y.std()),1e-9))})
    detail=pd.DataFrame(rows);detail.to_csv(OUT/'detail.csv',index=False);tables=[];tests=[]
    for cohort,g in detail.groupby('cohort'):
        v=g[g.rho>0].groupby(['data_id','method'])[['r2','nrmse']].mean();clean=g[g.rho==0].groupby('method').r2.mean()
        for name in ['Base','A','B','C','D']:
            a=v.xs(name,level='method');tables.append({'cohort':cohort,'method':name,'datasets':len(a),
                'r2':a.r2.mean(),'nrmse':a.nrmse.mean(),'clean_r2':clean[name]})
        for a,b in old.CONTRASTS:tests.append({'cohort':cohort,'contrast':a+'-'+b,**paired_statistics(v.xs(a,level='method').r2-v.xs(b,level='method').r2)})
    table=pd.DataFrame(tables);stats=pd.DataFrame(tests)
    for cohort,g in stats.groupby('cohort'):
        prev=0.
        for rank,i in enumerate(sorted(g.index,key=lambda j:stats.loc[j,'p_two_sided'])):
            prev=max(prev,min(1.,(4-rank)*stats.loc[i,'p_two_sided']));stats.loc[i,'p_holm_four']=prev
    table.to_csv(OUT/'tables.csv',index=False);stats.to_csv(OUT/'contrasts.csv',index=False)
    atomic_json(OUT/'audit.json',{'status':'PASS','heldout_source64_excluded_from_all_contexts':True,
        'global_calibration_dataset_disjoint':True,'B_D_offline_concept_temporal_excluded':True,
        'previously_evaluated_development_cohorts':True,'independent_confirmation':False,
        'source_supervised_reliability_adaptation':True,'A_C_family_specialized_selection':True,
        'all_four_development_mean_contrasts_positive':bool((stats.gain>0).all()),
        'Markdown_bytes':(RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size})
    print(table.to_string(index=False),flush=True);print(stats.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','prepare-worker','train','report']);p.add_argument('--shard',type=int,default=0)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='prepare-worker':prepare_worker(a.shard)
    elif a.phase=='train':train()
    else:report()
