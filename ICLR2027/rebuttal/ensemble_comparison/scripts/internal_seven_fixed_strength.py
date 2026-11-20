"""Internal component benchmark with strictly fixed s=.2,.5,.8,1.

Only named GENERATOR CONSTANTS are calibrated. This is explicitly a new
development benchmark, not a reconstruction or a held-out paper experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import warnings
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from sklearn.linear_model import LogisticRegression
from prepare_data import RUN,ROOT,atomic_json,protocol

NAME='internal_seven_fixed_strength_v2'
OUT=RUN/'results'/NAME
SELECTED=OUT/'frozen_benchmark.json'
PLAN=RUN/'configs'/f'{NAME}.json'
FAMILIES=('covariate','label','concept','selection','domain','temporal','support')


def factor_name(f):return f'{f:.6f}'.rstrip('0').rstrip('.')


def candidate_name(coefficient_scale,label_direction):
    return f'coeff_{factor_name(coefficient_scale)}_label_{label_direction}'


def parameters(scale,label_direction):
    return {'x_weight_exponent':3.*scale,'max_boundary_flip_fraction':.3*scale,
            'affine_scale_increment':.75*scale,'affine_translation_sd':.5*scale,
            'support_translation_sd':1.5*scale,'label_prior_displacement_coefficient':scale,
            'label_prior_direction':int(label_direction),'label_prior_extremes':[.1,.9]}


def initialize():
    if PLAN.exists():raise RuntimeError('Plan already exists')
    atomic_json(PLAN,{'status':'DECLARED','name':NAME,'internal_only':True,'paper_reproduction':False,
        'user_authorized_development_benchmark':True,'nominal_strengths':[.2,.5,.8,1.],
        'strength_argument_rescaled':False,'datasets':35,'seed':0,'source_context_rows':256,'query_rows':128,
        'initial_coefficient_scales':[.75,.875,1.],'label_prior_directions':[-1,1],
        'maximum_coefficient_scales':5,'maximum_initial_and_refined_configurations':10,
        'target_Base':{'accuracy':.763,'auc':.826},'absolute_tolerance':.002,
        'selection':'minimize squared development Base Acc/AUC error; no ensemble outcomes consulted',
        'uses_development_evaluation_labels_for_benchmark_selection':True,
        'refinement':'At most two additional coefficient midpoints neighboring the best initial scale',
        'all_trials_retained':True,'Clean_five_frozen':True,'Shared_run':False,'Markdown_written':False,
        'prior_scaled_strength_branch_is_exploration_only':'results/internal_seven_component_v1',
        'reusing_prior_predictions_requires_exact_input_hash_match':True})
    print('FIXED_STRENGTH_PLAN_DECLARED',flush=True)


def make_condition(meta,family,rho,configuration,clean):
    assert rho in [20,50,80,100],rho
    p=configuration['generator_parameters'] if 'generator_parameters' in configuration else configuration
    s=rho/100.
    from seven_shifts import standardize
    H,hy,R,ry,Q,t=[v.copy() for v in clean]
    all_source=np.vstack([H,R]);j=int(np.argmax(np.var(all_source,axis=0)))

    def biased(score,n):
        logits=np.clip(p['x_weight_exponent']*s*np.asarray(score),-15,15)
        probability=np.exp(logits-logits.max());probability/=probability.sum()
        return np.random.default_rng(0).choice(n,size=n,replace=True,p=probability)

    def affine(A):
        mean=H.mean(0);sd=np.maximum(H.std(0),1e-6)
        return mean+(A-mean)*(1+p['affine_scale_increment']*s)+p['affine_translation_sd']*s*sd

    def flip(A,b):
        b=b.copy();feature=int(np.argmax(np.var(H,axis=0)))
        ids=np.argsort(np.abs(A[:,feature]-np.median(H[:,feature])),kind='stable')
        b[ids[:int(np.ceil(p['max_boundary_flip_fraction']*s*len(b)))]]=1-b[ids[:int(np.ceil(p['max_boundary_flip_fraction']*s*len(b)))]]
        return b

    if family=='covariate':
        for A,b in [(H,hy),(R,ry)]:
            ids=biased(standardize(A,all_source)[:,j],len(b));A[:],b[:]=A[ids].copy(),b[ids].copy()
    elif family=='label':
        original=float(t.mean());extreme=.9 if original>=.5 else .1
        if p['label_prior_direction']==-1:extreme=1-extreme
        prior=original+s*p['label_prior_displacement_coefficient']*(extreme-original)
        assert 0<=prior<=1
        rng=np.random.default_rng(0);labels=(rng.random(len(t))<prior).astype(int)
        ids=np.asarray([rng.choice(np.flatnonzero(t==label)) for label in labels]);Q,t=Q[ids],t[ids]
    elif family=='concept':ry=flip(R,ry);t=flip(Q,t)
    elif family=='selection':
        learner=LogisticRegression(max_iter=500,random_state=0).fit(standardize(all_source,all_source),np.r_[hy,ry])
        majority=int(np.mean(np.r_[hy,ry])>=.5)
        for A,b in [(H,hy),(R,ry)]:
            score=learner.predict_proba(standardize(A,all_source))[:,majority]
            ids=biased(score,len(b));A[:],b[:]=A[ids].copy(),b[ids].copy()
    elif family=='domain':Q=affine(Q)
    elif family=='temporal':ry=flip(R,ry);t=flip(Q,t);R=affine(R);Q=affine(Q)
    elif family=='support':
        sd=np.maximum(H.std(0),1e-6);ids=np.argsort(H.var(0))[-min(3,H.shape[1]):]
        Q[:,ids]+=p['support_translation_sd']*s*sd[ids]
    else:raise ValueError(family)
    arrays={'X_source':np.asarray(np.vstack([H[:192],R]),dtype=np.float32),'y_source':np.r_[hy[:192],ry],
        'X_pool':np.asarray(np.vstack([H,R]),dtype=np.float32),'y_pool':np.r_[hy,ry],
        'X_target':np.asarray(Q,dtype=np.float32),'y_target':t,'X_history':np.asarray(H,dtype=np.float32),
        'y_history':hy,'X_reference':np.asarray(R,dtype=np.float32),'y_reference':ry}
    assert all(np.isfinite(a).all() for a in arrays.values())
    info=meta|{'track':NAME,'family':family,'rho':rho,'strength':s,'strength_rescaled':False,
        'generator_parameters':p,'internal_only':True,'scale':1.,'n_classes':2,'condition_materialized':True}
    return arrays,info


def cells():
    from seven_shifts import registry
    return [(m,f,r) for m in registry() if m['task']=='classification'
            for f in FAMILIES for r in [20,50,80,100]]


def base_run(scales,directions,shard,shards,limit):
    from run_tabicl import Predictor,metrics
    import torch,tabicl
    from seven_shifts import classification_clean
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    assert all(0<a<=1 for a in scales) and all(d in [-1,1] for d in directions)
    existing=set()
    if (OUT/'base').exists():
        existing={float(p.name.split('_')[1]) for p in (OUT/'base').iterdir() if p.is_dir()}
    assert len(existing|set(scales))<=5,'Declared coefficient-search budget exhausted'
    predictor=Predictor('classification','cuda:0');cleans={}
    jobs=[(m,f,r,a,d) for a in scales for d in directions for m,f,r in cells()][shard::shards]
    if limit:jobs=jobs[:limit]
    for meta,family,rho,scale,direction in jobs:
        candidate=candidate_name(scale,direction);out=OUT/'base'/candidate/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        if meta['task_id'] not in cleans:cleans[meta['task_id']]=classification_clean(meta)
        cfg=parameters(scale,direction);data,info=make_condition(meta,family,rho,cfg,cleans[meta['task_id']])
        hashes={k:protocol.array_sha256(data[k]) for k in ['X_source','y_source','X_target','y_target']}
        old=RUN/'results/internal_seven_component_v1/base'/factor_name(scale)/str(meta['task_id'])/out.name
        previous=None;reuse=None
        if old.with_suffix('.json').exists():
            previous=json.loads(old.with_suffix('.json').read_text())
            if previous['input_hashes']==hashes:reuse=old
        if reuse is None and scale==1. and (family!='label' or direction==1):
            cached=RUN/'cache/seven_reimplemented/classification'/str(meta['task_id'])/out.name
            with np.load(cached) as z:equal=all(np.array_equal(data[k],z[k]) for k in hashes)
            if equal:
                reuse=RUN/'results/tabicl/seven_reimplemented/classification'/str(meta['task_id'])/out.name
                previous=json.loads(reuse.with_suffix('.json').read_text())
        if reuse is not None:
            with np.load(reuse) as z:
                key='prediction' if 'prediction' in z else 'prediction__base_protocol_check'
                pred=z[key];assert np.array_equal(data['y_target'],z['y_target'])
            observed=previous['metrics'] if 'metrics' in previous else previous['methods']['base_protocol_check']
        else:
            tick=time.perf_counter();pred,_=predictor(data['X_source'],data['y_source'],data['X_target'],n_classes=2)
            observed=metrics(data['y_target'],pred,'classification',1.)|{'seconds':time.perf_counter()-tick}
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prediction=pred,y_target=data['y_target']);tmp.replace(out)
        atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','candidate':candidate,'input_hashes':hashes,
           'coefficient_scale':scale,'label_direction':direction,'metrics':observed,
           'exact_input_reuse_from':str(reuse.relative_to(RUN)) if reuse else None,'TabICL_library':tabicl.__file__})
        print('BASE_DONE',candidate,meta['task_id'],family,rho,bool(reuse),flush=True)
    print('WORKER_FINISHED',shard,flush=True)


def select():
    plan=json.loads(PLAN.read_text());trials=[];counts={}
    for folder in sorted((OUT/'base').iterdir()):
        files=list(folder.glob('*/*.json'));counts[folder.name]=len(files)
        if len(files)!=980:continue
        rows=[json.loads(p.read_text()) for p in files]
        assert len({(r['task_id'],r['family'],r['rho']) for r in rows})==980
        assert all(r['strength']==r['rho']/100. and not r['strength_rescaled'] for r in rows)
        avg={k:float(np.mean([r['metrics'][k] for r in rows])) for k in ['accuracy','auc']}
        diff={k:avg[k]-plan['target_Base'][k] for k in avg}
        trials.append({'candidate':folder.name,'generator_parameters':rows[0]['generator_parameters'],'Base':avg,
                       'difference':diff,'distance':sum(v*v for v in diff.values()),
                       'within_tolerance':all(abs(v)<=plan['absolute_tolerance'] for v in diff.values())})
    trials.sort(key=lambda r:r['distance'])
    result={'condition_counts':counts,'trials':trials,'best':trials[0] if trials else None,
            'internal_only':True,'development_labels_used_for_benchmark_selection':True,'strengths':[.2,.5,.8,1.]}
    atomic_json(OUT/'calibration_summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


def freeze():
    assert not SELECTED.exists(),'Already frozen'
    summary=json.loads((OUT/'calibration_summary.json').read_text())
    assert all(summary['condition_counts'].get(candidate_name(a,d))==980
               for a in [.75,.875,1.] for d in [-1,1]),'Initial declared grid incomplete'
    best=summary['best'];assert best and best['within_tolerance']
    atomic_json(SELECTED,best|{'status':'FROZEN','internal_only':True,'strengths':[.2,.5,.8,1.],
               'strength_rescaled':False,'development_labels_used_for_benchmark_selection':True,
               'ensemble_performance_used_for_benchmark_selection':False,'Clean_five_unchanged':True,'Shared_run':False})
    print('BENCHMARK_FROZEN',best,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','base','select','freeze'])
    p.add_argument('--scales',type=float,nargs='+',default=[.75,.875,1.]);p.add_argument('--directions',type=int,nargs='+',default=[-1,1])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.phase=='init':initialize()
    elif a.phase=='base':base_run(a.scales,a.directions,a.shard,a.shards,a.limit)
    elif a.phase=='select':select()
    else:freeze()
