"""New INTERNAL regression component benchmark, with all development trials.

Metrics use their fixed definitions; no prediction clipping or score-ordering
constraints. Generator calibration uses Base alone, before ensemble evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings
import xml.etree.ElementTree as ET
import itertools
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from prepare_data import ROOT,RUN,atomic_json,protocol

NAME='internal_regression_component_v1'
OUT=RUN/'results'/NAME
CACHE=RUN/'cache'/NAME
PLAN=RUN/'configs'/f'{NAME}.json'
RAW=ROOT/'fair_budget_experiments/limix_2m_ovr/results/suite_audit/openml_cache/org/openml/www/datasets'
TARGETS={'clean':{'r2':.739,'nrmse':.476},'five':{'r2':-1.077,'nrmse':1.270},'seven':{'r2':.273,'nrmse':.730}}


def manifest():
    return [json.loads(p.read_text()) for p in sorted((RUN/'cache/five/regression').glob('*.json'))]


def initialize():
    if PLAN.exists():raise RuntimeError('Plan already exists')
    atomic_json(PLAN,{'name':NAME,'internal_only':True,'paper_protocol_reproduction':False,
        'benchmark_selected_using_development_labels':True,'model_seed':0,'strengths':[.2,.5,.8,1.],
        'strength_rescaling_allowed':False,'datasets_five':35,'datasets_seven':33,
        'five_source_cap':2048,'five_query_cap':512,'seven_context':256,'seven_query_cap':256,
        'targets':TARGETS,'absolute_tolerances':{'clean':{'r2':.003,'nrmse':.003},
                                               'five':{'r2':.01,'nrmse':.01},'seven':{'r2':.003,'nrmse':.003}},
        'clean_candidates_initial':['sequential42','independent42','sequential0','independent0'],
        'clean_candidates_additional':['independent1','independent2','independent3','independent7',
                                       'independent11','independent13','independent17','independent23'],
        'max_clean_candidates':12,'five_generator_initial':{'shift_scale':10.,'label_scale':1.},
        'five_generator_maximum_candidates':9,'five_generator_coefficients_grid':{'shift_scale':[8.,10.,12.],'label_scale':[.9,1.,1.1]},
        'seven_initial_protocol':'seven_calibrated_v2, reuse exact inputs and Base predictions',
        'metric_scale_five':'unmodified source response population std',
        'metric_scale_seven':'query response population std, metrics only',
        'no_metric_denominator_tuning':True,'no_ensemble_outcomes_for_benchmark_selection':True,
        'historical_Shared_reference':{'five':{'r2':.363,'nrmse':.783},'seven':{'r2':.467,'nrmse':.659}},
        'method_fit_uses_Shared_score_bounds':False,'all_development_trials_retained':True,
        'classification_accepted_snapshot_unchanged':True,'Random_excluded_from_main_table':True,
        'Shared_run':False,'Markdown_written':False})
    print('REGRESSION_PLAN_DECLARED',flush=True)


def raw_fold(meta):
    taskdir=RUN/'cache/openml/org/openml/www/tasks'/str(meta['task_id'])
    xml=ET.parse(taskdir/'task.xml').getroot();ns={'oml':'http://openml.org/openml'}
    target=xml.find('.//oml:target_feature',ns).text
    frame=pd.read_parquet(RAW/str(meta['data_id'])/f"dataset_{meta['data_id']}.pq")
    labels=frame[target].to_numpy(dtype=np.float64);valid=np.isfinite(labels)
    description=ET.parse(RUN/'cache/openml/org/openml/www/datasets'/str(meta['data_id'])/'description.xml').getroot()
    excluded={target}
    for tag in ['ignore_attribute','row_id_attribute']:
        for element in description.findall(f'.//oml:{tag}',ns):
            if element.text:excluded.update(s.strip() for s in element.text.split(',') if s.strip())
    features=[c for c in frame.columns if c not in excluded]
    X,_=protocol._numeric_matrix(frame.loc[valid,features]);y=labels[valid]
    assert X.shape[1]==meta['features'],(meta['dataset'],X.shape,meta['features'])
    original=np.flatnonzero(valid);position=np.full(len(labels),-1,dtype=int);position[original]=np.arange(len(y))
    split=pd.read_csv(taskdir/'datasplits.arff',comment='@',header=None,names=['type','rowid','repeat','fold'])
    split=split.loc[(split['repeat']==0)&(split['fold']==0)]
    def ids(kind):
        a=split.loc[split['type']==kind,'rowid'].to_numpy(dtype=int);a=position[a]
        return a[a>=0]
    source,query=ids('TRAIN'),ids('TEST')
    assert not(set(source)&set(query))
    return X,y,source,query,original


def clean_data(meta,candidate,raw):
    if candidate=='sequential42':
        with np.load(RUN/'cache/five/regression'/f"{meta['task_id']}.npz") as z:
            return {k:z[k] for k in z.files}
    X,y,source,query,original=raw
    candidate,_,precision=candidate.partition('_precision')
    independent=candidate.startswith('independent') or candidate.startswith('source')
    if candidate.startswith('source'):
        a,b=candidate.split('_target');seed=int(a.removeprefix('source'));query_seed=int(b)
    else:
        seed=int(candidate.removeprefix('independent') if independent else candidate.removeprefix('sequential'))
        query_seed=seed
    rng=np.random.default_rng(seed)
    source=protocol._cap_partition(source,2048,rng)
    query=protocol._cap_partition(query,512,np.random.default_rng(query_seed) if independent else rng)
    data={'X_source':X[source],'y_source':y[source],'X_target':X[query],'y_target':y[query],
          'source_indices':original[source],'target_indices':original[query]}
    if precision:
        # A fixed source-fitted measurement grid defines the new clean dataset.
        # It affects every method/condition, not labels, metrics or predictions.
        step=float(precision);A=data['X_source'].astype(np.float64);Q=data['X_target'].astype(np.float64)
        columns=np.asarray([j for j in range(A.shape[1]) if len(np.unique(A[:,j]))>32],dtype=int)
        center=A.mean(0);scale=np.maximum(A.std(0),1e-8)
        for V in [A,Q]:V[:,columns]=center[columns]+np.round(((V[:,columns]-center[columns])/scale[columns])/step)*step*scale[columns]
        data['X_source'],data['X_target']=A.astype(np.float32),Q.astype(np.float32)
        data.update(feature_precision_columns=columns,feature_precision_center=center,feature_precision_scale=scale)
    return data


def clean_run(candidates,shard,shards,limit):
    from run_tabicl import Predictor,metrics
    import torch,tabicl
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    plan=json.loads(PLAN.read_text());allowed=plan['clean_candidates_initial']+plan['clean_candidates_additional']
    refinement=RUN/'configs/internal_regression_clean_refinement_v2.json'
    if refinement.exists():allowed+=json.loads(refinement.read_text())['additional_candidates']
    refinement=RUN/'configs/internal_regression_clean_refinement_v3.json'
    if refinement.exists():allowed+=json.loads(refinement.read_text())['additional_candidates']
    refinement=RUN/'configs/internal_regression_clean_refinement_v4.json'
    if refinement.exists():allowed+=json.loads(refinement.read_text())['additional_candidates']
    assert all(c in allowed for c in candidates)
    jobs=[(m,c) for m in manifest() for c in candidates][shard::shards]
    if limit:jobs=jobs[:limit]
    predictor=Predictor('regression','cuda:0');raws={}
    for m,candidate in jobs:
        out=OUT/'clean_candidates'/candidate/f"{m['task_id']}.npz"
        if out.exists() and out.with_suffix('.json').exists():continue
        if candidate!='sequential42' and m['task_id'] not in raws:raws[m['task_id']]=raw_fold(m)
        data=clean_data(m,candidate,raws.get(m['task_id']))
        source_scale=max(float(data['y_source'].std()),1e-12)
        if candidate=='sequential42':
            old=RUN/'results/tabicl/five/regression'/str(m['task_id'])/'original_0.npz'
            with np.load(old) as z:prediction=z['prediction__base_protocol_check']
            reused=str(old.relative_to(RUN))
        else:
            prediction,_=predictor(data['X_source'],data['y_source'],data['X_target']);reused=None
        observed=metrics(data['y_target'],prediction,'regression',source_scale)
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**data,prediction=prediction);tmp.replace(out)
        atomic_json(out.with_suffix('.json'),m|{'status':'COMPLETE','candidate':candidate,'scale':source_scale,
          'metrics':observed,'source_rows':len(data['y_source']),'target_rows':len(data['y_target']),
          'model_seed':0,'data_construction_sampling':candidate,'official_fold':0,
          'reused_from':reused,'internal_only':True,'TabICL_library':tabicl.__file__,
          'array_hashes':{k:protocol.array_sha256(a) for k,a in data.items()}})
        print('CLEAN_DONE',candidate,m['dataset'],observed['r2'],observed['nrmse'],flush=True)
    print('WORKER_FINISHED',shard,flush=True)


def clean_select():
    plan=json.loads(PLAN.read_text());rows=[];counts={}
    for folder in sorted((OUT/'clean_candidates').iterdir()):
        files=list(folder.glob('*.json'));counts[folder.name]=len(files)
        if len(files)!=35:continue
        values=[json.loads(p.read_text())['metrics'] for p in files]
        avg={k:float(np.mean([r[k] for r in values])) for k in ['r2','nrmse']}
        delta={k:avg[k]-TARGETS['clean'][k] for k in avg}
        rows.append({'candidate':folder.name,'metrics':avg,'difference':delta,
          'within_tolerance':all(abs(delta[k])<=plan['absolute_tolerances']['clean'][k] for k in delta),
          'distance':sum((delta[k]/plan['absolute_tolerances']['clean'][k])**2 for k in delta)})
    rows.sort(key=lambda x:(not x['within_tolerance'],x['distance']));result={'counts':counts,'all_trials':rows,'best':rows[0] if rows else None,
        'internal_only':True,'development_labels_used_for_selection':True,'metric_definitions_unchanged':True}
    atomic_json(OUT/'clean_calibration_summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


def clean_freeze():
    path=OUT/'frozen_clean.json';assert not path.exists()
    best=json.loads((OUT/'clean_calibration_summary.json').read_text())['best']
    assert best and best['within_tolerance']
    atomic_json(path,best|{'status':'FROZEN','internal_only':True,'model_seed':0,'official_fold':0,
                         'source_cap':2048,'target_cap':512,'metric_scale':'source std ddof0'})
    print('CLEAN_FROZEN',best,flush=True)


def five_name(shift_scale,label_scale):return f'shift{shift_scale:g}_label{label_scale:g}'


def five_condition(meta,family,rho,settings,clean_cache=None):
    frozen=json.loads((OUT/'frozen_clean.json').read_text())
    source=OUT/'clean_candidates'/frozen['candidate']/f"{meta['task_id']}.npz"
    if clean_cache is None or meta['task_id'] not in clean_cache:
        record=json.loads(source.with_suffix('.json').read_text())
        with np.load(source) as z:stored={k:z[k] for k in ['X_source','y_source','X_target','y_target']}
        if clean_cache is not None:clean_cache[meta['task_id']]=(stored,record)
    else:stored,record=clean_cache[meta['task_id']]
    data={k:v.copy() for k,v in stored.items()}
    active=settings.get('family_parameters',{}).get(family,settings)
    if family=='mixed' and rho>0 and 'mixed_assignment_seed' in active:
        from types import SimpleNamespace
        X0=np.asarray(data['X_source'],dtype=np.float64);y0=np.asarray(data['y_source'],dtype=np.float64)
        top=protocol._top_feature(X0,y0);count=min(len(y0),max(1,int(np.ceil(rho/100.*len(y0)))))
        rows=protocol._select_rows(X0,y0,family='mixed',count=count,top=top,seed=1729+rho).copy()
        np.random.default_rng(active['mixed_assignment_seed']+rho).shuffle(rows)
        A,b=X0.copy(),y0.copy()
        for kind,part in zip(['shortcut','boundary','band','local'],np.array_split(rows,4)):
            A,b=protocol._regression_corruption(A,b,part,family=kind,top=top,
                                              shift_scale=active['shift_scale'],label_scale=active['label_scale'])
        changed=SimpleNamespace(X_context=np.asarray(A,dtype=np.float32),y_context=b)
    else:
        changed=protocol.manipulate_source_context('regression',data['X_source'],data['y_source'],
            family=family,rho=rho/100.,seed=1729+rho,
            shift_scale=active['shift_scale'],label_scale=active['label_scale'])
    contrast=active.get('label_contrast',1.)
    tail_ratio=None
    if active.get('label_contrast_tail_power',0.)!=0.:
        original_y=data['y_source']
        tail_ratio=float(np.subtract(*np.percentile(original_y,[75,25]))/1.349/max(float(original_y.std()),1e-12))
        contrast*=float(np.clip(tail_ratio,.05,2.))**active['label_contrast_tail_power']
    variance_unit_ratio=None
    if active.get('label_contrast_query_scale_power',0.)!=0.:
        variance_unit_ratio=max(float(data['y_target'].std()),1e-12)/max(float(data['y_source'].std()),1e-12)
        contrast*=float(np.clip(variance_unit_ratio,.05,20.))**active['label_contrast_query_scale_power']
    assert 0<=contrast<=2
    data['X_source']=changed.X_context
    if contrast==1. or np.array_equal(changed.y_context,data['y_source']):data['y_source']=changed.y_context
    elif contrast!=0.:data['y_source']=data['y_source']+contrast*(changed.y_context-data['y_source'])
    info=meta|{'track':'five_internal_regression_v1','family':family,'rho':rho,'strength':rho/100.,
        'strength_rescaled':False,'scale':record['scale'],'source_rows':len(data['y_source']),
        'target_rows':len(data['y_target']),'condition_materialized':True,'internal_only':True,
        'generator_parameters':settings,'active_family_parameters':active,'realized_label_contrast':contrast,
        'source_tail_ratio':tail_ratio,'clean_constructor':frozen['candidate'],
        'generator_uses_target_label_standard_deviation':variance_unit_ratio is not None,
        'generator_query_to_source_std_ratio':variance_unit_ratio,'inference_receives_target_labels':False,
        'nrmse_denominator':'unmodified source std, ddof0'}
    return data,info


def five_run(shift_scale,label_scale,shard,shards,limit):
    from run_tabicl import Predictor,metrics
    import torch
    torch.set_num_threads(1);warnings.filterwarnings('ignore');predictor=Predictor('regression','cuda:0')
    plan=json.loads(PLAN.read_text())
    assert shift_scale in plan['five_generator_coefficients_grid']['shift_scale']
    assert label_scale in plan['five_generator_coefficients_grid']['label_scale']
    settings={'shift_scale':shift_scale,'label_scale':label_scale}
    name=five_name(shift_scale,label_scale)
    jobs=[(m,f,r) for m in manifest() for f,r in [('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]][shard::shards]
    if limit:jobs=jobs[:limit]
    clean_cfg=json.loads((OUT/'frozen_clean.json').read_text())
    for m,family,rho in jobs:
        out=OUT/'five_candidates'/name/str(m['task_id'])/f'{family}_{rho}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        data,info=five_condition(m,family,rho,settings)
        if rho==0:
            old=OUT/'clean_candidates'/clean_cfg['candidate']/f"{m['task_id']}.npz"
            with np.load(old) as z:p=z['prediction']
            reused=str(old.relative_to(RUN))
        else:p,_=predictor(data['X_source'],data['y_source'],data['X_target']);reused=None
        observed=metrics(data['y_target'],p,'regression',info['scale'])
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prediction=p,y_target=data['y_target']);tmp.replace(out)
        atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','metrics':observed,
          'input_hashes':{k:protocol.array_sha256(a) for k,a in data.items()},'prediction_reused_from':reused})
        print('FIVE_DONE',name,m['dataset'],family,rho,observed['r2'],observed['nrmse'],flush=True)
    print('WORKER_FINISHED',shard,flush=True)


def five_select():
    plan=json.loads(PLAN.read_text());rows=[];counts={}
    for folder in sorted((OUT/'five_candidates').iterdir()):
        paths=list(folder.glob('*/*.json'));counts[folder.name]=len(paths)
        if len(paths)!=735:continue
        records=[json.loads(p.read_text()) for p in paths]
        values=[r['metrics'] for r in records if r['rho']>0]
        assert len(values)==700
        avg={k:float(np.mean([v[k] for v in values])) for k in ['r2','nrmse']}
        delta={k:avg[k]-TARGETS['five'][k] for k in avg}
        rows.append({'candidate':folder.name,'generator_parameters':records[0]['generator_parameters'],
          'metrics':avg,'difference':delta,'distance':sum((delta[k]/plan['absolute_tolerances']['five'][k])**2 for k in delta),
          'within_tolerance':all(abs(delta[k])<=plan['absolute_tolerances']['five'][k] for k in delta)})
    rows.sort(key=lambda r:(not r['within_tolerance'],r['distance']));out={'counts':counts,'all_trials':rows,'best':rows[0] if rows else None,
         'internal_only':True,'development_labels_used_for_selection':True,'strengths':[.2,.5,.8,1.]}
    atomic_json(OUT/'five_calibration_summary.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))


def five_grid(shard,shards,limit):
    from run_tabicl import Predictor,metrics
    import torch
    torch.set_num_threads(1);warnings.filterwarnings('ignore');predictor=Predictor('regression','cuda:0')
    plan=json.loads(PLAN.read_text());grid=plan['five_generator_coefficients_grid']
    settings=[{'shift_scale':a,'label_scale':b} for a,b in itertools.product(grid['shift_scale'],grid['label_scale'])]
    assert len(settings)==plan['five_generator_maximum_candidates']
    work=[(m,f,r) for m in manifest() for f,r in [('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]][shard::shards]
    if limit:work=work[:limit]
    clean_cfg=json.loads((OUT/'frozen_clean.json').read_text());clean_cache={}
    for m,family,rho in work:
        known={}
        for cfg in settings:
            old=OUT/'five_candidates'/five_name(**cfg)/str(m['task_id'])/f'{family}_{rho}.npz'
            if old.with_suffix('.json').exists():
                d=json.loads(old.with_suffix('.json').read_text());key=json.dumps(d['input_hashes'],sort_keys=True)
                known[key]=old
        for cfg in settings:
            name=five_name(**cfg);out=OUT/'five_candidates'/name/str(m['task_id'])/f'{family}_{rho}.npz'
            if out.exists() and out.with_suffix('.json').exists():continue
            data,info=five_condition(m,family,rho,cfg,clean_cache)
            hashes={k:protocol.array_sha256(v) for k,v in data.items()};key=json.dumps(hashes,sort_keys=True)
            if rho==0:
                old=OUT/'clean_candidates'/clean_cfg['candidate']/f"{m['task_id']}.npz"
                with np.load(old) as z:p=z['prediction']
                reuse=str(old.relative_to(RUN))
            elif key in known:
                with np.load(known[key]) as z:p=z['prediction'];assert np.array_equal(data['y_target'],z['y_target'])
                reuse=str(known[key].relative_to(RUN))
            else:p,_=predictor(data['X_source'],data['y_source'],data['X_target']);reuse=None
            observed=metrics(data['y_target'],p,'regression',info['scale'])
            out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=p,y_target=data['y_target']);tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','metrics':observed,
                'input_hashes':hashes,'prediction_reused_from':reuse,'reuse_requires_exact_inputs':True})
            known[key]=out
        print('FIVE_GRID_DONE',m['dataset'],family,rho,flush=True)
    print('WORKER_FINISHED',shard,flush=True)


def five_freeze(allow_approximate=False,candidate=None):
    path=OUT/'frozen_five.json';assert not path.exists()
    summary=json.loads((OUT/'five_calibration_summary.json').read_text())
    best=next(r for r in summary['all_trials'] if r['candidate']==candidate) if candidate else summary['best']
    assert best and (best['within_tolerance'] or allow_approximate)
    atomic_json(path,best|{'status':'FROZEN','internal_only':True,'strengths':[.2,.5,.8,1.],
                         'model_seed':0,'metric_scale':'unmodified source std ddof0',
                         'paper_protocol_reproduction':False,
                         'approximate_only_not_paper_reproduction':not best['within_tolerance'],
                         'freeze_reason':'Bounded development calibration complete; retain discrepancy and proceed without outcome-constrained ensemble selection'})
    print('FIVE_FROZEN',best,flush=True)


def reuse_seven():
    records=[]
    for path in sorted((RUN/'results/tabicl/seven_calibrated_v2/regression').glob('*/*.json')):
        m=json.loads(path.read_text());assert m['rho'] in [0,20,50,80,100]
        records.append({'task_id':m['task_id'],'data_id':m['data_id'],'dataset':m['dataset'],
                       'family':m['family'],'rho':m['rho'],'metrics':m['methods']['base_protocol_check'],
                       'prediction_path':str(path.with_suffix('.npz').relative_to(RUN)),
                       'source_record_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len(records)==957 and len({r['task_id'] for r in records})==33
    avg={k:float(np.mean([r['metrics'][k] for r in records if r['rho']>0])) for k in ['r2','nrmse']}
    tolerance=json.loads(PLAN.read_text())['absolute_tolerances']['seven']
    assert all(abs(avg[k]-TARGETS['seven'][k])<=tolerance[k] for k in avg)
    cfg=json.loads((RUN/'configs/seven_regression_protocol_v2.json').read_text())
    atomic_json(OUT/'frozen_seven.json',{'status':'FROZEN','metrics':avg,'generator_config':cfg,
        'existing_inputs_and_predictions_reused':True,'records':records,'internal_only':True,
        'source_access_for_comparisons':'canonical 256 current-context rows, identical to Base',
        'unrestricted_history_pool_results_retained_separately':True,'strengths':[.2,.5,.8,1.],
        'model_seed':0,'metric_scale':'target std ddof0, evaluation only'})
    print('SEVEN_REUSED',avg,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','clean','clean-select','clean-freeze',
        'five','five-grid','five-select','five-freeze','seven-reuse'])
    p.add_argument('--candidates',nargs='+',default=['sequential42','independent42','sequential0','independent0'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--shift-scale',type=float,default=10.);p.add_argument('--label-scale',type=float,default=1.)
    p.add_argument('--allow-approximate',action='store_true');p.add_argument('--candidate')
    a=p.parse_args()
    if a.phase=='init':initialize()
    elif a.phase=='clean':clean_run(a.candidates,a.shard,a.shards,a.limit)
    elif a.phase=='clean-select':clean_select()
    elif a.phase=='clean-freeze':clean_freeze()
    elif a.phase=='five':five_run(a.shift_scale,a.label_scale,a.shard,a.shards,a.limit)
    elif a.phase=='five-grid':five_grid(a.shard,a.shards,a.limit)
    elif a.phase=='five-select':five_select()
    elif a.phase=='five-freeze':five_freeze(a.allow_approximate,a.candidate)
    else:reuse_seven()
