"""Explicitly score-calibrated INTERNAL seven-shift component test.

This is a new development benchmark, NOT restoration of the paper protocol.
All calibration trials remain available. Compared methods do not participate
in choosing benchmark parameters. Existing Clean/five artifacts are read-only.
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
from prepare_data import ROOT,RUN,atomic_json,protocol

NAME='internal_seven_component_v1'
OUT=RUN/'results'/NAME
PLAN=RUN/'configs'/f'{NAME}.json'
SELECTED=OUT/'frozen_benchmark.json'


def factor_name(factor):return f'{factor:.6f}'.rstrip('0').rstrip('.')


def initialize():
    from seven_shifts import PARAMETERS
    if PLAN.exists():raise RuntimeError('Plan already exists; preserve original declaration')
    atomic_json(PLAN,{'name':NAME,'purpose':'Internal component validation, user explicitly excludes paper use',
       'paper_protocol_reproduction':False,'benchmark_parameters_selected_using_development_labels':True,
       'seed':0,'task':'classification','backbone':'TabICL-v2','datasets':35,
       'nominal_strengths':[.2,.5,.8,1.],'initial_factors':[.75,.875,1.],
       'maximum_distinct_factors':5,'target_development_Base':{'accuracy':.763,'auc':.826},
       'target_absolute_tolerance':.002,'benchmark_parameterization':'one common multiplier for six non-label families, one for label shift',
       'selection':'minimum sum of squared Acc/AUC differences to stated development targets; no ensemble scores consulted',
       'bounded_refinement':'At most two additional midpoints neighboring the best common factor; all trials retained',
       'original_shift_constants':PARAMETERS,'original_data_split':'seven_reimplemented, unchanged balanced cap',
       'Clean_five_results_frozen':True,'Clean_zero_predictions_reused':True,'Shared_run':False,
       'ensemble_policy':'frozen existing dataset-OOF policies; no target-label adaptation on evaluated dataset',
       'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'Markdown_written':False})
    print('PLAN_FROZEN',PLAN,flush=True)


def cells():
    from seven_shifts import registry,FAMILIES
    return [(m,f,r) for m in registry() if m['task']=='classification'
            for f in FAMILIES for r in [20,50,80,100]]


def make_condition(meta,family,rho,factor,clean):
    from seven_shifts import materialize
    data,info=materialize(meta,family,float(rho)*factor,clean)
    info.update(track=NAME,rho=rho,nominal_strength=rho/100.,effective_strength=rho/100.*factor,
                internal_development_only=True,paper_protocol_reproduction=False,n_classes=2,
                benchmark_factor=factor,benchmark_development_labels_used_for_parameter_selection=True)
    return data,info


def base_run(factors,shard,shards,limit):
    # Match original run_seven.py library import order.
    from run_tabicl import Predictor,metrics
    import torch,tabicl
    from seven_shifts import classification_clean
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    assert PLAN.exists()
    predictor=Predictor('classification','cuda:0');clean_cache={}
    jobs=[(m,f,r,a) for a in factors for m,f,r in cells()][shard::shards]
    if limit:jobs=jobs[:limit]
    for meta,family,rho,factor in jobs:
        out=OUT/'base'/factor_name(factor)/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        if factor==1.:
            old=RUN/'results/tabicl/seven_reimplemented/classification'/str(meta['task_id'])/out.name
            previous=json.loads(old.with_suffix('.json').read_text())
            with np.load(old) as z:p=z['prediction__base_protocol_check'];truth=z['y_target']
            observed=previous['methods']['base_protocol_check'];info=meta|{'family':family,'rho':rho,'benchmark_factor':factor}
            reuse=str(old.relative_to(RUN));input_hashes=None
        else:
            if meta['task_id'] not in clean_cache:clean_cache[meta['task_id']]=classification_clean(meta)
            data,info=make_condition(meta,family,rho,factor,clean_cache[meta['task_id']])
            tick=time.perf_counter();p,_=predictor(data['X_source'],data['y_source'],data['X_target'],n_classes=2)
            truth=data['y_target'];observed=metrics(truth,p,'classification',1.)|{'seconds':time.perf_counter()-tick}
            reuse=None;input_hashes={k:protocol.array_sha256(data[k]) for k in ['X_source','y_source','X_target','y_target']}
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prediction=p,y_target=truth);tmp.replace(out)
        atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','metrics':observed,
            'prediction_reused_from':reuse,'input_hashes':input_hashes,'internal_only':True,
            'original_benchmark_restored':False,'TabICL_library':tabicl.__file__})
        print('BASE_DONE',factor,meta['task_id'],family,rho,observed['accuracy'],observed['auc'],flush=True)
    print('WORKER_FINISHED',shard,flush=True)


def select_benchmark():
    from seven_shifts import FAMILIES
    plan=json.loads(PLAN.read_text());complete={};counts={}
    for folder in sorted((OUT/'base').iterdir()):
        if not folder.is_dir():continue
        files=list(folder.glob('*/*.json'));counts[folder.name]=len(files)
        if len(files)!=980:continue
        rows=[json.loads(p.read_text()) for p in files];factor=float(folder.name)
        assert len({(r['task_id'],r['family'],r['rho']) for r in rows})==980
        complete[factor]={f:{k:float(np.mean([r['metrics'][k] for r in rows if r['family']==f]))
                            for k in ['accuracy','auc']} for f in FAMILIES}
    trials=[];target=plan['target_development_Base']
    for common,label in itertools.product(sorted(complete),repeat=2):
        values={k:sum(complete[label if f=='label' else common][f][k] for f in FAMILIES)/7
                for k in ['accuracy','auc']}
        diff={k:values[k]-target[k] for k in values};distance=sum(v*v for v in diff.values())
        trials.append({'common_factor':common,'label_factor':label,'Base':values,
                       'difference_vs_development_target':diff,'distance':distance,
                       'within_tolerance':all(abs(v)<=plan['target_absolute_tolerance'] for v in diff.values())})
    trials.sort(key=lambda x:(x['distance'],x['common_factor']!=x['label_factor']))
    result={'status':'CALIBRATION_COMPLETE' if trials else 'INCOMPLETE','condition_counts':counts,
            'all_trials':trials,'all_factor_family_means':complete,'best':trials[0] if trials else None,
            'internal_development_only':True,'parameter_selection_uses_benchmark_labels':True,
            'method_results_used_for_benchmark_selection':False,'original_protocol_reproduced':False}
    atomic_json(OUT/'calibration_summary.json',result)
    print(json.dumps({'counts':counts,'best':result['best']},ensure_ascii=False,indent=2),flush=True)


def freeze_selection():
    if SELECTED.exists():raise RuntimeError('Benchmark is already frozen')
    result=json.loads((OUT/'calibration_summary.json').read_text());best=result['best']
    assert best and best['within_tolerance'],'Do not freeze unsuccessful development calibration'
    from seven_shifts import FAMILIES
    atomic_json(SELECTED,{'status':'FROZEN','family_factors':{f:best['label_factor'] if f=='label' else best['common_factor'] for f in FAMILIES},
        'Base':best['Base'],'development_calibration':best,'original_protocol_reproduced':False,
        'parameter_selection_uses_development_labels':True,'ensemble_performance_used_for_selection':False,
        'existing_Clean_five_unchanged':True,'Shared_run':False,'all_trials_retained':'calibration_summary.json'})
    print('BENCHMARK_FROZEN',best,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','base','select','freeze'])
    p.add_argument('--factors',nargs='+',type=float,default=[.75,.875,1.])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.phase=='init':initialize()
    elif a.phase=='base':base_run(a.factors,a.shard,a.shards,a.limit)
    elif a.phase=='select':select_benchmark()
    else:freeze_selection()
