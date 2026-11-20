"""Replay/export the frozen v22 development suite, with explicit task provenance.

Classification and seven-shift regression retain real tasks. Regression Clean
and five perturbations form the separately authorized SEMISYNTHETIC track.
No part of this suite is claimed to recover the original paper protocol.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np

from prepare_data import RUN,atomic_json,protocol
import tabicl_base256_calibration_v22 as base
import tabicl_semisynthetic256_v22 as semi

OUT=base.OUT
CONFIG=OUT/'frozen_suite.json'
_CACHE={}


def freeze():
    assert not CONFIG.exists()
    clean=json.loads((semi.OUT/'frozen_clean.json').read_text())
    five=json.loads((semi.OUT/'five_selection.json').read_text())['best']
    clsfive=json.loads((OUT/'frozen_classification_five.json').read_text())
    assert five['within_005_each'] and clsfive['status']=='FROZEN'
    atomic_json(semi.OUT/'frozen_five.json',{'status':'FROZEN',**five,'context_rows':256,'seed':0,
                'semi_synthetic_development_only':True,'shared_specific_not_used':True})
    from seven_shifts import registry
    cohorts={}
    for task in ['classification','regression']:
        cohorts[task+'/five']=[json.loads(p.read_text()) for p in sorted((RUN/'cache/five'/task).glob('*.json'))]
        cohorts[task+'/seven']=[m for m in registry() if m['task']==task]
    classseven=RUN/'results/internal_seven_fixed_strength_v2/frozen_benchmark.json'
    regseven=RUN/'configs/seven_regression_protocol_v2.json'
    cfg={'status':'FROZEN','seed':0,'n_estimators':1,'context_rows':256,
         'strengths':[.2,.5,.8,1.],'absolute_tolerance':.005,'targets':base.TARGETS,'cohorts':cohorts,
         'classification_five':clsfive,'classification_seven':json.loads(classseven.read_text()),
         'regression_clean':clean,'regression_five':five,'regression_seven':json.loads(regseven.read_text()),
         'task_definition':{'classification/all':'real labels, development-calibrated physical perturbations',
                            'regression/clean_five':'real covariates, TRAIN-only teacher and synthetic noisy labels',
                            'regression/seven':'previous frozen real-label seven-shift development track'},
         'development_selection_uses_Base_evaluation_labels':True,'independent_test_performance_claim':False,
         'original_paper_protocol_recovered':False,'Shared_Specific_not_run_or_used_for_selection':True,
         'benchmark_model_access':'only canonical 256 source feature/label rows and query features; generator teacher/source pool not available to predictors',
         'reference_files_sha256':{str(p.relative_to(RUN)):base.sha(p) for p in
             [classseven,regseven,OUT/'frozen_classification_five.json',semi.OUT/'frozen_clean.json',semi.OUT/'frozen_five.json']}}
    atomic_json(CONFIG,cfg)
    print('V22_SUITE_FROZEN_FOR_AUDIT',flush=True)


@lru_cache(None)
def configuration():return json.loads(CONFIG.read_text())


def cohort(task,block):return configuration()['cohorts'][task+('/five' if block=='clean' else '/'+block)]


def load(task,block,meta,family='original',rho=0):
    cfg=configuration()
    if block=='clean':assert rho==0
    else:assert rho in [20,50,80,100]
    key=(task,block,meta['task_id'])
    if task=='classification' and block in ['clean','five']:
        if key not in _CACHE:
            with np.load(RUN/'cache/five/classification'/f'{meta["task_id"]}.npz') as z:
                _CACHE[key]={k:z[k] for k in ['X_source','y_source','X_target','y_target']}
        data={k:v.copy() for k,v in _CACHE[key].items()}
        if block=='five':
            from sklearn.ensemble import ExtraTreesClassifier
            from scaled_historical_feature_protocol import apply_feature_scaled_shift
            tk=('teacher',meta['task_id'])
            if tk not in _CACHE:
                _CACHE[tk]=ExtraTreesClassifier(n_estimators=128,random_state=0,n_jobs=1).fit(data['X_source'],data['y_source']).predict(data['X_target'])
            s=cfg['classification_five']['family_settings'][family]
            kind='aucband' if family=='band' else family
            op=f'publicshift_d{meta["data_id"]}_{kind}_rho{rho}_beta{int(s["beta"]*100)}'
            Q,_=apply_feature_scaled_shift(op,data['X_source'],data['y_source'],data['X_target'],_CACHE[tk],0,feature_multiplier=s['feature_multiplier'])
            labels,counts=np.unique(data['y_source'],return_counts=True)
            positive=labels[1] if len(labels)==2 else labels[int(np.argmax(counts))]
            direction=data['X_source'][data['y_source']==positive].mean(0)-data['X_source'][data['y_source']!=positive].mean(0)
            data['X_target']=Q+s.get('location_delta',0.)*(rho/100.)*direction
        scale=1.
    elif task=='classification':
        from seven_shifts import classification_clean
        from internal_seven_fixed_strength import make_condition
        if key not in _CACHE:_CACHE[key]=classification_clean(meta)
        data,_=make_condition(meta,family,rho,cfg['classification_seven'],_CACHE[key])
        data={k:data[k] for k in ['X_source','y_source','X_target','y_target']};scale=1.
    elif block in ['clean','five']:
        s=cfg['regression_clean'];data,scale,_=semi.materialize(meta,s['mu'],s['h'])
        if block=='five':
            from tabicl_semisynthetic_five_v22 import corrupt_data
            p=cfg['regression_five']['families'][family]['parameters']
            data=corrupt_data(data,family,rho,p)
    else:
        from calibrated_seven_protocol import materialize
        from seven_shifts import regression_clean
        if key not in _CACHE:_CACHE[key]=regression_clean(meta)
        data,_=materialize(meta,family,rho,_CACHE[key],cfg['regression_seven']['family_amplitude_multipliers'][family])
        data={k:data[k] for k in ['X_source','y_source','X_target','y_target']}
        scale=max(float(data['y_target'].std()),1e-12)
    assert len(data['y_source'])==256
    return data,scale


def reference(task,block,meta,family,rho):
    cfg=configuration();name=f'{family}_{rho}.npz';task_id=str(meta['task_id'])
    if task=='classification' and block=='clean':
        path=RUN/'results/tabicl/five_feature_calibrated/classification'/task_id/name
        return path,'prediction__base_protocol_check'
    if task=='classification' and block=='five':
        path=RUN/cfg['classification_five']['family_settings'][family]['folder']/task_id/name
        return path,'prediction' if 'tabicl_base256_calibration_v22' in str(path) else 'prediction__base_protocol_check'
    if task=='classification':
        path=RUN/'results/internal_seven_fixed_strength_v2/base'/cfg['classification_seven']['candidate']/task_id/name
        return path,'prediction'
    if block=='clean':return semi.OUT/'clean_profiles'/cfg['regression_clean']['profile']/f'{task_id}.npz','prediction'
    if block=='five':return semi.OUT/'five_profiles'/cfg['regression_five']['families'][family]['profile']/task_id/name,'prediction'
    return RUN/'results/tabicl/seven_calibrated_v2/regression'/task_id/name,'prediction__base_protocol_check'


def jobs():
    from seven_shifts import FAMILIES
    rows=[]
    for task in ['classification','regression']:
        for block in ['clean','five','seven']:
            grid=[('original',0)] if block=='clean' else [(f,r) for f in (protocol.FAMILIES if block=='five' else FAMILIES) for r in [20,50,80,100]]
            for meta in cohort(task,block):rows.append((task,block,meta,grid))
    return rows


def export_worker(shard,shards):
    from run_tabicl import metrics
    for task,block,meta,grid in jobs()[shard::shards]:
        for family,rho in grid:
            path=OUT/'input_banks'/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
            if path.exists() and path.with_suffix('.json').exists():continue
            data,scale=load(task,block,meta,family,rho)
            origin,key=reference(task,block,meta,family,rho)
            old_meta=json.loads(origin.with_suffix('.json').read_text())
            with np.load(origin) as z:
                pred=z[key];assert np.array_equal(data['y_target'],z['y_target'])
            hashes={k:protocol.array_sha256(v) for k,v in data.items()}
            expected=old_meta.get('input_hashes')
            if expected is None:
                expected={k:old_meta.get(k+'_sha256') for k in data}
                if task=='classification' and block=='clean':
                    expected={'X_source':old_meta['X_context_sha256'],'y_source':old_meta['y_context_sha256'],
                              'X_target':old_meta['X_shifted_target_sha256'],'y_target':old_meta['y_target_sha256']}
            assert all(expected[k]==hashes[k] for k in data),(task,block,meta['task_id'],family,rho)
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**data,Base_prediction=pred);tmp.replace(path)
            atomic_json(path.with_suffix('.json'),{'status':'COMPLETE','task':task,'block':block,'task_id':meta['task_id'],
                'dataset':meta['dataset'],'family':family,'rho':rho,'context_rows':256,'seed':0,'n_estimators':1,
                'scale':scale,'input_hashes':hashes,'prediction_origin':str(origin.relative_to(RUN)),
                'prediction_key':key,'metrics':metrics(data['y_target'],pred,task,scale),
                'task_provenance':cfg_role(task,block),'development_calibration':True,'paper_replication':False})
        print('V22_INPUT_EXPORT_COMPLETE',task,block,meta['task_id'],flush=True)


def cfg_role(task,block):
    return 'semisynthetic regression development' if task=='regression' and block in ['clean','five'] else 'real-label development'


def launch_export():
    def one(gpu):
        env=os.environ|{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        with (RUN/'logs'/f'benchmark256_v22_export{gpu}.log').open('a') as f:
            p=subprocess.run([str(RUN/'.venv/bin/python'),'-u',str(Path(__file__)),'export_worker',
                              '--shard',str(gpu),'--shards','8'],env=env,stdout=f,stderr=subprocess.STDOUT)
        row={'worker':gpu,'exit_code':p.returncode};print(json.dumps(row),flush=True);return row
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,range(8)))
    atomic_json(OUT/'export_execution.json',rows)
    if any(r['exit_code'] for r in rows):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','export','export_worker'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='freeze':freeze()
    elif a.phase=='export':launch_export()
    else:export_worker(a.shard,a.shards)
