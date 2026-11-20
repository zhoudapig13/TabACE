"""Calibrate five physical source corruptions after semisynthetic Clean freezes."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np

from prepare_data import RUN,atomic_json,protocol
import tabicl_base256_calibration_v22 as base
import tabicl_semisynthetic256_v22 as semi

PROFILES=[('mild',4.,.8,.8),('standard',8.,1.,1.),('strong',12.,1.2,1.2),('contrast14',8.,1.,1.4)]
OUT=semi.OUT


def all_profiles():
    p=OUT/'five_refinement_plan.json'
    rows=PROFILES+([tuple(x) for x in json.loads(p.read_text())['additional_profiles']] if p.exists() else [])
    p=OUT/'five_refinement2_plan.json'
    rows+=([tuple(x) for x in json.loads(p.read_text())['additional_profiles']] if p.exists() else [])
    p=OUT/'five_refinement3_plan.json'
    return rows+([tuple(x) for x in json.loads(p.read_text())['additional_profiles']] if p.exists() else [])


def declare_refinement3():
    p=OUT/'five_refinement3_plan.json';assert not p.exists()
    old=OUT/'five_selection.json';(OUT/'five_selection_refinement2.json').write_bytes(old.read_bytes())
    (OUT/'five_refinement2_execution.json').write_bytes((OUT/'five_refinement_execution.json').read_bytes())
    atomic_json(p,{'status':'DECLARED_AFTER_REFINEMENT2_BEFORE_NEW_PREDICTIONS',
        'additional_profiles':[['shortcut2_s1',1.,1.,1.,2],['shortcut2_s2',2.,1.,1.,2],
                               ['shortcut3_s1',1.,1.,1.,3],['shortcut3_s2',2.,1.,1.,3]],
        'parameter_order':['shift_scale','label_scale','label_contrast','shortcut_feature_coverage'],
        'coverage_rule':'source correlation ranking, same affected rows and original-y direction; only shortcut family changes',
        'previous_selection_sha256':base.sha(old),'new_profiles':4,
        'Clean_labels_queries_normalization_unchanged':True,'nominal_strengths':[.2,.5,.8,1.],
        'all_trials_retained':True,'development_only':True})


def declare_refinement2():
    p=OUT/'five_refinement2_plan.json';assert not p.exists()
    old=OUT/'five_selection.json';(OUT/'five_selection_refinement1.json').write_bytes(old.read_bytes())
    (OUT/'five_refinement1_execution.json').write_bytes((OUT/'five_refinement_execution.json').read_bytes())
    atomic_json(p,{'status':'DECLARED_AFTER_REFINEMENT1_BEFORE_NEW_PREDICTIONS',
        'additional_profiles':[['noshift_c11',0.,1.,1.1],['noshift_c12',0.,1.,1.2],
                               ['lowshift02_c11',.2,1.,1.1],['lowshift05_c11',.5,1.,1.1]],
        'previous_selection_sha256':base.sha(old),'new_profiles':4,
        'Clean_labels_queries_and_normalization_unchanged':True,
        'nominal_strengths':[.2,.5,.8,1.],'all_trials_retained':True,
        'selection_is_development_calibration':True})


def declare_refinement():
    p=OUT/'five_refinement_plan.json';assert not p.exists()
    old=OUT/'five_selection.json'
    (OUT/'five_selection_initial.json').write_bytes(old.read_bytes())
    atomic_json(p,{'status':'DECLARED_AFTER_INITIAL_SCORING_BEFORE_ADDITIONAL_PREDICTIONS',
        'additional_profiles':[['lowshift05',.5,1.,1.],['lowshift1',1.,1.,1.],
                               ['lowshift2',2.,1.,1.],['softcontrast',1.,1.,.9]],
        'initial_selection_sha256':base.sha(old),'Clean_labels_and_queries_unchanged':True,
        'nominal_strengths':[.2,.5,.8,1.],'only_generator_coefficients_changed':True,
        'all_trials_preserved':True,'selection_is_development_calibration':True})


def declare():
    p=OUT/'five_plan.json';assert not p.exists()
    clean=json.loads((OUT/'frozen_clean.json').read_text())
    atomic_json(p,{'status':'DECLARED_AFTER_CLEAN_FREEZE_BEFORE_FIVE_SCORING',
        'clean_frozen_sha256':base.sha(OUT/'frozen_clean.json'),'clean_profile':clean,
        'profiles':PROFILES,'families':list(protocol.FAMILIES),'strengths':[.2,.5,.8,1.],
        'label_contrast_rule':'original_source_y + contrast*(corrupted_y-original_source_y)',
        'label_changes_are_explicit_benchmark_source_corruptions':True,
        'query_labels_same_as_frozen_semisynthetic_Clean':True,
        'nrmse_scale_same_as_semisynthetic_Clean_source_std':True,
        'context_rows':256,'seed':0,'n_estimators':1,
        'familywise_parameter_combinations_retained':4**5,
        'Base_development_metrics_select_generator':True,'Shared_Specific_not_used':True,
        'original_paper_protocol_reproduction':False,'all_trials_retained':True})


def materialize(meta,profile,family,rho):
    clean=json.loads((OUT/'frozen_clean.json').read_text())
    data,scale,_=semi.materialize(meta,clean['mu'],clean['h'])
    settings=next(p for p in all_profiles() if p[0]==profile)[1:]
    return corrupt_data(data,family,rho,settings),scale


def corrupt_data(data,family,rho,settings):
    shift,label,contrast=settings[:3]
    coverage=int(settings[3]) if len(settings)>3 else 1
    original_X=data['X_source'].astype(np.float64);original_y=data['y_source'].copy()
    changed=protocol.manipulate_source_context('regression',data['X_source'],data['y_source'],
        family=family,rho=rho/100.,seed=1729+rho,shift_scale=shift,label_scale=label)
    data['X_source']=changed.X_context.copy()
    data['y_source']=data['y_source']+contrast*(changed.y_context-data['y_source'])
    if family=='shortcut' and coverage>1:
        top=protocol._top_feature(original_X,original_y)
        count=min(len(original_y),max(1,int(np.ceil(rho/100.*len(original_y)))))
        rows=protocol._select_rows(original_X,original_y,family=family,count=count,top=top,seed=1729+rho)
        Z=original_X-original_X.mean(0);target=original_y-original_y.mean()
        score=np.abs(Z.T@target)/np.maximum(np.linalg.norm(Z,axis=0),1e-8)
        cols=[int(j) for j in np.argsort(score,kind='stable')[::-1] if int(j)!=top][:coverage-1]
        mean=original_X.mean(0);sd=np.where(original_X.std(0)>1e-12,original_X.std(0),1.)
        direction=np.where(original_y[rows]>=np.median(original_y),-1.,1.)
        for col in cols:data['X_source'][rows,col]=mean[col]+direction*shift*sd[col]
    return data


def worker(shard,shards,refinement=False):
    import torch
    import warnings
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0')
    profiles=all_profiles()[len(PROFILES):] if refinement else PROFILES
    for meta in base.manifest()[shard::shards]:
        for profile,*_ in profiles:
            for family in protocol.FAMILIES:
                for rho in [20,50,80,100]:
                    path=OUT/'five_profiles'/profile/str(meta['task_id'])/f'{family}_{rho}.npz'
                    if path.exists() and path.with_suffix('.json').exists():continue
                    data,scale=materialize(meta,profile,family,rho);assert len(data['y_source'])==256
                    p,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,random_state=0)
                    assert np.isfinite(p).all()
                    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp,prediction=p,y_target=data['y_target']);tmp.replace(path)
                    atomic_json(path.with_suffix('.json'),{'task_id':meta['task_id'],'profile':profile,'family':family,
                        'rho':rho,'seed':0,'context_rows':256,'n_estimators':1,'scale':scale,
                        'semi_synthetic_development_only':True,'query_labels_used_for_inference':False,
                        'metrics':metrics(data['y_target'],p,'regression',scale),
                        'input_hashes':{k:protocol.array_sha256(v) for k,v in data.items()}})
        print('SEMISYNTHETIC_FIVE_DATASET_COMPLETE',meta['task_id'],flush=True)


def launch(refinement=False):
    def one(gpu):
        env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        suffix='_refinement' if refinement else ''
        cmd=[str(RUN/'.venv/bin/python'),'-u',str(Path(__file__)),'worker','--shard',str(gpu),'--shards','8']
        if refinement:cmd+=['--refinement']
        with (RUN/'logs'/f'tabicl_semisynthetic_five_v22{suffix}_gpu{gpu}.log').open('a') as f:
            p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        row={'gpu':gpu,'exit_code':p.returncode};print(json.dumps(row),flush=True);return row
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,range(8)))
    atomic_json(OUT/('five_refinement_execution.json' if refinement else 'five_execution.json'),rows)
    if any(r['exit_code'] for r in rows):raise SystemExit(1)


def select():
    families={f:[] for f in protocol.FAMILIES};global_rows=[]
    profiles=all_profiles()
    for profile,*parameters in profiles:
        files=[p for p in (OUT/'five_profiles'/profile).glob('*/*.json') if not p.name.endswith('.tmp.json')]
        assert len(files)==700
        rows=[json.loads(p.read_text()) for p in files]
        global_rows.append({'profile':profile,**{k:float(np.mean([m['metrics'][k] for m in rows])) for k in ['r2','nrmse']}})
        for family in protocol.FAMILIES:
            sub=[m for m in rows if m['family']==family];assert len(sub)==140
            families[family].append({'profile':profile,'parameters':parameters,
                **{k:float(np.mean([m['metrics'][k] for m in sub])) for k in ['r2','nrmse']}})
    indices=np.array(list(itertools.product(range(len(profiles)),repeat=5)),dtype=np.int8)
    means=sum(np.asarray([[r['r2'],r['nrmse']] for r in families[f]])[indices[:,j]] for j,f in enumerate(protocol.FAMILIES))/5.
    difference=means-np.array([-1.077,1.270]);ok=np.all(abs(difference)<=.005,axis=1)
    distance=np.sum((difference/.005)**2,axis=1)
    pick=int(np.argmin(np.where(ok,distance,np.inf))) if ok.any() else int(np.argmin(distance))
    best={'families':{f:families[f][int(indices[pick,j])] for j,f in enumerate(protocol.FAMILIES)},
          'metrics':dict(zip(['r2','nrmse'],means[pick].tolist())),
          'difference':dict(zip(['r2','nrmse'],difference[pick].tolist())),
          'within_005_each':bool(ok[pick]),'distance':float(distance[pick])}
    np.savez_compressed(OUT/'five_all_combinations.npz',indices=indices,means=means,
                        difference=difference,within_005_each=ok,distance=distance)
    atomic_json(OUT/'five_selection.json',{'global_profiles':global_rows,'family_options':families,'best':best,
                'combinations':len(indices),'passing':int(ok.sum()),'development_selection_only':True})
    print('GLOBAL',global_rows,flush=True);print('BEST',json.dumps(best),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['declare','declare_refinement','declare_refinement2','declare_refinement3','launch','worker','select'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--refinement',action='store_true');a=p.parse_args()
    if a.phase=='declare':declare()
    elif a.phase=='declare_refinement':declare_refinement()
    elif a.phase=='declare_refinement2':declare_refinement2()
    elif a.phase=='declare_refinement3':declare_refinement3()
    elif a.phase=='launch':launch(a.refinement)
    elif a.phase=='worker':worker(a.shard,a.shards,a.refinement)
    else:select()
