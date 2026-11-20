"""Bounded Base-only development calibration for a separate TabPFN benchmark.

This is explicitly not recovery of the paper's original protocol. Every trial
is retained; neither Shared bounds nor ensemble outcomes select this benchmark.
"""
import argparse
import json
import warnings
import sys
from pathlib import Path
import numpy as np
from prepare_data import RUN,ROOT,atomic_json,protocol
sys.path.insert(0,str(ROOT/'src'))
from internal_regression_benchmark import OUT as TABICL_OUT,manifest

OUT=RUN/'results/internal_tabpfn_regression_v1'
PLAN=RUN/'configs/internal_tabpfn_regression_v1.json'
TARGETS={'clean':{'r2':.739,'nrmse':.476},'five':{'r2':-1.253,'nrmse':1.313},'seven':{'r2':.426,'nrmse':.669}}
CLEAN=['sequential42','source11_target7','source11_target7_precision0.03']
FIVE=[{'name':f'label{x:g}','shift_scale':10.,'label_scale':x} for x in [.9,1.,1.1]]
SEVEN=[{'name':f'amplitude{x:g}','amplitude':x} for x in [.25,.375,.5]]


def declare():
    if PLAN.exists():return
    atomic_json(PLAN,{'status':'DECLARED_BEFORE_NEW_TABPFN_INFERENCE','internal_only':True,
        'benchmark_independent_of_TabICL_per_user_choice':True,'cross_backbone_benchmarks_not_identical':True,
        'paper_protocol_reproduction':False,'development_labels_used_for_Base_calibration':True,
        'ensemble_outcomes_or_Shared_bounds_used_for_benchmark_selection':False,
        'seed':0,'strengths':[.2,.5,.8,1.],'clean_candidates':CLEAN,'five_candidates':FIVE,'seven_candidates':SEVEN,
        'candidate_budget_per_block':3,'targets':TARGETS,'select_nearest_and_report_actual_gap':True,
        'all_trials_preserved':True,'no_test_score_ordering_constraints':True,'Markdown_written':False})


def clean_input(m,candidate):
    with np.load(TABICL_OUT/'clean_candidates'/candidate/f"{m['task_id']}.npz") as z:
        return {k:z[k] for k in ['X_source','y_source','X_target','y_target']}


def condition(track,m,family,rho,settings):
    if track in ['clean','five']:
        candidate=settings if track=='clean' else json.loads((OUT/'frozen_clean.json').read_text())['candidate']
        data=clean_input(m,candidate);scale=max(float(data['y_source'].std()),1e-12)
        if track=='five':
            changed=protocol.manipulate_source_context('regression',data['X_source'],data['y_source'],
                family=family,rho=rho/100.,seed=1729+rho,shift_scale=settings['shift_scale'],label_scale=settings['label_scale'])
            data['X_source'],data['y_source']=changed.X_context,changed.y_context
    else:
        from seven_shifts import regression_clean
        from calibrated_seven_protocol import materialize
        data,_=materialize(m,family,rho,regression_clean(m),settings['amplitude'])
        data={k:data[k] for k in ['X_source','y_source','X_target','y_target']}
        scale=max(float(data['y_target'].std()),1e-12)
    return data,m|{'track':f'{track}_internal_tabpfn_regression_v1','family':family,'rho':rho,
        'scale':scale,'nominal_strength':rho/100.,'strength_rescaled':False,'generator_parameters':settings,
        'internal_only':True,'eligible_source_rows':len(data['y_source']),
        'metric_scale':'unmodified source std ddof0' if track!='seven' else 'target std ddof0 evaluation only',
        'inference_receives_target_labels':False,'backbone':'TabPFN-v3','model_seed':0}


def grid(track):
    if track=='clean':return [(m,'original',0,c,c) for m in manifest() for c in CLEAN]
    if track=='five':return [(m,f,r,c['name'],c) for m in manifest() for c in FIVE
        for f,r in [('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]]
    from seven_shifts import registry,FAMILIES
    return [(m,f,r,c['name'],c) for m in registry() if m['task']=='regression' for c in SEVEN
        for f,r in [('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]]


def run(track,shard,shards,limit):
    import torch
    from run_tabpfn import model,fit_predict
    from run_tabicl import metrics
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    declare();est=model('regression','cuda:0');work=grid(track)[shard::shards]
    if limit:work=work[:limit]
    for m,f,r,name,settings in work:
        path=OUT/'calibration'/track/name/str(m['task_id'])/f'{f}_{r}.npz'
        if path.exists() and path.with_suffix('.json').exists():continue
        data,info=condition(track,m,f,r,settings)
        p=fit_predict(est,data['X_source'],data['y_source'],data['X_target'],'regression',None)
        assert np.isfinite(p).all()
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prediction=p,y_target=data['y_target']);tmp.replace(path)
        atomic_json(path.with_suffix('.json'),info|{'status':'COMPLETE','metrics':metrics(data['y_target'],p,'regression',info['scale']),
            'input_hashes':{k:protocol.array_sha256(a) for k,a in data.items()},'backbone_weights_frozen':True})
        print('PFN_BASE_DONE',track,name,m['dataset'],f,r,flush=True)
    print('PFN_BASE_WORKER_FINISHED',track,shard,flush=True)


def select(track):
    expected={'clean':35,'five':735,'seven':957}[track];rows=[]
    for folder in sorted((OUT/'calibration'/track).iterdir()):
        values=[json.loads(p.read_text()) for p in folder.glob('*/*.json')]
        assert len(values)==expected,(folder,len(values),expected)
        actual=[r['metrics'] for r in values if track=='clean' or r['rho']>0]
        avg={k:float(np.mean([v[k] for v in actual])) for k in ['r2','nrmse']}
        gap={k:avg[k]-TARGETS[track][k] for k in avg}
        rows.append({'candidate':folder.name,'generator_parameters':values[0]['generator_parameters'],
            'metrics':avg,'gap_to_paper_Base':gap,'distance':sum((v/.01)**2 for v in gap.values()),
            'within_003_each':all(abs(v)<=.003 for v in gap.values()),'prediction_count':len(values)})
    assert len(rows)==3
    rows.sort(key=lambda r:r['distance']);best=rows[0]
    atomic_json(OUT/f'{track}_calibration_summary.json',{'all_trials':rows,'selected':best,
        'benchmark_development_only':True,'selection_uses_Base_labels_only':True,'paper_reproduction':False})
    p=OUT/f'frozen_{track}.json';assert not p.exists()
    atomic_json(p,best|{'status':'FROZEN','internal_only':True,'paper_reproduction':False,
        'approximate_only':not best['within_003_each'],'cross_backbone_benchmark_differs':True,
        'strengths':[.2,.5,.8,1.],'benchmark_calibrated_using_development_labels':True,
        'Shared_bounds_used_for_selection':False,'all_trials_retained':True})
    print('PFN_BENCHMARK_FROZEN',track,json.dumps(best),flush=True)


def frozen_jobs():
    from seven_shifts import registry,FAMILIES
    jobs=[]
    for track,items,families,original in [('five',manifest(),protocol.FAMILIES,'original'),
         ('seven',[m for m in registry() if m['task']=='regression'],FAMILIES,'covariate')]:
        jobs.extend((track,m,f,r) for m in items for f,r in [(original,0)]+[(f,r) for f in families for r in [20,50,80,100]])
    return jobs


def frozen_input(track,m,f,r):
    cfg=json.loads((OUT/f'frozen_{track}.json').read_text());data,info=condition(track,m,f,r,cfg['generator_parameters'])
    path=OUT/'calibration'/track/cfg['candidate']/str(m['task_id'])/f'{f}_{r}.npz'
    old=json.loads(path.with_suffix('.json').read_text());assert old['input_hashes']=={k:protocol.array_sha256(v) for k,v in data.items()}
    with np.load(path) as z:base=z['prediction'];assert np.array_equal(z['y_target'],data['y_target'])
    return data,info,base


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['declare','run','select']);p.add_argument('--track',choices=['clean','five','seven'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.phase=='declare':declare()
    elif a.phase=='run':run(a.track,a.shard,a.shards,a.limit)
    else:select(a.track)
