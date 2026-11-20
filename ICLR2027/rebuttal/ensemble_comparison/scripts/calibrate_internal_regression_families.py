"""Transparent family-parameter probes for the INTERNAL regression benchmark."""
import argparse
import json
import shutil
from pathlib import Path
import warnings
import itertools
import numpy as np
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,metrics
from internal_regression_benchmark import OUT,manifest,five_condition,TARGETS

CONFIG=RUN/'configs/internal_regression_five_family_refinement.json'


def run(shard,shards,limit):
    import torch
    torch.set_num_threads(1);warnings.filterwarnings('ignore');predictor=Predictor('regression','cuda:0')
    cfg=json.loads(CONFIG.read_text());cache={}
    jobs=[(m,p,f,r) for m in manifest() for p in cfg['probes'] for f in p['families'] for r in [20,50,80,100]][shard::shards]
    if limit:jobs=jobs[:limit]
    for m,probe,family,rho in jobs:
        out=OUT/'five_family_probes'/probe['name']/str(m['task_id'])/f'{family}_{rho}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        data,info=five_condition(m,family,rho,probe['parameters'],cache)
        p,_=predictor(data['X_source'],data['y_source'],data['X_target'])
        observed=metrics(data['y_target'],p,'regression',info['scale'])
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prediction=p,y_target=data['y_target']);tmp.replace(out)
        atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','metrics':observed,'probe':probe['name'],
             'input_hashes':{k:protocol.array_sha256(a) for k,a in data.items()}})
        print('PROBE_DONE',probe['name'],m['dataset'],family,rho,flush=True)
    print('WORKER_FINISHED',shard,flush=True)


def select():
    cfg=json.loads(CONFIG.read_text());base=OUT/'five_candidates/shift8_label1'
    folders={'baseline':base}|{p['name']:OUT/'five_family_probes'/p['name'] for p in cfg['probes']}
    averages={};counts={}
    for name,folder in folders.items():
        rows=[json.loads(p.read_text()) for p in folder.glob('*/*.json')]
        counts[name]=len(rows)
        families=[f for f in protocol.FAMILIES if any(r['family']==f and r['rho']>0 for r in rows)]
        averages[name]={}
        for f in families:
            values=[r['metrics'] for r in rows if r['family']==f and r['rho']>0]
            assert len(values)==140,(name,f,len(values))
            averages[name][f]={k:float(np.mean([r[k] for r in values])) for k in ['r2','nrmse']}
    for p in cfg['probes']:assert counts[p['name']]==140*len(p['families'])
    trials=[]
    shortcuts=[.5,1.,2.,8.] if any(p['name'].startswith('shortcut_feature_shift') for p in cfg['probes']) else [8.]
    mixed_seeds=[None,0,1,2] if any(p['name'].startswith('mixed_assignment') for p in cfg['probes']) else [None]
    for contrast,local_shift,shortcut_shift,mixed_seed in itertools.product([.8,.9,1.],[0.,2.,8.],shortcuts,mixed_seeds):
            names={f:'baseline' for f in protocol.FAMILIES}
            if contrast!=1.:
                for f in ['boundary','band']:names[f]=f'label_contrast{contrast:g}'
            if local_shift!=8.:names['local']=f'local_feature_shift{local_shift:g}'
            if shortcut_shift!=8.:names['shortcut']=f'shortcut_feature_shift{shortcut_shift:g}'
            if mixed_seed is not None:names['mixed']=f'mixed_assignment{mixed_seed}'
            avg={k:sum(averages[names[f]][f][k] for f in protocol.FAMILIES)/5 for k in ['r2','nrmse']}
            diff={k:avg[k]-TARGETS['five'][k] for k in avg}
            trials.append({'contrast':contrast,'local_shift':local_shift,'shortcut_shift':shortcut_shift,'mixed_assignment_seed':mixed_seed,'metrics':avg,'difference':diff,
                'distance':sum((v/.01)**2 for v in diff.values()),'within_tolerance':all(abs(v)<=.01 for v in diff.values()),
                'source_folders':{f:str(folders[names[f]].relative_to(RUN)) for f in protocol.FAMILIES}})
    trials.sort(key=lambda r:(not r['within_tolerance'],r['distance']))
    result={'status':'COMPLETE','all_trials':trials,'best':trials[0],'counts':counts,'family_means':averages,
            'internal_only':True,'benchmark_selection_uses_development_Base_only':True}
    atomic_json(OUT/'five_family_calibration_summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


def materialize_best(allow_approximate=False):
    result=json.loads((OUT/'five_family_calibration_summary.json').read_text());best=result['best']
    assert best['within_tolerance'] or allow_approximate
    cfg=json.loads(CONFIG.read_text());settings=cfg['baseline_other_families'].copy()
    settings['family_parameters']={f:{'shift_scale':8.,'label_scale':1.,'label_contrast':best['contrast']} for f in ['boundary','band']}
    settings['family_parameters']['local']={'shift_scale':best['local_shift'],'label_scale':1.,'label_contrast':1.}
    settings['family_parameters']['shortcut']={'shift_scale':best.get('shortcut_shift',8.),'label_scale':1.,'label_contrast':1.}
    if best.get('mixed_assignment_seed') is not None:
        settings['family_parameters']['mixed']={'shift_scale':8.,'label_scale':1.,'label_contrast':1.,
                                               'mixed_assignment_seed':best['mixed_assignment_seed']}
    candidate='family_calibrated_v1';cache={}
    for m in manifest():
        for family,rho in [('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]:
            if rho==0:src=OUT/'five_candidates/shift8_label1'/str(m['task_id'])/'original_0.npz'
            else:src=RUN/best['source_folders'][family]/str(m['task_id'])/f'{family}_{rho}.npz'
            previous=json.loads(src.with_suffix('.json').read_text())
            data,info=five_condition(m,family,rho,settings,cache)
            hashes={k:protocol.array_sha256(a) for k,a in data.items()}
            assert hashes==previous['input_hashes'],(m['task_id'],family,rho)
            dest=OUT/'five_candidates'/candidate/str(m['task_id'])/src.name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            atomic_json(dest.with_suffix('.json'),info|{'status':'COMPLETE','metrics':previous['metrics'],
              'input_hashes':hashes,'prediction_reused_from':str(src.relative_to(RUN)),
              'internal_family_parameter_calibration':True,'all_four_strengths_unchanged':True,
              'approximate_only_not_paper_reproduction':not best['within_tolerance']})
    print('COHERENT_FAMILY_BENCHMARK_MATERIALIZED',candidate,best['metrics'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','select','materialize'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--allow-approximate',action='store_true')
    a=p.parse_args()
    if a.phase=='run':run(a.shard,a.shards,a.limit)
    elif a.phase=='select':select()
    else:materialize_best(a.allow_approximate)
