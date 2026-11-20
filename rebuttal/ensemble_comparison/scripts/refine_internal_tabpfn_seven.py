"""Explicit family-wise benchmark development using saved real Base predictions."""
import argparse
import itertools
import json
import shutil
import warnings
import numpy as np
from prepare_data import RUN,atomic_json,protocol
from internal_tabpfn_regression import OUT,condition,TARGETS
from seven_shifts import registry,FAMILIES

AMPLITUDES=[.25,.375,.5,.4125,.425]
LABEL_PROBES=[.4375,.45,.4625]


def run(shard,shards):
    import torch
    from run_tabpfn import model,fit_predict
    from run_tabicl import metrics
    torch.set_num_threads(1);warnings.filterwarnings('ignore');est=model('regression','cuda:0')
    work=[(m,a,r) for m in registry() if m['task']=='regression' for a in LABEL_PROBES for r in [20,50,80,100]][shard::shards]
    for m,a,r in work:
        path=OUT/'calibration/seven_family_probes'/f'label_amplitude{a:g}'/str(m['task_id'])/f'label_{r}.npz'
        if path.exists() and path.with_suffix('.json').exists():continue
        data,info=condition('seven',m,'label',r,{'amplitude':a})
        p=fit_predict(est,data['X_source'],data['y_source'],data['X_target'],'regression',None)
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prediction=p,y_target=data['y_target']);tmp.replace(path)
        atomic_json(path.with_suffix('.json'),info|{'status':'COMPLETE','metrics':metrics(data['y_target'],p,'regression',info['scale']),
            'input_hashes':{k:protocol.array_sha256(v) for k,v in data.items()}})
        print('LABEL_FAMILY_BASE_DONE',m['dataset'],a,r,flush=True)
    print('LABEL_FAMILY_WORKER_FINISHED',shard,flush=True)


def select():
    choices={f:[] for f in FAMILIES}
    for f in FAMILIES:
        for a in AMPLITUDES+(LABEL_PROBES if f=='label' else []):
            folder=OUT/'calibration/seven'/f'amplitude{a:g}' if a in AMPLITUDES else OUT/'calibration/seven_family_probes'/f'label_amplitude{a:g}'
            rows=[json.loads(p.read_text()) for p in folder.glob(f'*/{f}_*.json')]
            rows=[r for r in rows if r['rho']>0];assert len(rows)==132,(f,a,len(rows))
            avg=[float(np.mean([r['metrics'][k] for r in rows])) for k in ['r2','nrmse']]
            choices[f].append({'amplitude':a,'folder':str(folder.relative_to(RUN)),'metrics':avg})
    combinations=np.array(list(itertools.product(*[range(len(choices[f])) for f in FAMILIES])),dtype=np.int16)
    means=sum(np.asarray([r['metrics'] for r in choices[f]])[combinations[:,j]] for j,f in enumerate(FAMILIES))/7
    target=np.array([TARGETS['seven'][k] for k in ['r2','nrmse']]);distance=np.sum(((means-target)/.003)**2,axis=1)
    best=int(np.argmin(distance));selected={f:choices[f][combinations[best,j]] for j,f in enumerate(FAMILIES)}
    np.savez_compressed(OUT/'seven_family_all_aggregate_trials.npz',choice_indices=combinations,
        means_R2_nRMSE=means,distance=distance,family_names=np.array(FAMILIES))
    result={'status':'COMPLETE','selection':'nearest historical Base using development labels only',
        'choices':choices,'selected':selected,'metrics':dict(zip(['r2','nrmse'],means[best].tolist())),
        'gap_to_paper_Base':dict(zip(['r2','nrmse'],(means[best]-target).tolist())),
        'aggregate_combinations_considered':len(combinations),'all_aggregate_trials':'seven_family_all_aggregate_trials.npz',
        'all_use_equal_dataset_family_and_strength_weights':True,'Shared_or_ensemble_outcomes_used':False,
        'candidate_generator_parameters_global_per_family_not_per_dataset':True,'paper_reproduction':False}
    atomic_json(OUT/'seven_family_calibration_summary_v3.json',result);print(json.dumps(result['metrics']),result['gap_to_paper_Base'],flush=True)


def materialize():
    result=json.loads((OUT/'seven_family_calibration_summary_v3.json').read_text());selected=result['selected']
    settings={'family_amplitude_multipliers':{f:row['amplitude'] for f,row in selected.items()}}
    name='family_calibrated_v3'
    for m in registry():
        if m['task']!='regression':continue
        for f,r in [('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]:
            folder=RUN/selected[f]['folder'];src=folder/str(m['task_id'])/f'{f}_{r}.npz'
            previous=json.loads(src.with_suffix('.json').read_text())
            data,info=condition('seven',m,f,r,settings);hashes={k:protocol.array_sha256(v) for k,v in data.items()}
            assert hashes==previous['input_hashes'],(m['task_id'],f,r)
            dest=OUT/'calibration/seven'/name/str(m['task_id'])/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest);atomic_json(dest.with_suffix('.json'),info|{'status':'COMPLETE','metrics':previous['metrics'],
                'input_hashes':hashes,'real_prediction_reused_from':str(src.relative_to(RUN))})
    path=OUT/'frozen_seven.json';assert not path.exists()
    atomic_json(path,{'status':'FROZEN','candidate':name,'generator_parameters':settings,'metrics':result['metrics'],
        'gap_to_paper_Base':result['gap_to_paper_Base'],'within_003_each':all(abs(v)<=.003 for v in result['gap_to_paper_Base'].values()),
        'internal_only':True,'paper_reproduction':False,'benchmark_calibrated_using_development_labels':True,
        'Shared_bounds_used_for_selection':False,'cross_backbone_benchmark_differs':True,'all_trials_retained':True,
        'strengths':[.2,.5,.8,1.],'selection_record':'seven_family_calibration_summary_v3.json'})
    print('PFN_SEVEN_FAMILY_CONFIG_FROZEN',result['metrics'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','select','materialize']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='run':run(a.shard,a.shards)
    elif a.phase=='select':select()
    else:materialize()
