"""Base-only INTERNAL benchmark development with a strict absolute 0.005 gate.

Every family uses one global generator setting across all datasets and all four
nominal strengths. Reuse requires exact physical input hashes. No ensemble or
Shared scores participate in benchmark selection; no predictions are mixed.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np
from prepare_data import RUN,ROOT,atomic_json,protocol
from internal_regression_benchmark import OUT as OLD,manifest,five_condition,TARGETS

OUT=RUN/'results/internal_regression_benchmark_v2'
PLAN=RUN/'configs/internal_regression_benchmark_v2.json'
FAMILIES=list(protocol.FAMILIES)
TOL=.005


def init():
    assert not PLAN.exists()
    OUT.mkdir(parents=True,exist_ok=True)
    frozen={name:hashlib.sha256((OLD/name).read_bytes()).hexdigest() for name in [
        'averages.csv','latest_summary.json','prediction_audit.json','frozen_clean.json','frozen_five.json','frozen_seven.json']}
    atomic_json(OUT/'previous_completed_run_lock.json',{'files_sha256':frozen})
    atomic_json(PLAN,{'status':'DECLARED_BEFORE_NEW_SELECTION','internal_only':True,'paper_protocol_recovered':False,
        'targets':TARGETS,'absolute_tolerance_each_metric':TOL,'all_six_metric_checks_required':True,
        'nominal_strengths':[.2,.5,.8,1.],'model_seed':0,'five_datasets':35,'seven_datasets':33,
        'clean_and_seven_reused_without_changes':True,'five_generator_choices_global_per_family':True,
        'datasets_or_strengths_dropped':False,'macro_averaging_unchanged':True,'metric_definitions_unchanged':True,
        'benchmark_selection_uses_Base_development_labels':True,'Shared_or_ensemble_outcomes_used':False,
        'new_GPU_probe_budget':8,'source_only_generator_profiles_first':True,
        'profiles_using_query_label_standard_deviation_excluded':True,'all_trials_retained':True,
        'previous_ensemble_results_not_valid_on_new_five_benchmark':True,'Markdown_written':False})
    for track in ['clean','seven']:
        old=json.loads((OLD/f'frozen_{track}.json').read_text())
        atomic_json(OUT/f'frozen_{track}.json',old|{'reused_unchanged_from':str((OLD/f'frozen_{track}.json').relative_to(RUN)),
            'absolute_tolerance_each_metric':TOL,'within_005_each':all(abs(old['metrics'][k]-TARGETS[track][k])<=TOL for k in ['r2','nrmse'])})
    print('STRICT_V2_PLAN_DECLARED',flush=True)


def collect(include_tail=False):
    folders=sorted((OLD/'five_candidates').glob('shift*'))+sorted((OLD/'five_family_probes').iterdir())
    if include_tail:folders+=sorted((OLD/'five_candidates').glob('tail_power*'))
    extra=OUT/'family_probes'
    if extra.exists():folders+=sorted(extra.iterdir())
    choices={f:[] for f in FAMILIES};seen={f:set() for f in FAMILIES}
    for folder in folders:
        rows=[json.loads(p.read_text()) for p in folder.glob('*/*.json')]
        for family in FAMILIES:
            values=sorted([r for r in rows if r['family']==family and r['rho']>0],key=lambda r:(int(r['task_id']),r['rho']))
            if not values:continue
            assert len(values)==140,(folder,family,len(values))
            assert not any(r.get('generator_uses_target_label_standard_deviation',False) for r in values)
            settings=values[0].get('active_family_parameters',values[0]['generator_parameters'])
            assert not settings.get('label_contrast_query_scale_power',0.)
            signature=hashlib.sha256(json.dumps([r['input_hashes'] for r in values],sort_keys=True).encode()).hexdigest()
            if signature in seen[family]:continue
            seen[family].add(signature)
            avg={k:float(np.mean([r['metrics'][k] for r in values])) for k in ['r2','nrmse']}
            choices[family].append({'name':folder.name,'folder':str(folder.relative_to(RUN)),
                'parameters':settings,'metrics':avg,'physical_family_input_signature':signature})
    return choices


def probe(shard,shards,plan_name='probe_plan_v1.json'):
    import torch
    import warnings
    from run_tabicl import Predictor,metrics
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    config=json.loads((OUT/plan_name).read_text());predictor=Predictor('regression','cuda:0');cache={}
    jobs=[(m,p,f,r) for m in manifest() for p in config['probes'] for f in p['parameters_by_family'] for r in [20,50,80,100]][shard::shards]
    for m,p,family,rho in jobs:
        path=OUT/'family_probes'/p['name']/str(m['task_id'])/f'{family}_{rho}.npz'
        if path.exists() and path.with_suffix('.json').exists():continue
        parameters=p['parameters_by_family'][family]
        data,info=five_condition(m,family,rho,parameters,cache)
        prediction,_=predictor(data['X_source'],data['y_source'],data['X_target'])
        assert np.isfinite(prediction).all()
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prediction=prediction,y_target=data['y_target']);tmp.replace(path)
        atomic_json(path.with_suffix('.json'),info|{'status':'COMPLETE','benchmark_revision':'internal_regression_benchmark_v2',
            'metrics':metrics(data['y_target'],prediction,'regression',info['scale']),
            'input_hashes':{k:protocol.array_sha256(v) for k,v in data.items()},'probe':p['name']})
        print('V2_BASE_PROBE_DONE',p['name'],m['dataset'],family,rho,flush=True)
    print('V2_BASE_PROBE_WORKER_FINISHED',shard,flush=True)


def select(include_tail=False):
    choices=collect(include_tail);sizes=[len(choices[f]) for f in FAMILIES]
    assert all(sizes)
    indices=np.array(list(itertools.product(*[range(n) for n in sizes])),dtype=np.int16)
    means=sum(np.asarray([[r['metrics'][k] for k in ['r2','nrmse']] for r in choices[f]])[indices[:,j]] for j,f in enumerate(FAMILIES))/5.
    target=np.asarray([TARGETS['five'][k] for k in ['r2','nrmse']]);diff=means-target
    eligible=np.all(np.abs(diff)<=TOL,axis=1);distance=np.sum((diff/TOL)**2,axis=1)
    # Always select a passing point first, then the closest actual Base means.
    selected=int(np.argmin(np.where(eligible,distance,np.inf))) if eligible.any() else int(np.argmin(distance))
    name='source_tail_profiles' if include_tail else 'simple_source_profiles'
    if (OUT/f'{name}_selection.json').exists():
        archive=OUT/'selection_history'/f'{name}_{time.time_ns()}'
        archive.mkdir(parents=True)
        for suffix in ['selection.json','all_trials.npz']:
            shutil.copy2(OUT/f'{name}_{suffix}',archive/f'{name}_{suffix}')
    np.savez_compressed(OUT/f'{name}_all_trials.npz',choice_indices=indices,means_R2_nRMSE=means,
        absolute_difference=np.abs(diff),within_005_each=eligible,family_names=np.array(FAMILIES))
    result={'status':'COMPLETE','stage':name,'choices':choices,'choice_counts':dict(zip(FAMILIES,sizes)),
        'aggregate_combinations':len(indices),'passing_combinations':int(eligible.sum()),
        'selected':{f:choices[f][indices[selected,j]] for j,f in enumerate(FAMILIES)},
        'metrics':dict(zip(['r2','nrmse'],means[selected].tolist())),
        'difference':dict(zip(['r2','nrmse'],diff[selected].tolist())),
        'within_005_each':bool(eligible[selected]),'all_trials_file':f'{name}_all_trials.npz',
        'all_families_have_35_datasets_and_four_strengths':True,
        'physical_inputs_not_prediction_interpolation':True,'internal_only':True,
        'Shared_or_ensemble_outcomes_used':False,'created_unix':time.time()}
    atomic_json(OUT/f'{name}_selection.json',result)
    print('V2_SELECTION',name,result['choice_counts'],'trials',len(indices),'passing',int(eligible.sum()),
          result['metrics'],result['difference'],flush=True)


def materialize(stage):
    result=json.loads((OUT/f'{stage}_selection.json').read_text());assert result['within_005_each']
    settings={'shift_scale':8.,'label_scale':1.,'family_parameters':{f:r['parameters'] for f,r in result['selected'].items()}}
    cache={};records=[]
    for m in manifest():
        for family,rho in [('original',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]:
            if rho==0:src=OLD/'five_candidates/shift8_label1'/str(m['task_id'])/'original_0.npz'
            else:src=RUN/result['selected'][family]['folder']/str(m['task_id'])/f'{family}_{rho}.npz'
            previous=json.loads(src.with_suffix('.json').read_text())
            data,info=five_condition(m,family,rho,settings,cache)
            hashes={k:protocol.array_sha256(v) for k,v in data.items()}
            assert hashes==previous['input_hashes'],(m['task_id'],family,rho,'physical-input mismatch')
            dest=OUT/'base_predictions/five'/str(m['task_id'])/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest);record=info|{'status':'COMPLETE','metrics':previous['metrics'],
                'input_hashes':hashes,'real_Base_prediction_reused_from':str(src.relative_to(RUN)),
                'benchmark_revision':'internal_regression_benchmark_v2'}
            atomic_json(dest.with_suffix('.json'),record);records.append(record)
    from run_tabicl import metrics
    observed=[]
    for path in (OUT/'base_predictions/five').glob('*/*.npz'):
        meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:actual=metrics(z['y_target'],z['prediction'],'regression',meta['scale'])
        for k in ['r2','nrmse']:assert np.isclose(actual[k],meta['metrics'][k],rtol=1e-10,atol=1e-10)
        if meta['rho']>0:observed.append(actual)
    assert len(records)==735 and len(observed)==700
    avg={k:float(np.mean([r[k] for r in observed])) for k in ['r2','nrmse']}
    assert all(abs(avg[k]-TARGETS['five'][k])<=TOL for k in avg)
    path=OUT/'frozen_five.json';assert not path.exists()
    atomic_json(path,{'status':'FROZEN','generator_parameters':settings,'metrics':avg,'target':TARGETS['five'],
        'absolute_tolerance_each_metric':TOL,'within_005_each':True,'strengths':[.2,.5,.8,1.],
        'source_predictions':'base_predictions/five','selection_stage':stage,'model_seed':0,
        'internal_only':True,'benchmark_calibrated_using_Base_development_labels':True,
        'paper_protocol_recovered':False,'Shared_or_ensemble_outcomes_used':False,
        'physical_input_hashes_verified':735,'prediction_metrics_recomputed':735})
    print('STRICT_V2_FIVE_FROZEN',avg,flush=True)


def audit():
    from run_tabicl import metrics
    frozen={t:json.loads((OUT/f'frozen_{t}.json').read_text()) for t in ['clean','five','seven']}
    values={t:[] for t in frozen};clean=frozen['clean']
    for path in (OLD/'clean_candidates'/clean['candidate']).glob('*.npz'):
        record=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            assert not np.intersect1d(z['source_indices'],z['target_indices']).size
            actual=metrics(z['y_target'],z['prediction'],'regression',record['scale'])
        for k in ['r2','nrmse']:assert np.isclose(actual[k],record['metrics'][k],rtol=1e-10,atol=1e-10)
        values['clean'].append(actual)
    assert len(values['clean'])==35
    files=list((OUT/'base_predictions/five').glob('*/*.json'));assert len(files)==735
    for path in files:
        r=json.loads(path.read_text())
        assert r['rho'] in [0,20,50,80,100] and not r['generator_uses_target_label_standard_deviation']
        if r['rho']>0:values['five'].append(r['metrics'])
    assert len(values['five'])==700 and frozen['five']['prediction_metrics_recomputed']==735
    assert frozen['five']['physical_input_hashes_verified']==735
    for record in frozen['seven']['records']:
        path=RUN/record['prediction_path']
        assert hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest()==record['source_record_sha256']
        with np.load(path) as z:
            actual=metrics(z['y_target'],z['prediction__base_protocol_check'],'regression',max(float(z['y_target'].std()),1e-12))
        for k in ['r2','nrmse']:assert np.isclose(actual[k],record['metrics'][k],rtol=1e-10,atol=1e-10)
        if record['rho']>0:values['seven'].append(actual)
    assert len(values['seven'])==924
    rows=[]
    for track in ['clean','five','seven']:
        observed={k:float(np.mean([r[k] for r in values[track]])) for k in ['r2','nrmse']}
        for k in observed:assert np.isclose(observed[k],frozen[track]['metrics'][k],rtol=1e-10,atol=1e-10)
        diff={k:observed[k]-TARGETS[track][k] for k in observed}
        assert all(abs(v)<=TOL for v in diff.values()),(track,diff)
        rows.append({'environment':track,'paper_R2':TARGETS[track]['r2'],'actual_R2':observed['r2'],
            'difference_R2':diff['r2'],'paper_nRMSE':TARGETS[track]['nrmse'],'actual_nRMSE':observed['nrmse'],
            'difference_nRMSE':diff['nrmse'],'all_within_005':True})
    lock=json.loads((OUT/'previous_completed_run_lock.json').read_text())
    for name,sha in lock['files_sha256'].items():assert hashlib.sha256((OLD/name).read_bytes()).hexdigest()==sha,name
    classification=json.loads((OLD/'classification_snapshot_lock.json').read_text())
    for path,sha in classification['files_sha256'].items():assert hashlib.sha256((RUN/path).read_bytes()).hexdigest()==sha,path
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    with (OUT/'Base_comparison.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'status':'PASS','all_six_absolute_differences_at_most_005':True,'rows':rows,
        'total_verified_Base_prediction_vectors':1727,'five_recomputed_during_materialization':735,
        'clean_and_seven_freshly_recomputed_during_final_audit':992,
        'all_four_nominal_strengths_unchanged':[.2,.5,.8,1.],
        'all_dataset_family_strength_cells_retained':True,'all_macro_averaging_weights_unchanged':True,
        'metric_definitions_and_denominators_unchanged':True,'old_completed_TabICL_results_unchanged':True,
        'accepted_classification_snapshot_unchanged':True,'Markdown_bytes':0,
        'internal_development_only':True,'paper_original_generator_recovered':False,
        'benchmark_calibrated_using_Base_development_labels':True,'Shared_or_ensemble_outcomes_used':False,
        'frozen_files_sha256':{f'frozen_{t}.json':hashlib.sha256((OUT/f'frozen_{t}.json').read_bytes()).hexdigest() for t in frozen}}
    atomic_json(OUT/'final_audit.json',report)
    atomic_json(OUT/'ensemble_results_status.json',{'status':'OLD_FIVE_ENSEMBLES_REQUIRE_RERUN_ON_V2',
        'five_changed':True,'clean_and_seven_unchanged':True,'old_results_preserved':True,
        'no_new_ensemble_performance_claims_from_Base_calibration':True})
    print('ALL_SIX_STRICT_005_CHECKS_PASS',json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','select','materialize','probe','audit'])
    p.add_argument('--include-tail',action='store_true');p.add_argument('--stage',default='simple_source_profiles')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--probe-plan',default='probe_plan_v1.json');a=p.parse_args()
    if a.phase=='init':init()
    elif a.phase=='select':select(a.include_tail)
    elif a.phase=='materialize':materialize(a.stage)
    elif a.phase=='audit':audit()
    else:probe(a.shard,a.shards,a.probe_plan)
