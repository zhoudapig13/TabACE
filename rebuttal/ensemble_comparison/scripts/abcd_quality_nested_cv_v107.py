"""Four-fold inner dataset CV instead of single-fold calibration selection.

Only the validation coverage changes from v106. Same source-heldout quality,
paired row membership,13 candidates, regularization grid, family visibility,
and query cohorts. All observed outcomes retained; no ordering constraint.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k,'1')
import numpy as np
import torch
import abcd_source_quality_v104 as algorithm
import abcd_balanced_quality_v106 as previous
import abcd_metric_aligned_v100 as old
from prepare_data import RUN,atomic_json

OUT=RUN/'results/abcd_quality_nested_cv_v107'


def freeze():
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/'plan.json'
    plan={'seed':0,'code_sha256':old.origin.sha(__file__),
        'input_protocol':str(previous.OUT/'plan.json'),
        'single_change':'all four inner dataset folds for validation rather than minimum single fold',
        'candidates_per_method':13,'regularizations':algorithm.REGS,
        'heldout_source_reference_labels':64,'query_labels_used_for_local_adaptation':False,
        'global_calibration_excludes_entire_evaluated_dataset':True,
        'B_D_offline_concept_temporal_excluded':True,'A_C_family_specialized_calibration':True,
        'context_cap':256,'deployed_alternatives':5,'TFM_fits_per_method':7,
        'source_reference_quality_module_and_constructor_adaptation':True,
        'selection_metric':'per-dataset mean inner-OOF R2, no method-order objective',
        'both_query_cohorts_are_previously_evaluated_development':True,
        'no_independent_confirmation_claim':True,'original_paper_replication':False,
        'bounded_final_calibration_diagnostic':True,'Markdown_written':False}
    if p.exists():assert json.loads(p.read_text())['code_sha256']==plan['code_sha256']
    else:atomic_json(p,plan)


def worker(fold):
    torch.set_num_threads(1);folder=OUT/'folds'/str(fold)
    for method,seen,matched in old.METHODS:
        cells=previous.load_cells(matched)
        source=[c for c in cells if c['fold']!=fold and (seen or c['meta']['rho']==0 or c['meta']['family'] not in ('concept','temporal'))]
        folds=sorted({c['fold'] for c in source});assert len(folds)==4
        validation=[c for c in source if not seen or c['meta']['rho']==0 or c['meta']['family']=='concept']
        candidates=[]
        fixed=[{'kind':'base'},{'kind':'best_reference_member'}]+[{'kind':'quality_fixed','temperature':t} for t in (.1,.3,1.)]
        for p in fixed:candidates.append({'kind':'fixed','policy':p,'score':algorithm.score(validation,p),'scope':'fixed'})
        specialized='concept' if seen else 'covariate'
        for scope in ('all',specialized+'_clean'):
            for regularization in algorithm.REGS:
                inner=[];weighted=0.;total=0
                for g in folds:
                    training=[c for c in source if c['fold']!=g and (scope=='all' or c['meta']['rho']==0 or c['meta']['family']==specialized)]
                    val=[c for c in validation if c['fold']==g]
                    train_ids={c['meta']['data_id'] for c in training};val_ids={c['meta']['data_id'] for c in val}
                    assert train_ids.isdisjoint(val_ids)
                    p=algorithm.fit(training,regularization);value=algorithm.score(val,p)
                    weighted+=len(val_ids)*value;total+=len(val_ids)
                    inner.append({'inner_fold':g,'policy':p,'score':value,
                        'training_data_ids':sorted(train_ids),'validation_data_ids':sorted(val_ids)})
                candidates.append({'kind':'fit','regularization':regularization,'scope':scope,
                    'score':weighted/total,'inner_folds':inner})
        assert len(candidates)==13
        best=max(candidates,key=lambda c:c['score'])
        if best['kind']=='fixed':p=best['policy']
        else:
            full=[c for c in source if best['scope']=='all' or c['meta']['rho']==0 or c['meta']['family']==specialized]
            p=algorithm.fit(full,best['regularization'])
        atomic_json(folder/f'{method}_policy.json',{'policy':p,'seen':seen,'matched':matched,'inner_folds':folds,
            'selected_scope':best['scope'],'selected_regularization':best.get('regularization'),'trials':candidates,
            'candidate_count':13,'training_data_ids':sorted({c['meta']['data_id'] for c in source}),
            'excluded_data_ids':sorted({c['meta']['data_id'] for c in cells if c['fold']==fold}),
            'offline_concept_temporal_excluded':not seen,'own_query_labels_for_fit':False})
        print('FULL_INNER_CV_POLICY',fold,method,p['kind'],best['scope'],best['score'],flush=True)
    return fold


def train():
    freeze();results=[]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for f in as_completed([pool.submit(worker,i) for i in range(5)]):results.append(f.result())
    files=list((OUT/'folds').glob('*/*_policy.json'));assert len(files)==20
    atomic_json(OUT/'policy_seal.json',{str(p.relative_to(OUT)):old.origin.sha(p) for p in files})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','folds':results})


def report():
    assert json.loads((OUT/'plan.json').read_text())['code_sha256']==old.origin.sha(__file__)
    for path in (OUT/'folds').glob('*/*_policy.json'):
        saved=json.loads(path.read_text());assert len(saved['trials'])==13
        assert set(saved['training_data_ids']).isdisjoint(saved['excluded_data_ids'])
        for trial in saved['trials']:
            if trial['kind']=='fit':
                assert len(trial['inner_folds'])==4
                assert all(set(f['training_data_ids']).isdisjoint(f['validation_data_ids']) for f in trial['inner_folds'])
    algorithm.OUT=OUT;algorithm.cells=previous.load_cells;algorithm.report()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','report']);a=p.parse_args()
    train() if a.phase=='train' else report()
