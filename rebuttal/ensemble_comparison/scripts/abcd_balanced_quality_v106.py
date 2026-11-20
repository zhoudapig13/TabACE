"""Equal-search-budget calibration for v105 paired-membership quality banks.

Every A-D has five fixed candidates plus2scopes x4regularizations. Seen
scopes are all/Concept+Clean; unseen scopes all/Covariate+Clean. Unseen never
fits or selects on Concept/Temporal. Changes are registered before scoring.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import abcd_source_quality_v104 as algorithm
import abcd_paired_quality_v105 as inputs
import abcd_metric_aligned_v100 as old
from prepare_data import RUN,atomic_json

OUT=RUN/'results/abcd_balanced_quality_v106'
ORIGINAL_CELLS=algorithm.cells


def load_cells(matched):
    before=algorithm.OUT;algorithm.OUT=inputs.OUT
    try:return ORIGINAL_CELLS(matched)
    finally:algorithm.OUT=before


def freeze():
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/'plan.json'
    plan={'seed':0,'code_sha256':old.origin.sha(__file__),
        'algorithm_sha256':old.origin.sha(algorithm.__file__),'input_driver_sha256':old.origin.sha(inputs.__file__),
        'input_plan':str(inputs.OUT/'plan.json'),'context_cap':256,'alternative_contexts':5,
        'total_TFM_candidate_fits_per_method':7,'source_heldout_labels':64,'new_target_labels':0,
        'candidate_counts_every_ABCD':13,'common_regularization_grid':algorithm.REGS,
        'A_C_scopes':['all','concept_clean'],'B_D_scopes':['all','covariate_clean'],
        'A_C_selection':'Concept plus Clean other-dataset inner validation',
        'B_D_selection':'only non-Concept/non-Temporal other-dataset inner validation',
        'family_specific_calibration_adaptation':True,'source_supervised_reliability_adaptation':True,
        'paired_bootstrap_row_membership':True,'loss_and_selection_have_no_method_order_constraints':True,
        'both_query_cohorts_previously_evaluated_development':True,'independent_confirmation':False,
        'budget':'one source-quality pipeline; no further scoring-based search',
        'original_paper_replication':False,'Markdown_written':False}
    if p.exists():assert json.loads(p.read_text())['code_sha256']==plan['code_sha256']
    else:atomic_json(p,plan)


def worker(fold):
    torch.set_num_threads(1);folder=OUT/'folds'/str(fold)
    for method,seen,matched in old.METHODS:
        bank=load_cells(matched)
        source=[c for c in bank if c['fold']!=fold and (seen or c['meta']['rho']==0 or c['meta']['family'] not in ('concept','temporal'))]
        inner=min(c['fold'] for c in source);training=[c for c in source if c['fold']!=inner];val=[c for c in source if c['fold']==inner]
        if seen:val=[c for c in val if c['meta']['rho']==0 or c['meta']['family']=='concept']
        trials=[]
        fixed=[{'kind':'base'},{'kind':'best_reference_member'}]+[{'kind':'quality_fixed','temperature':t} for t in (.1,.3,1.)]
        for p in fixed:trials.append({'policy':p,'score':algorithm.score(val,p),'scope':'fixed'})
        specialized='concept' if seen else 'covariate'
        for scope in ('all',specialized+'_clean'):
            train=[c for c in training if scope=='all' or c['meta']['rho']==0 or c['meta']['family']==specialized]
            for regularization in algorithm.REGS:
                p=algorithm.fit(train,regularization)
                trials.append({'policy':p,'score':algorithm.score(val,p),'scope':scope})
        assert len(trials)==13
        best=max(trials,key=lambda t:t['score']);p=best['policy']
        if p['kind']=='source_quality_fusion':
            full=[c for c in source if best['scope']=='all' or c['meta']['rho']==0 or c['meta']['family']==specialized]
            p=algorithm.fit(full,p['regularization'])
        atomic_json(folder/f'{method}_policy.json',{'policy':p,'seen':seen,'matched':matched,'inner_fold':inner,
            'selected_scope':best['scope'],'trials':trials,'candidate_count':13,'validation_family_specialized':seen,
            'training_data_ids':sorted({c['meta']['data_id'] for c in source}),
            'excluded_data_ids':sorted({c['meta']['data_id'] for c in bank if c['fold']==fold}),
            'offline_concept_temporal_excluded':not seen,'own_query_labels_for_fit':False,
            'source_quality_labels_are_heldout_source_reference_not_query':True})
        print('BALANCED_QUALITY_POLICY',fold,method,p['kind'],best['scope'],best['score'],flush=True)
    return fold


def train():
    freeze();assert json.loads((inputs.OUT/'preparation.json').read_text())['status']=='COMPLETE'
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(worker,i) for i in range(5)]):results.append(f.result())
    paths=list((OUT/'folds').glob('*/*_policy.json'));assert len(paths)==20
    atomic_json(OUT/'policy_seal.json',{str(p.relative_to(OUT)):old.origin.sha(p) for p in paths})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','folds':results})


def audit_before_report():
    plan=json.loads((OUT/'plan.json').read_text());assert plan['code_sha256']==old.origin.sha(__file__)
    assert plan['algorithm_sha256']==old.origin.sha(algorithm.__file__)
    assert plan['input_driver_sha256']==old.origin.sha(inputs.__file__)
    count=0
    for path in (inputs.OUT/'banks').glob('*/*.npz'):
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['matched_and_generic_candidates_use_identical_rows']
        assert meta['heldout_source_positions']==list(range(320,384))
        previous=inputs.PREVIOUS/'banks'/path.parent.name/path.name
        with np.load(path) as z,np.load(previous) as oldz:
            for key in ('base','generic_members','generic_quality'):
                assert np.array_equal(z[key],oldz[key])
            selected=meta['matched_bootstrap_seed'];mse=meta['matched_source_quality_mse_choices'][selected]
            assert np.isclose(z['matched_quality'][1],-np.log1p(np.clip(mse,0,1e6)),atol=1e-12)
            assert np.array_equal(z['matched_members'][1:],z['generic_members'][1:])
            assert np.array_equal(z['matched_quality'][np.array([0,2,3,4,5])],z['generic_quality'][np.array([0,2,3,4,5])])
        count+=1
    assert count==957
    for path in (OUT/'folds').glob('*/*_policy.json'):
        saved=json.loads(path.read_text());assert len(saved['trials'])==13
        assert set(saved['training_data_ids']).isdisjoint(saved['excluded_data_ids'])
    assert (RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(OUT/'pre_report_audit.json',{'status':'PASS','paired_banks':count,'policies':20,
        'same_generic_bank_and_Base_for_every_method':True,'Source_heldout_labels_excluded_from_constructor_pool':True,
        'same_source_rows_between_paired_repair_and_no_repair_candidates':True,
        'same13_candidates_each_method':True,'Markdown_bytes':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','report']);a=p.parse_args()
    if a.phase=='train':train()
    else:
        audit_before_report();algorithm.OUT=OUT;algorithm.cells=load_cells;algorithm.report()
