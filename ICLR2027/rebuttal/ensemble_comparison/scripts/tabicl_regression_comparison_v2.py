"""TabICL comparisons on the frozen, audited strict-tolerance v2 benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from prepare_data import RUN,atomic_json,protocol
from internal_regression_benchmark import OUT as OLD
from recalibrate_tabicl_regression_v2 import OUT as BENCH,load_frozen_five
import run_internal_regression_ensembles as bank_runner

OUT=RUN/'results/internal_regression_comparison_v2'
PLAN=RUN/'configs/internal_regression_comparison_v2.json'
_LEGACY_LOAD=bank_runner.load_condition
_CLEAN_CACHE={}


def initialize():
    assert not PLAN.exists()
    audit=json.loads((BENCH/'final_audit.json').read_text());assert audit['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    for track in ['clean','five','seven']:
        source=BENCH/f'frozen_{track}.json';data=json.loads(source.read_text())
        atomic_json(OUT/source.name,data|{'benchmark_source':str(source.relative_to(RUN)),
            'benchmark_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'within_tolerance':True})
    # These predictions and first-stage policies use unchanged physical inputs.
    reuse=[('ensemble_banks/seven','ensemble_banks/seven'),('policies/seven','policies/seven'),
           ('predictions/seven','predictions/seven')]
    for dest,source in reuse:
        p=OUT/dest;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
        p.symlink_to(OLD/source,target_is_directory=True)
    atomic_json(PLAN,{'status':'DECLARED_BEFORE_NEW_V2_ENSEMBLE_INFERENCE','seed':0,'backbone':'TabICL-v2',
        'benchmark_version':'internal_regression_benchmark_v2','benchmark_frozen':True,'benchmark_changes_allowed':False,
        'nominal_strengths':[.2,.5,.8,1.],'candidate_budget':6,'source_access':'Base canonical source only',
        'methods':['Native6','GES6','ConvexStacking6','RidgeStacking6','Median6','FeatureMean6','TrimmedMean6','IQR6'],
        'first_stage':'Previously declared nested dataset-OOF grid; preserve all raw and adapted results',
        'recommended_additional_adaptation':'Calibration safety gate, explicitly separate from original methods',
        'gate_calibration':'Inner cross-fitting over four calibration dataset folds; fifth dataset fold held out',
        'gate_evidence':'One-sided 95% Bonferroni-adjusted normal lower bounds on dataset-level R2 and nRMSE gains',
        'gate_requires_both_metric_lower_bounds_positive':True,'gate_Clean_tolerance':.005,
        'fallback':'Base when calibration evidence or Clean feasibility is insufficient; disclose every fallback fold',
        'fallback_is_ensemble_improvement':False,'deployed_candidates_maximum':6,
        'native_fallback_branch_deploys_Base_only_not_seven_calls':True,
        'Shared_bounds_used_for_selection':False,'outer_test_labels_used_for_selection':False,
        'all_original_and_failed_variants_preserved':True,'backbone_weights_updated':False,
        'single_seed_per_user':True,'Shared_run':False,'Markdown_written':False})
    lock={}
    for root,label in [(BENCH,'benchmark'),(OLD,'completed_v1')]:
        for name in (['frozen_clean.json','frozen_five.json','frozen_seven.json','final_audit.json'] if label=='benchmark' else ['latest_summary.json','averages.csv','prediction_audit.json']):
            p=root/name;lock[str(p.relative_to(RUN))]=hashlib.sha256(p.read_bytes()).hexdigest()
    atomic_json(OUT/'input_and_previous_results_lock.json',{'sha256':lock})
    print('V2_COMPARISON_INITIALIZED',flush=True)


def load_condition(track,m,family,rho):
    if track=='seven':return _LEGACY_LOAD(track,m,family,rho)
    data,info,base=load_frozen_five(m,family,rho,_CLEAN_CACHE)
    # Reuse only exact old physical inputs; changed perturbations are rerun.
    old=OLD/'ensemble_banks/five'/str(m['task_id'])/f'{family}_{rho}.npz'
    dest=OUT/'ensemble_banks/five'/str(m['task_id'])/old.name
    if old.with_suffix('.json').exists() and not dest.exists():
        previous=json.loads(old.with_suffix('.json').read_text());hashes={k:protocol.array_sha256(v) for k,v in data.items()}
        if previous['input_hashes']==hashes:
            with np.load(old) as z:
                assert np.array_equal(z['prediction__Base'],base) and np.array_equal(z['y_target'],data['y_target'])
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dest)
            atomic_json(dest.with_suffix('.json'),previous|info|{'status':'COMPLETE','input_hashes':hashes,
                'reused_exact_input_bank_from':str(old.relative_to(RUN)),'benchmark_version':'v2'})
    return data,info,base


def banks(shard,shards,limit):
    # The shared runner checks completion before loading a condition. Preload the
    # reusable cases once, then let it process only the remaining changed inputs.
    for track,m,f,r in [j for j in bank_runner.jobs() if j[0]=='five'][shard::shards]:
        dest=OUT/'ensemble_banks/five'/str(m['task_id'])/f'{f}_{r}.npz'
        if not dest.with_suffix('.json').exists():load_condition(track,m,f,r)
    bank_runner.OUT=OUT;bank_runner.load_condition=load_condition
    bank_runner.run(shard,shards,limit,['five'])


def fit_first(track,fold):
    import fit_internal_regression_ensembles as fitter
    fitter.OUT=OUT;fitter.fit(track,fold)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','banks','fit_first'])
    p.add_argument('--track',choices=['five','seven'],default='five');p.add_argument('--fold',type=int)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.phase=='init':initialize()
    elif a.phase=='banks':banks(a.shard,a.shards,a.limit)
    else:fit_first(a.track,a.fold)
