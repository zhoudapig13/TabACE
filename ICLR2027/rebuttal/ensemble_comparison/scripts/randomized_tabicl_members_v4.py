"""One predeclared single-seed member-randomization sensitivity experiment.

This is not selection for prettier benchmark scores. The previous default
experiment is preserved. Checkpoints, data, query labels and strengths stay
fixed; only documented member context/feature sampling changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from internal_regression_benchmark import manifest
from train_tabicl_robust_regression_v3 import OUT as OLD
import evaluate_tabicl_finetuned_ensembles_v3 as engine

OUT=RUN/'results/tabicl_randomized_members_sensitivity_v4'
PLAN=RUN/'configs/tabicl_randomized_members_sensitivity_v4.json'
TAG='single_seed0'
RUN_SEED=0
OriginalPredictor=engine.UpdatedPredictor


def init():
    assert not PLAN.exists()
    assert json.loads((OLD/'reports/initial128/audit.json').read_text())['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'checkpoints').mkdir(exist_ok=True)
    (OUT/'checkpoints'/TAG).symlink_to(OLD/'checkpoints/initial128',target_is_directory=True)
    atomic_json(PLAN,{'status':'DECLARED_BEFORE_NEW_RANDOMIZED_PREDICTIONS','purpose':'Measure member diversity and robustness sensitivity',
        'outer_seed':RUN_SEED,'outer_repetitions':1,'candidate_count':6,'new_sensitivity_configurations':1,
        'feature_bank':'random subspaces, all canonical rows, column fraction Uniform(.85,1)',
        'bootstrap_bank':'random patches, row fraction Uniform(.70,1), column fraction Uniform(.85,1), replacement=True',
        'member_preprocessing_seeds':'SeedSequence([0, task_id, bank_id, member_id]).spawn(4)',
        'fractions_fixed_per_dataset_and_member_across_conditions':True,
        'same_random_candidate_banks_for_all_applicable_fusion_methods':True,
        'single_and_native6_controls_unchanged':True,'weights_retrained':False,
        'reused_checkpoint_set':'tabicl_regression_weight_adaptation_v3/checkpoints/initial128',
        'context_row_count_may_be_less_than_Base':True,'canonical_source_access_unchanged':True,
        'no_new_history_reference_rows_or_labels_at_inference':True,'benchmark_unchanged':True,
        'nominal_strengths':[.2,.5,.8,1.],'test_scores_used_to_choose_randomization_config':False,
        'decimal_prefix_or_score_dispersion_selection_objective':False,'original_default_results_retained':True,
        'all_results_reported_even_if_similar_or_worse':True,'additional_clean_TRAIN_supervision_same_as_v3':True,
        'single_seed_does_not_mean_identical_member_random_streams':True,'Markdown_written':False})
    atomic_json(OUT/'previous_result_lock.json',{'sha256':{str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [OLD/'reports/initial128/latest_summary.json',OLD/'reports/initial128/averages.csv',OLD/'reports/initial128/audit.json']}})
    print('RANDOMIZED_SINGLE_SEED_SENSITIVITY_DECLARED',flush=True)


def member_spec(task_id,bank_id,member,n,p):
    streams=np.random.SeedSequence([RUN_SEED,int(task_id),bank_id,member]).spawn(4)
    config=np.random.default_rng(streams[0]);rows_rng=np.random.default_rng(streams[1]);cols_rng=np.random.default_rng(streams[2])
    row_fraction=1. if bank_id==0 else float(config.uniform(.70,1.))
    col_fraction=float(config.uniform(.85,1.))
    size=min(n,max(2,int(np.ceil(row_fraction*n))));count=min(p,max(1,int(np.ceil(col_fraction*p))))
    rows=np.arange(n) if bank_id==0 else rows_rng.choice(n,size=size,replace=True)
    cols=np.sort(cols_rng.choice(p,size=count,replace=False))
    seed=int(streams[3].generate_state(1,dtype=np.uint32)[0])
    return rows,cols,seed,row_fraction,col_fraction


class RandomizedPredictor(OriginalPredictor):
    def __init__(self,path):
        super().__init__(path);self.task_id=int(Path(path).parent.name)


def randomized_bank(predictor,X,y,Q):
    single,_=predictor(X,y,Q,quantiles=True,random_state=RUN_SEED)
    native,_=predictor(X,y,Q,members=6,random_state=RUN_SEED)
    result={'single':single,'native':native}
    for bank_id,name in [(0,'feature'),(1,'bootstrap')]:
        preds=[];uncertainty=[];indices=np.full((6,len(y)),-1,dtype=np.int64)
        masks=np.zeros((6,X.shape[1]),dtype=bool);seeds=[];fractions=[];counts=[]
        for member in range(6):
            rows,cols,seed,rf,cf=member_spec(predictor.task_id,bank_id,member,len(y),X.shape[1])
            pred,iqr=predictor(X[rows][:,cols],y[rows],Q[:,cols],quantiles=True,random_state=seed)
            preds.append(pred);uncertainty.append(iqr);indices[member,:len(rows)]=rows;masks[member,cols]=True
            seeds.append(seed);fractions.append([rf,cf]);counts.append([len(rows),len(cols),len(np.unique(rows))])
        result[name]=np.stack(preds);result[name+'_row_indices']=indices;result[name+'_column_masks']=masks
        result[name+'_member_seeds']=np.asarray(seeds,dtype=np.uint32)
        result[name+'_fractions']=np.asarray(fractions);result[name+'_counts']=np.asarray(counts,dtype=np.int64)
        if name=='bootstrap':result['iqr']=np.stack(uncertainty)
    assert all(np.isfinite(result[k]).all() for k in ['single','native','feature','bootstrap','iqr'])
    return result


def configure():
    engine.OUT=OUT;engine.UpdatedPredictor=RandomizedPredictor;engine.bank=randomized_bank


def run(shard,shards,limit):
    configure();torch.set_num_threads(1);warnings.filterwarnings('ignore')
    work=manifest()[shard::shards]
    if limit:work=work[:limit]
    for m in work:engine.evaluate(m,TAG)
    print('RANDOMIZED_MEMBER_WORKER_FINISHED',shard,flush=True)


def report():
    import report_tabicl_actual_finetuning_v3 as reporter
    reporter.OUT=OUT;reporter.report(TAG)
    path=OUT/'reports'/TAG/'latest_summary.json';summary=json.loads(path.read_text())
    summary.update({'experiment_role':'single-seed randomized-member sensitivity, not a replacement for original defaults',
        'randomization_plan':str(PLAN.relative_to(RUN)),'outer_repetitions':1,
        'checkpoints_and_native_controls_unchanged':True,'test_score_or_decimal_dispersion_selection':False,
        'default_baseline_reference':str((OLD/'reports/initial128/averages.csv').relative_to(RUN)),
        'bootstrap_row_budget':'70%–100% of canonical rows, with replacement',
        'feature_column_budget':'85%–100% of canonical columns',
        'all_configs_and_outcomes_preserved':True})
    atomic_json(path,summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['init','run','report'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.phase=='init':init()
    elif a.phase=='run':run(a.shard,a.shards,a.limit)
    else:report()
