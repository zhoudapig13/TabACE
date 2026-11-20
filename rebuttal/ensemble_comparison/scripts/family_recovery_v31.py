"""Predeclared, TRAIN-only mechanism-wise continuous recovery calibration.

Prefer a successful global v30 fraction. Otherwise search the existing source
validation grid for one fraction per mechanism, shared across ALL datasets,
strengths and context budgets. This is a new DEVELOPMENT protocol, not an
original-paper reproduction. No original query labels or scores are read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import continuous_recovery_v30 as source
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabicl_family_recovery_v31'
TARGET=np.asarray([.02,.035,.04])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declare():
    path=OUT/'plan.json';assert not path.exists(),'Do not change a declared search after observing queries'
    atomic_json(path,dict(status='DECLARED_BEFORE_FAMILY_GRID_SEARCH',seed=0,
        hypothesis='Feature corruptions and response corruptions can need different restoration fractions',
        source_prediction_registry=str(source.OUT),fractions=source.INITIAL,
        prefer_successful_global_fraction=True,otherwise_one_fraction_per_family=True,
        context_specific_parameters=False,dataset_specific_parameters=False,strength_specific_parameters=False,
        nominal_strengths=[.2,.5,.8,1.],context_budgets=source.v.BUDGETS,
        original_corrupted256_core_unchanged=True,clean_and_classification_unchanged=True,
        extra_clean_TRAIN_information=True,selection_reads_original_queries=False,
        selection_uses_same_source_disjoint128_validation_rows=True,
        validation_final_R2_gain_bounds=[.005,.08],desired_validation_R2_deltas=TARGET.tolist(),
        minimum_validation_R2_step=.001,nrmse_must_be_nonincreasing=True,
        original_paper_replication=False,development_only=True,Markdown_written=False,
        script_sha256=sha(__file__)))


def solve_block(frame,block,reference):
    families=list(source.v.protocol.FAMILIES) if block=='five' else ['covariate','label','concept','selection','domain','temporal','support']
    rates=source.INITIAL
    expected=35 if block=='five' else 33
    part=frame[frame.block==block]
    grouped=part.groupby(['family','restoration_fraction','requested_context'],as_index=False).agg(
        datasets=('task_id','nunique'),conditions=('rho','size'),r2=('r2','mean'),nrmse=('nrmse','mean'))
    options=np.empty((len(families),len(rates),3,2))
    for j,family in enumerate(families):
        for i,rate in enumerate(rates):
            selected=grouped[(grouped.family==family)&(grouped.restoration_fraction==rate)].sort_values('requested_context')
            assert len(selected)==3 and (selected.datasets==expected).all() and (selected.conditions==expected*4).all(),(block,family,rate,'source validation incomplete')
            assert selected.requested_context.tolist()==[1024,4096,8192]
            options[j,i]=selected[['r2','nrmse']].values
    base=np.asarray([reference.r2,reference.nrmse])
    globals=[]
    for i,rate in enumerate(rates):
        mean=options[:,i].mean(0);curve=np.vstack([base,mean]);delta=mean[:,0]-base[0]
        valid=bool(np.all(np.diff(curve[:,0])>=.001) and np.all(np.diff(curve[:,1])<=0) and .005<=delta[-1]<=.08)
        globals.append(dict(fraction=rate,validation_goal_met=valid,means=mean.tolist(),loss=float(np.sum(((delta-TARGET)/.02)**2))))
    successful=[r for r in globals if r['validation_goal_met']]
    if successful:
        winner=min(successful,key=lambda r:r['loss'])
        return dict(status='VALIDATED_GLOBAL_RULE',fractions={f:winner['fraction'] for f in families},
                    validation_Base=base.tolist(),validation_means=winner['means'],validation_loss=winner['loss'],
                    globals=globals,checked_family_combinations=0,all_constraints_met=True)
    combinations=len(rates)**len(families);best=None;valid_count=0
    for start in range(0,combinations,100000):
        encoded=np.arange(start,min(start+100000,combinations),dtype=np.int64)
        digits=np.stack([(encoded//(len(rates)**j))%len(rates) for j in range(len(families))],axis=1)
        means=sum(options[j,digits[:,j]] for j in range(len(families)))/len(families)
        dr=means[:,:,0]-base[0]
        rsteps=np.column_stack([dr[:,0],np.diff(means[:,:,0],axis=1)])
        nsteps=np.column_stack([means[:,0,1]-base[1],np.diff(means[:,:,1],axis=1)])
        valid=(rsteps>=.001).all(1)&(nsteps<=0).all(1)&(dr[:,-1]>=.005)&(dr[:,-1]<=.08)
        valid_count+=int(valid.sum())
        if not valid.any():continue
        cost=np.sum(((dr-TARGET)/.02)**2,axis=1)+.001*np.asarray(rates)[digits].mean(1)
        idx=int(np.argmin(np.where(valid,cost,np.inf)))
        if best is None or cost[idx]<best['validation_loss']:
            best=dict(status='VALIDATED_FAMILY_RULE',fractions={f:float(rates[k]) for f,k in zip(families,digits[idx])},
                validation_Base=base.tolist(),validation_means=means[idx].tolist(),validation_loss=float(cost[idx]),all_constraints_met=True)
    if best is None:best=dict(status='NO_VALID_FAMILY_COMBINATION',all_constraints_met=False)
    return best|dict(globals=globals,checked_family_combinations=combinations,passing_family_combinations=valid_count)


def solve():
    path=OUT/'selection.json';assert not path.exists(),'A frozen selection cannot be repeated from query feedback'
    assert (OUT/'plan.json').exists()
    frame=pd.read_csv(source.OUT/'validation_detail.csv')
    reference=pd.read_csv(source.v.OUT/'validation_base256.csv').set_index('block')
    best={block:solve_block(frame,block,reference.loc[block]) for block in ['five','seven']}
    valid=all(r['all_constraints_met'] for r in best.values())
    atomic_json(path,dict(status='SOURCE_VALIDATION_GOAL_MET' if valid else 'SOURCE_VALIDATION_GOAL_NOT_MET',best=best,
        source_validation_sha256=sha(source.OUT/'validation_detail.csv'),plan_sha256=sha(OUT/'plan.json'),
        source_validation_only=True,original_query_labels_used=False,query_evaluations=0))
    if valid:
        params=OUT/'parameters.json';assert not params.exists()
        atomic_json(params,dict(status='FROZEN_BEFORE_QUERY_EVALUATION',fractions={block:best[block]['fractions'] for block in best},
            selection_sha256=sha(path),seed=0,n_estimators=1,development_only=True,original_paper_replication=False,
            corrupted256_core_unchanged=True,classification_and_clean_unchanged=True,
            new_protocol_has_extra_pre_corruption_TRAIN_information=True))
    print(json.dumps(best,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['declare','solve']);args=parser.parse_args()
    globals()[args.phase]()
