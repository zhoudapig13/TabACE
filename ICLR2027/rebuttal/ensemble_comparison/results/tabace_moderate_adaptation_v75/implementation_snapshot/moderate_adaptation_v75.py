"""TRAIN-calibrated residual regularization for the updated user constraints.

Select and freeze coefficients using held-out TRAIN outcomes only. Original
query data, frozen Base, classification-five and regression-Clean never change.
Query means may miss the requested preferences; such failures are reported.
"""
import argparse
import json
import os
import time
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name,'1')
import numpy as np
import pandas as pd
import balanced_clean_calibration_v71 as clean
import source_expert_v53 as expert
import bagged_specific_v56 as bags
from replay_head_oof_v47 import replay
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import normalize,metrics

OUT=RUN/'results/tabace_moderate_adaptation_v75'
BUDGETS=[256,1024,4096,8192]
ALPHAS=np.round(np.arange(.01,1.001,.01),2).tolist()
SMALL=[.01,.02,.05,.10]


def root(budget,task,block):
    return OUT/'policies'/str(budget)/task/block


def regression_source(budget,block):
    h=expert.h
    cells=h.risk.load(h.source.OUT,'validation',budget,block)
    base={c['key']:c['base'] for c in cells}
    if block=='five':
        folder=h.OUT/'policies'/str(budget)/'regression/five'
        policy=json.loads((folder/'Shared/policy.json').read_text())['model']
        shared,_=replay(cells,10,block,policy,base)
        specific={}
        for family in sorted({c['meta']['family'] for c in cells}):
            group=[c for c in cells if c['meta']['family']==family]
            policy=json.loads((folder/f'Specific_{family}/policy.json').read_text())['model']
            values,_=replay(group,21,block,policy,shared)
            specific.update(values)
        origin=h.OUT
    else:
        policy=json.loads((expert.OUT/'policies'/str(budget)/'regression/seven/all_policies_frozen.json').read_text())
        bag_policy=json.loads((bags.OUT/'policies'/str(budget)/'regression/seven/all_policies_frozen.json').read_text())
        shared={};specific={};banks={};aux={}
        for c in cells:
            tid=str(c['meta']['task_id']);family=c['meta']['family']
            if tid not in aux:
                with np.load(expert.OUT/'experts'/str(budget)/'seven'/f'{tid}.npz') as z:
                    assert np.array_equal(z['y_validation'],c['y']);aux[tid]=z['predictions']
                with np.load(bags.OUT/'experts'/str(budget)/'seven'/f'{tid}.npz') as z:
                    assert np.array_equal(z['y_validation'],c['y']);banks[tid]=z['predictions']
            def mix(selected,anchor):
                if selected is None:return anchor
                a=selected['blend'];p=aux[tid][selected['profile_index']]
                return (1-a)*anchor+a*p
            shared[c['key']]=mix(policy['Shared']['chosen'],c['base'])
            p=mix(policy['Specific'][family]['chosen'],shared[c['key']])
            selected=bag_policy['Specific'][family]['chosen']
            if selected is not None:
                v=banks[tid][selected['profile_index'],:selected['members']].mean(0);a=selected['blend'];p=(1-a)*p+a*v
            specific[c['key']]=p
        origin=bags.OUT
    return cells,base,shared,specific,origin


def reg_select(cells,baseline,candidate,kind):
    initial=expert.score(cells,baseline)[0]
    if kind=='five_shared':target=.463;low=.363;high=.563
    else:target=.12 if kind=='five_specific' else .20;low=1e-9;high=.20 if kind=='five_specific' else .30
    trials=[];best=None;loss=float('inf');best_pred=baseline
    for blend in ALPHAS:
        values={key:(1-blend)*baseline[key]+blend*candidate[key] for key in baseline}
        scores=expert.score(cells,values)[0]
        gain=scores['r2']-initial['r2']
        measure=scores['r2'] if kind=='five_shared' else gain
        eligible=bool(low<=measure<=high and gain>1e-9 and scores['nrmse']<initial['nrmse']-1e-9)
        distance=abs(measure-target)
        trials.append(dict(blend=blend,source_scores=scores,r2_gain=gain,eligible=eligible,distance=distance))
        if eligible and distance<loss-1e-12:
            loss=distance;best=dict(blend=blend);best_pred=values
    return dict(chosen=best,source_baseline=initial,target_kind=kind,target=target,trials=trials),best_pred


def save_source(cells,base,strong_sh,strong_sp,budget,task,block):
    for c in cells:
        path=OUT/'source_candidates'/str(budget)/c['key']
        clean.full.reference.scaling.npwrite(path,base=base[c['key']],strong_shared=strong_sh[c['key']],strong_specific=strong_sp[c['key']],y_validation=c['y'])
        atomic_json(path.with_suffix('.json'),dict(source_bank=str(c['path']),task_id=c['meta']['task_id'],task=task,block=block,
            budget=budget,source_prediction_hashes={name:protocol.array_sha256(value) for name,value in
                [('base',base[c['key']]),('strong_shared',strong_sh[c['key']]),('strong_specific',strong_sp[c['key']])]},
            source_calibration_y_sha256=protocol.array_sha256(c['y']),query_labels_used=False))


def write_query(c,base,shared,specific,record,budget,task,block,policy_path):
    out=OUT/'predictions'/str(budget)/c['key']
    clean.full.reference.scaling.npwrite(out,Base=base,Shared=shared,Specific=specific,y_target=c['y'])
    record.update(adaptation_version='v75',residual_regularization=True,selection_policy=str(policy_path),
        Base_unchanged=True,query_labels_used_for_fit_or_selection=False,extra_clean_TRAIN_supervision=True,
        development_only=True,query_feedback_used_for_development_design=True,independent_test_claim=False,
        original_paper_replication=False,checkpoint_finetuned=False,
        methods={name:metrics(c['y'],p,task,c['meta']['scale']) for name,p in [('Base',base),('Shared',shared),('Specific',specific)]})
    atomic_json(out.with_suffix('.json'),record)
    return [dict(context=budget,task=task,block=block,method=name,task_id=c['meta']['task_id'],**values) for name,values in record['methods'].items()]


def run_regression(budget,block):
    expert.h.source.verify_registry()
    cells,base,strong_sh,strong_sp,origin=regression_source(budget,block)
    save_source(cells,base,strong_sh,strong_sp,budget,'regression',block)
    sh,shared=reg_select(cells,base,strong_sh,f'{block}_shared')
    sp,_=reg_select(cells,shared,strong_sp,f'{block}_specific')
    folder=root(budget,'regression',block);freeze=folder/'all_policies_frozen.json'
    atomic_json(freeze,dict(status='FROZEN_BEFORE_QUERY',Shared=sh,Specific=sp,source_candidate_origin=str(origin),
        source_calibration='grouped original-row OOF heads for five; disjoint-TRAIN auxiliary validation models for seven',
        query_labels_used_for_fit_or_selection=False,unchanged_Base_and_query_conditions=True,seed=0))
    rows=[]
    for c in expert.h.risk.load(expert.h.source.OUT,'query',budget,block):
        old=origin/'predictions'/str(budget)/c['key']
        with np.load(old) as z:
            assert np.array_equal(z['Base'],c['base']) and np.array_equal(z['y_target'],c['y'])
            b=z['Base'];sh0=z['Shared'];sp0=z['Specific']
        a=0. if sh['chosen'] is None else sh['chosen']['blend'];shared=(1-a)*b+a*sh0
        a=0. if sp['chosen'] is None else sp['chosen']['blend'];specific=(1-a)*shared+a*sp0
        rows+=write_query(c,b,shared,specific,json.loads(old.with_suffix('.json').read_text()),budget,'regression',block,freeze)
    pd.DataFrame(rows).to_csv(folder/'query_detail.csv',index=False)
    atomic_json(folder/'completion.json',dict(status='COMPLETE',conditions=len(rows)//3,source_only_selection=True))
    print('MODERATE_REGRESSION_COMPLETE',budget,block,sh['chosen'],sp['chosen'],flush=True)


def cls_select(cells,baseline,candidate,base,role):
    score=clean.parent.prior.score
    initial=score(cells,baseline)[0];reference=score(cells,base)[0]
    best=None;best_pred=baseline;trials=[];best_utility=-np.inf
    for blend in SMALL:
        pred={key:normalize((1-blend)*baseline[key]+blend*candidate[key]) for key in baseline}
        scores,utility=score(cells,pred)
        gains={key:scores[key]-reference[key] for key in scores}
        eligible=all(scores[key]>initial[key]+1e-10 and 0<gains[key]<=.005+1e-10 for key in scores)
        trials.append(dict(blend=blend,source_scores=scores,source_gains_over_Base=gains,eligible=eligible))
        if eligible and utility>best_utility+1e-10:
            best_utility=utility;best=dict(blend=blend);best_pred=pred
    return dict(chosen=best,source_baseline=initial,source_Base_scores=reference,trials=trials),best_pred


def run_clean(budget):
    assert budget in [1024,4096,8192]
    clean.parent.configure();cells=clean.source_cells(budget)
    old_sh,previous=clean.parent_source(cells,budget)
    policy=json.loads((clean.OUT/'policies'/str(budget)/'classification/clean/all_policies_frozen.json').read_text())
    best=policy['Specific'];strong_sp=old_sh
    if best is not None:
        strong_sp={}
        for c in cells:
            p=clean.candidate(c,previous[c['key']],best['source'],best['prior_power'],best['temperature']);a=best['blend']
            strong_sp[c['key']]=normalize((1-a)*old_sh[c['key']]+a*p)
    base={c['key']:c['base'] for c in cells}
    save_source(cells,base,old_sh,strong_sp,budget,'classification','clean')
    sh,shared=cls_select(cells,base,old_sh,base,'Shared')
    sp,_=cls_select(cells,shared,strong_sp,base,'Specific')
    folder=root(budget,'classification','clean');freeze=folder/'all_policies_frozen.json'
    atomic_json(freeze,dict(status='FROZEN_BEFORE_QUERY',Shared=sh,Specific=sp,
        source_only_selection=True,residual_blend_grid=SMALL,source_small_gain_cap=.005,
        source_calibration='v71 disjoint class-covered TRAIN holdout, parent gates exclude whole target dataset',
        old_fresh_Native1_replay_failure_preserved=True,seed=0,query_labels_used_for_fit_or_selection=False))
    rows=[]
    for c in clean.parent.load('query',budget):
        old=clean.OUT/'predictions'/str(budget)/c['key']
        with np.load(old) as z:
            b=z['Base'];sh0=z['Shared'];sp0=z['Specific'];assert np.array_equal(b,c['base']) and np.array_equal(z['y_target'],c['y'])
        a=0. if sh['chosen'] is None else sh['chosen']['blend'];shared=normalize((1-a)*b+a*sh0)
        a=0. if sp['chosen'] is None else sp['chosen']['blend'];specific=normalize((1-a)*shared+a*sp0)
        rows+=write_query(c,b,shared,specific,json.loads(old.with_suffix('.json').read_text()),budget,'classification','clean',freeze)
    pd.DataFrame(rows).to_csv(folder/'query_detail.csv',index=False)
    atomic_json(folder/'completion.json',dict(status='COMPLETE',conditions=72,source_only_selection=True))
    print('MODERATE_CLASSIFICATION_CLEAN_COMPLETE',budget,sh['chosen'],sp['chosen'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['clean','regression'])
    parser.add_argument('--budget',type=int,required=True);parser.add_argument('--block',choices=['five','seven'])
    args=parser.parse_args()
    if args.phase=='clean':run_clean(args.budget)
    else:run_regression(args.budget,args.block)
