"""Audit saved arithmetic and latest user bounds, keeping all48 checks visible.

This is not a fresh-backbone reproducibility certificate. The outstanding
CIFAR TRAIN replay failure is explicitly preserved in the output.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import moderate_adaptation_v75 as e
from prepare_data import atomic_json,protocol,RUN
from run_tabicl import metrics,normalize


def pair(task,base,shared,specific):
    if task=='classification':
        a,b=normalize((1-shared)*base[0]+shared*base[1]),None
        b=normalize((1-specific)*a+specific*base[2])
        return a,b
    a=(1-shared)*base[0]+shared*base[1]
    return a,(1-specific)*a+specific*base[2]


def audit_completed_initial_groups():
    query_count=source_count=0;maximum=0.
    source_scores=[]
    for budget in e.BUDGETS:
        for task,blocks in [('classification',['clean']),('regression',['five','seven'])]:
            if task=='classification' and budget==256:continue
            for block in blocks:
                folder=e.root(budget,task,block);freeze=folder/'all_policies_frozen.json'
                if not (folder/'completion.json').exists():continue
                policy=json.loads(freeze.read_text());sh=policy['Shared'];sp=policy['Specific']
                a=0. if sh['chosen'] is None else sh['chosen']['blend']
                b=0. if sp['chosen'] is None else sp['chosen']['blend']
                if task=='classification':
                    raw_cells=e.clean.source_cells(budget);cells={c['key']:c for c in raw_cells}
                else:
                    raw_cells=e.expert.h.risk.load(e.expert.h.source.OUT,'validation',budget,block);cells={c['key']:c for c in raw_cells}
                candidates={};truth={};new_shared={};new_specific={};base={}
                for key,c in cells.items():
                    path=e.OUT/'source_candidates'/str(budget)/key;record=json.loads(path.with_suffix('.json').read_text())
                    with np.load(path) as z:
                        p=[z[k] for k in ['base','strong_shared','strong_specific']]
                        assert np.array_equal(z['y_validation'],c['y']) and np.array_equal(p[0],c['base'])
                        assert record['source_calibration_y_sha256']==protocol.array_sha256(c['y'])
                        assert all(record['source_prediction_hashes'][name]==protocol.array_sha256(v) for name,v in zip(['base','strong_shared','strong_specific'],p))
                        assert all(np.isfinite(v).all() for v in p)
                    if task=='regression':
                        cal=e.expert.h.calibration(c['meta'],block);ids=cal['validation_original_ids']
                        assert len(ids)==len(np.unique(ids))==len(c['y'])
                        folds=e.expert.h.folds(c['meta'],block,len(ids))
                        for f in range(4):assert not np.intersect1d(ids[folds==f],ids[folds!=f]).size
                    else:
                        part=e.clean.partition(c['meta'],e.clean.full.reference.scaling.prepare('classification','five',c['meta']))
                        assert not np.intersect1d(part['source_original_ids'],part['validation_original_ids']).size
                    base[key]=p[0];candidates[key]=p
                    new_shared[key],new_specific[key]=pair(task,p,a,b);source_count+=1
                scorer=e.clean.parent.prior.score if task=='classification' else e.expert.score
                for role,recipe,anchor,predictions in [('Shared',sh,base,new_shared),('Specific',sp,new_shared,new_specific)]:
                    initial=scorer(raw_cells,anchor)[0]
                    assert all(abs(initial[k]-recipe['source_baseline'][k])<1e-12 for k in initial)
                    if recipe['chosen'] is not None:
                        trial=next(t for t in recipe['trials'] if t['blend']==recipe['chosen']['blend'])
                        actual=scorer(raw_cells,predictions)[0]
                        assert all(abs(actual[k]-trial['source_scores'][k])<1e-12 for k in actual)
                        source_scores.append(dict(budget=budget,task=task,block=block,role=role,score=actual))
                query=e.clean.parent.load('query',budget) if task=='classification' else e.expert.h.risk.load(e.expert.h.source.OUT,'query',budget,block)
                origin=e.clean.OUT if task=='classification' else Path(policy['source_candidate_origin'])
                for c in query:
                    old=origin/'predictions'/str(budget)/c['key']
                    with np.load(old) as z:
                        original=[z[k] for k in ['Base','Shared','Specific']]
                        assert np.array_equal(z['y_target'],c['y']) and np.array_equal(original[0],c['base'])
                    expected=pair(task,original,a,b)
                    out=e.OUT/'predictions'/str(budget)/c['key'];record=json.loads(out.with_suffix('.json').read_text())
                    assert freeze.stat().st_mtime<=out.stat().st_mtime
                    with np.load(out) as z:
                        assert np.array_equal(z['Base'],c['base']) and np.array_equal(z['y_target'],c['y'])
                        for name,p in zip(['Shared','Specific'],expected):
                            error=float(np.max(np.abs(p-z[name])));maximum=max(maximum,error);assert error<1e-12
                        for name in ['Base','Shared','Specific']:
                            values=metrics(c['y'],z[name],task,c['meta']['scale'])
                            assert all((v is None and record['methods'][name][k] is None) or abs(v-record['methods'][name][k])<1e-12 for k,v in values.items())
                    query_count+=1
    frozen_count=0
    for budget in e.BUDGETS:
        for task,block,parent in [('classification','five',RUN/'results/tabace_probability_head_v51'),('regression','clean',RUN/'results/tabace_clean_calibration_v59')]:
            destination=e.OUT/'predictions'/str(budget)/task/block;origin=parent/'predictions'/str(budget)/task/block
            assert destination.resolve()==origin.resolve()
            for path in destination.glob('*/*.json'):
                row=json.loads(path.read_text())
                with np.load(path.with_suffix('.npz')) as z:
                    for name in ['Base','Shared','Specific']:
                        measured=metrics(z['y_target'],z[name],task,row['scale'])
                        assert all((v is None and row['methods'][name][k] is None) or abs(v-row['methods'][name][k])<1e-12 for k,v in measured.items())
                frozen_count+=1
    report=dict(status='PASS',scope='Saved source-selection scores, Source ID partitions, query arithmetic, exact metric recomputation, and frozen block references',
        new_source_cases_verified=source_count,new_query_cases_replayed=query_count,frozen_block_cases_verified=frozen_count,
        maximum_fusion_error=maximum,source_selected_scores=source_scores,
        classification_five_and_regression_clean_bitwise_unchanged=True,Base_bitwise_unchanged=True,
        query_labels_used_for_fit_or_selection=False,policies_frozen_before_query_outputs=True,
        fresh_backbone_reproducibility_verified=False,
        outstanding_fresh_replay_failure=str(e.clean.OUT/'raw_audit_failure.json'),
        not_full_goal_completion=True,Markdown_written=False,time_unix=time.time())
    atomic_json(e.OUT/'arithmetic_audit.json',report)
    print('MODERATE_INITIAL_ARITHMETIC_AUDIT_PASS',query_count,source_count,frozen_count,maximum,flush=True)


def report():
    checks=[];rows=[];missing=[];small_gain_flags=[]
    raw=e.expert.h.source.adaptation.raw
    for budget in e.BUDGETS:
        for task in ['classification','regression']:
            for block in ['clean','five','seven']:
                expected=len(raw.frozen.cohort(task,block))*len(raw.case_grid(block))
                paths=list((e.OUT/'predictions'/str(budget)/task/block).glob('*/*.json'))
                if len(paths)!=expected:
                    missing.append(dict(context=budget,task=task,block=block,completed=len(paths),expected=expected));continue
                records=[json.loads(p.read_text()) for p in paths]
                names=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
                means={name:{k:float(np.mean([r['methods'][name][k] for r in records])) for k in names} for name in ['Base','Shared','Specific']}
                for name,scores in means.items():rows.append(dict(context=budget,task=task,block=block,method=name,datasets=len(raw.frozen.cohort(task,block)),conditions=len(records),**scores))
                for metric in names:
                    direction=-1 if metric=='nrmse' else 1
                    v={m:means[m][metric] for m in means}
                    g1=direction*(v['Shared']-v['Base']);g2=direction*(v['Specific']-v['Shared'])
                    ordered=g1>1e-9 and g2>1e-9;bound=True;rule='strict oriented Base->Shared->Specific improvement'
                    if task=='classification' and block=='seven' and metric=='accuracy':
                        bound=g1<=.02+1e-9 and .02-1e-9<=g2<=.03+1e-9;rule='Shared-Base Acc<=0.02; Specific-Shared Acc in[0.02,0.03]'
                    elif task=='regression' and block=='five' and metric=='r2':
                        bound=.363-1e-9<=v['Shared']<=.563+1e-9;rule='Shared R2 in[0.363,0.563], Specific further improves'
                    elif task=='regression' and block=='seven' and metric=='r2':
                        bound=g1<=.3+1e-9 and g2<=.3+1e-9;rule='both R2 method-to-method gains<=0.3'
                    if task=='classification' and block=='clean':
                        small_gain_flags.append(dict(context=budget,metric=metric,Shared_gain=g1,Specific_gain_over_Base=direction*(v['Specific']-v['Base']),
                            working_source_target=.005,within_query_working_target=bool(max(g1,direction*(v['Specific']-v['Base']))<=.005+1e-9),
                            user_supplied_hard_numeric_Clean_cap=False))
                    checks.append(dict(context=budget,task=task,block=block,metric=metric,values=v,Shared_over_Base=g1,Specific_over_Shared=g2,
                        strict_order_pass=bool(ordered),latest_bounds_pass=bool(bound),rule=rule,pass_all=bool(ordered and bound)))
    if rows:pd.DataFrame(rows).to_csv(e.OUT/'averages.csv',index=False)
    result=dict(status='ALL_NUMERIC_CHECKS_MET_REQUIRES_PROVENANCE_REVIEW' if len(checks)==48 and not missing and all(c['pass_all'] for c in checks) else 'INCOMPLETE_OR_CONSTRAINTS_UNMET',
        required_metric_checks=48,completed_metric_checks=len(checks),passed_metric_checks=sum(c['pass_all'] for c in checks),
        missing=missing,failed_checks=[c for c in checks if not c['pass_all']],checks=checks,Clean_small_gain_working_target_flags=small_gain_flags,
        classification_five_frozen=True,regression_clean_frozen=True,development_only=True,
        fresh_backbone_reproducibility_verified=False,full_goal_complete_claim=False,time_unix=time.time(),Markdown_written=False)
    atomic_json(e.OUT/'goal_audit_latest_constraints.json',result)
    print('MODERATE_UPDATED_CONSTRAINT_CHECKS',result['passed_metric_checks'],'/48','missing',len(missing),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initial','report']);args=p.parse_args()
    audit_completed_initial_groups() if args.phase=='initial' else report()
