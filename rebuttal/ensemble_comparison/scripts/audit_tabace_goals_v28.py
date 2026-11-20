"""Aggregate paired results and audit every requested budget/metric/gain."""
import argparse
import json
import numpy as np
import pandas as pd
import tabace_context_reproduction_v28 as main
from prepare_data import atomic_json
from run_tabicl import metrics


def report():
    targets=json.loads((main.OUT/'paper_targets.json').read_text());rows=[];missing=[];checks=[]
    for budget in main.BUDGETS:
        for task in ['classification','regression']:
            for block in ['clean','five','seven']:
                root=main.OUT/'predictions'/str(budget)/task/block;records=[]
                expected=len(main.frozen.cohort(task,block))*len(main.case_grid(block))
                for path in root.glob('*/*.json'):
                    if not path.name.endswith('.tmp.json'):records.append(json.loads(path.read_text()))
                if len(records)!=expected:
                    missing.append(dict(context=budget,task=task,block=block,completed=len(records),expected=expected));continue
                metrics_names=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
                means={method:{k:float(np.mean([r['methods'][method][k] for r in records])) for k in metrics_names} for method in ['Base','Shared','Specific']}
                for method in means:rows.append(dict(context=budget,task=task,block=block,method=method,datasets=len(main.frozen.cohort(task,block)),conditions=len(records),**means[method]))
                for j,key in enumerate(metrics_names):
                    direction=-1 if key=='nrmse' else 1
                    p=targets[task][block];required_shared=max(0.,direction*(p['Shared'][j]-p['Base'][j]));required_specific=max(0.,direction*(p['Specific'][j]-p['Base'][j]))
                    gains=dict(Shared=direction*(means['Shared'][key]-means['Base'][key]),Specific=direction*(means['Specific'][key]-means['Base'][key]),
                               Specific_over_Shared=direction*(means['Specific'][key]-means['Shared'][key]))
                    strict=gains['Shared']>1e-9 and gains['Specific_over_Shared']>1e-9
                    increment=gains['Shared']>=required_shared-1e-9 and gains['Specific']>=required_specific-1e-9
                    checks.append(dict(context=budget,task=task,block=block,metric=key,values={m:means[m][key] for m in means},
                        oriented_gains=gains,required_Shared_over_Base=required_shared,required_Specific_over_Base=required_specific,
                        strict_order_pass=bool(strict),minimum_gain_pass=bool(increment),pass_all=bool(strict and increment),
                        provisional_Base_calibration=any(r['provisional_Base_calibration'] for r in records)))
    pd.DataFrame(rows).to_csv(main.OUT/'averages.csv',index=False)
    atomic_json(main.OUT/'goal_audit.json',dict(status='ALL_NUMERIC_TARGETS_MET' if not missing and checks and all(c['pass_all'] for c in checks) else 'INCOMPLETE_OR_TARGETS_UNMET',
        required_context_budgets=main.BUDGETS,required_metric_checks=48,completed_metric_checks=len(checks),
        passed_metric_checks=sum(c['pass_all'] for c in checks),missing=missing,checks=checks,
        methods='Raw observed-source constructors with held-out clean-TRAIN fusion calibration; reconstruction, not archived exact implementation',
        unresolved_Base_calibration=any(c['provisional_Base_calibration'] for c in checks)))
    if rows:print(pd.DataFrame(rows).to_string(index=False))
    print('GOAL_CHECKS',sum(c['pass_all'] for c in checks),'/',48,'missing blocks',len(missing))


def validation_capacity(budget):
    from tabace_fusion_v28 import load
    rows=[]
    for task in ['classification','regression']:
        for block in ['clean','five','seven']:
            cells=load('validation',budget,task,block)
            for name,count in [('Shared',5),('Specific',10)]:
                values=[]
                for c in cells:
                    bank=np.concatenate([c['base'][None],c['bank'][:count]],axis=0)
                    if task=='regression':
                        # A true bound for any nonnegative query-dependent
                        # mixture of these predictions. Not a deployable model.
                        best=np.clip(c['y'],bank.min(0),bank.max(0));score=metrics(c['y'],best,task,c['meta']['scale'])
                    else:
                        truth=c['y'].astype(int);p=bank[:,np.arange(len(truth)),truth]
                        score=dict(oracle_true_class_probability=float(p.max(0).mean()),
                            any_member_correct_fraction=float(np.any(bank.argmax(-1)==truth[None],axis=0).mean()))
                    values.append(score)
                avg={k:float(np.mean([r[k] for r in values])) for k in values[0]}
                rows.append(dict(context=budget,task=task,block=block,bank=name,role='held-out TRAIN diagnostic only',**avg))
    pd.DataFrame(rows).to_csv(main.OUT/f'validation_bank_capacity_{budget}.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['report','capacity']);p.add_argument('--budget',type=int,default=256);a=p.parse_args()
    report() if a.phase=='report' else validation_capacity(a.budget)
