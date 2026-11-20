"""Dataset-OOF adaptation of combiners on the fixed internal benchmark.

The benchmark is fixed BEFORE these fits. No ranking versus historical Shared
or requested score intervals participates in policy fitting or selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_data import RUN,atomic_json
from run_tabicl import metrics
from internal_seven_fixed_strength import OUT,SELECTED
from fit_robust_six import train,predict,loss,clean_scores
from fit_moderate_random_dense import outputs,score,clean
from run_internal_seven_ensembles import moderate_bank


def load():
    paths=sorted((OUT/'ensembles').glob('*/*.npz'));assert len(paths)==980
    cells=[];seen={}
    for p in paths:
        m=json.loads(p.with_suffix('.json').read_text());seen[m['task_id']]=m
        with np.load(p) as z:
            cells.append({'meta':m,'path':p,'base':z['prediction__Base'],'bank_raw':z['raw_bank'],
                          'bank_adapt':z['adapted_bank'],'y':z['y_target'],
                          'fold':int(m['outer_fold']),'clean_weight':1.})
    for task,m in seen.items():
        p=RUN/'results/robust_six_all/tabicl/seven_reimplemented/classification'/str(task)/'covariate_0.npz'
        old=json.loads(p.with_suffix('.json').read_text())
        with np.load(p) as z:
            cells.append({'meta':old,'path':p,'base':z['base'],'bank_raw':z['raw_bank'],
                          'bank_adapt':z['bank'],'y':z['y_target'],
                          'fold':int(m['outer_fold']),'clean_weight':7.})
    assert len(cells)==1015
    for c in cells:
        c['outputs']=np.stack([outputs(moderate_bank(c['bank_raw'],c['bank_adapt'],pattern)) for pattern in range(6)])
    return cells


def persist(test,method,fold,predictions,policy):
    rows=[]
    for c,pred in zip(test,predictions):
        m=c['meta'];observed=metrics(c['y'],pred,'classification',1.)
        rows.append({'task_id':m['task_id'],'data_id':m['data_id'],'family':m['family'],'rho':m['rho'],
                     'outer_fold':fold,'method':method,**observed})
        out=OUT/'recalibrated_predictions'/method/str(m['task_id'])/f"{m['family']}_{m['rho']}.npz"
        out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,prediction=pred,y_target=c['y']);tmp.replace(out)
    atomic_json(OUT/'recalibrated_policies'/f'fold_{fold}_{method}.json',policy|{'rows':rows})


def fit(fold,methods):
    frozen=SELECTED.read_bytes();cells=load()
    training=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
    train_ids={int(c['meta']['data_id']) for c in training};test_ids={int(c['meta']['data_id']) for c in test}
    assert train_ids.isdisjoint(test_ids)
    inner=min(c['fold'] for c in training);inner_train=[c for c in training if c['fold']!=inner]
    validation=[c for c in training if c['fold']==inner]
    base_clean=clean_scores(validation,'GES',3,None)
    common={'status':'COMPLETE','fold':fold,'seed':0,'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),
       'benchmark_frozen_before_method_fitting':True,'benchmark_parameters_changed':False,
       'method_selection_uses_historical_Shared_or_score_bounds':False,'outer_test_labels_used_for_fitting':False,
       'original_Clean_five_results_overwritten':False,'seven_Clean_is_diagnostic_only':True,'context_count':6,
       'internal_development_only':True,'benchmark_calibration_used_development_labels':True}
    for method in methods:
        out=OUT/'recalibrated_policies'/f'fold_{fold}_{method}.json'
        if out.exists():continue
        trials=[];best=None
        if method in ['Random','DENSE']:
            calibration_clean=clean(training)
            for pattern in range(6):
                for rule in ([0] if method=='Random' else range(3)):
                    observed=clean(training,pattern,rule)
                    feasible=all(observed[k]>=calibration_clean[k]-.005 for k in observed)
                    value=score(training,pattern,rule)
                    trials.append({'pattern':pattern,'rule':rule,'calibration_score':value,'Clean':observed,'feasible':feasible})
                    if feasible and (best is None or value>best[0]):best=(value,pattern,rule)
            if best is None:
                predictions=[c['base'] for c in test];selected={'Base_fallback':True}
            else:
                _,pattern,rule=best;predictions=[c['outputs'][pattern,rule] for c in test]
                selected={'pattern':pattern,'rule':rule,'Base_fallback':False}
            persist(test,method,fold,predictions,common|selected|{'method':method,'trials':trials,
                'selection':'other-four-dataset-fold Acc/AUC; Clean tolerance .005; at most two modified members',
                'maximum_modified_members':2})
        else:
            regs=[0.,.001,.01,.1] if method=='Convex' else [.001,.01,.1] if method=='Loglinear' else [0.]
            for mode in [0,1,2]:
                for reg in regs:
                    model=train(inner_train,method,mode,reg)
                    value=loss(validation,method,mode,model,'acc_auc')
                    observed=clean_scores(validation,method,mode,model)
                    feasible=all(observed[k]>=base_clean[k]-.005 for k in observed)
                    trials.append({'mode':mode,'regularization':reg,'inner_objective':value,'Clean':observed,'feasible':feasible})
                    if feasible and (best is None or value<best[0]):best=(value,mode,reg)
            _,mode,reg=best if best is not None else (None,3,0.)
            model=train(training,method,mode,reg) if mode!=3 else None
            predictions=[predict(c,method,mode,model) for c in test]
            persist(test,method,fold,predictions,common|{'method':method,'model':model,'mode':mode,
                'regularization':reg,'inner_validation_fold':inner,'trials':trials,
                'selection':'nested dataset-fold Acc/AUC; inner Clean tolerance .005'})
        assert SELECTED.read_bytes()==frozen,'Benchmark changed during method fitting'
        print('OOF_RECALIBRATED',fold,method,flush=True)


def report():
    old=json.loads((OUT/'latest_ensemble_summary.json').read_text());policies={}
    for p in (OUT/'recalibrated_policies').glob('*.json'):
        d=json.loads(p.read_text());policies.setdefault(d['method'],[]).append(d)
    rows=[]
    for r in old['records']:
        short={'Random6_adapted':'Random','DENSE6_adapted':'DENSE','GES6_adapted':'GES',
               'Convex6_adapted':'Convex','Loglinear6_adapted':'Loglinear'}.get(r['method'])
        if short is None:rows.append(r|{'policy_stage':'existing frozen strategy'});continue
        ps=policies.get(short,[])
        assert len(ps)==5,(short,len(ps))
        values=[x for p in ps for x in p['rows'] if x['rho']>0]
        zeros=[x for p in ps for x in p['rows'] if x['rho']==0]
        assert len(values)==980 and len(zeros)==35
        avg={k:float(np.mean([x[k] for x in values])) for k in ['accuracy','auc']}
        clean_avg={k:float(np.mean([x[k] for x in zeros])) for k in ['accuracy','auc']}
        rows.append(r|{'seven_internal_before_recalibration':r['seven_internal'],'seven_internal':avg,
                       'seven_Clean_diagnostic':clean_avg,'policy_stage':'dataset-OOF recalibrated on internal benchmark'})
    result=old|{'records':rows,'recalibration_status':'COMPLETE','all_first_round_variants_retained':True,
                'benchmark_unchanged_during_recalibration':True}
    atomic_json(OUT/'latest_recalibrated_summary.json',result)
    print(json.dumps({'status':result['status'],'records':rows},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',action='store_true');p.add_argument('--fold',type=int)
    p.add_argument('--methods',nargs='+',default=['Random','DENSE','GES','Convex','Loglinear']);a=p.parse_args()
    if a.report:report()
    else:
        assert a.fold in range(5)
        fit(a.fold,a.methods)
