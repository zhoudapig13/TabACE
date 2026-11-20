"""Evaluation-label oracle bounds for diagnosis, never deployable results.

Optimistic convex-hull correctness is determined analytically when possible
and by a small LP for ambiguous multiclass errors. Classwise interval AUC is
an oracle upper bound; multiclass classwise optima need not be jointly feasible.
No predictor, policy, benchmark, or final experiment score is modified.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import roc_auc_score

import semisynthetic_abcd_v92 as source
from prepare_data import atomic_json

OUT=source.RUN/'results/convex_bank_diagnosis_v95'


def feasible_correct(experts, truth):
    """Weak correct-or-tie feasibility; strict negative means impossible."""
    if np.any(experts.argmax(1)==truth):
        return True, 'one_expert_correct'
    differences=experts-experts[:,truth,None]
    others=np.arange(experts.shape[1])!=truth
    if np.any(np.min(differences[:,others],axis=0)>1e-9):
        return False, 'common_dominating_wrong_class'
    if experts.shape[1]==2:
        return bool(experts[:,truth].max()>=.5-1e-9), 'binary_interval'
    members=len(experts)
    objective=np.r_[np.zeros(members),-1.]
    inequalities=np.column_stack([differences[:,others].T,np.ones(np.sum(others))])
    equality=np.r_[np.ones(members),0.][None]
    result=linprog(objective,A_ub=inequalities,b_ub=np.zeros(np.sum(others)),
                   A_eq=equality,b_eq=[1.],bounds=[(0.,1.)]*members+[(-1.,1.)],method='highs')
    assert result.success, result.message
    return bool(result.x[-1]>=-1e-9), 'multiclass_LP'


def one(path_text):
    from pathlib import Path
    path=Path(path_text);meta=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as z:
        base=z['base'];bank=z['members'].copy();bank[1]=z['density'];y=z['y_target'].astype(int);bootstrap=z['bootstrap']
    output=[]
    for matched in (True,False):
        members=bank.copy()
        if not matched:members[3]=bootstrap
        experts=np.concatenate([base[None],members])
        K=base.shape[1];N=len(y);idx=np.arange(N)
        b=np.clip(base,1e-12,1.)
        entropy=np.clip(-np.sum(b*np.log(b),axis=1)/np.log(K),0,1)
        disagreement=.5*np.sum(np.abs(members-base[None]),axis=2).max(0)
        relaxed=np.maximum(entropy,disagreement)
        base_true=base[idx,y];bank_true=members[:,idx,y].max(0)
        entropy_upper=base_true+entropy*np.maximum(bank_true-base_true,0)
        relaxed_upper=base_true+relaxed*np.maximum(bank_true-base_true,0)
        base_correct=base.argmax(1)==y
        possible=[];routes={}
        for i in range(N):
            ok,route=feasible_correct(experts[:,i,:],int(y[i]))
            possible.append(ok);routes[route]=routes.get(route,0)+1
        possible=np.array(possible)
        oracle_auc=[];baseline_auc=[]
        low,high=experts.min(0),experts.max(0)
        for k in range(K):
            binary=y==k
            if binary.any() and (~binary).any():
                score=np.where(binary,high[:,k],low[:,k])
                oracle_auc.append(roc_auc_score(binary,score));baseline_auc.append(roc_auc_score(binary,base[:,k]))
        errors=~base_correct
        output.append({'task_id':meta['task_id'],'data_id':meta['data_id'],'rho':meta['rho'],
            'bank':'with_matched_feature_drop' if matched else 'strict_no_feature_drop',
            'classes':K,'queries':N,'Base_accuracy':float(base_correct.mean()),
            'Base_auc':float(np.mean(baseline_auc)),
            'oracle_convex_correct_or_tie_upper_bound':float(possible.mean()),
            'Base_error_fraction_impossible_for_any_convex_mixture':float((errors&~possible).sum()/max(errors.sum(),1)),
            'all_query_fraction_impossible_for_any_convex_mixture':float((~possible).mean()),
            'Base_error_fraction_certified_impossible_under_entropy_gate':float((errors&(entropy_upper<1/K-1e-9)).sum()/max(errors.sum(),1)),
            'Base_error_fraction_certified_impossible_under_disagreement_gate':float((errors&(relaxed_upper<1/K-1e-9)).sum()/max(errors.sum(),1)),
            'oracle_classwise_interval_auc_upper_bound':float(np.mean(oracle_auc)),
            'LP_queries':routes.get('multiclass_LP',0)})
    return output


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[]
    for p in sorted((source.OUT/'banks/five_corrected').glob('*/*.npz')):
        m=json.loads(p.with_suffix('.json').read_text())
        if m['new_semisynthetic_condition'] and m['rho']:
            paths.append(str(p))
    assert len(paths)==288
    atomic_json(OUT/'plan.json',{'scope':'diagnostic oracle bounds, not methods or measured improvements',
        'evaluation_labels_used_to_define_oracle_bounds':True,'oracle_weights_used_in_reported_ABCD':False,
        'predictors_and_frozen_inputs_changed':False,'candidate_bank':'v92 density-adapted, matched vs strict no feature deletion',
        'weak_tie_feasibility_is_an_optimistic_upper_bound':True,
        'entropy_probability_barrier_is_only_a_sufficient_impossibility_certificate_in_multiclass':True,
        'multiclass_classwise_auc_bound_need_not_be_jointly_attainable':True,
        'code_sha256':source.original.sha(__file__),'Markdown_written':False})
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(one,p) for p in paths]),1):
            rows.extend(f.result())
            if i%48==0:print('BANK_DIAGNOSED',i,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'detail.csv',index=False)
    numeric=[c for c in frame.select_dtypes('number').columns if c not in ['task_id','data_id','rho','classes','queries']]
    summary=frame.groupby(['bank','data_id'])[numeric].mean().groupby('bank').mean()
    summary.to_csv(OUT/'summary.csv')
    frame.groupby(['bank','rho'])[numeric].mean().to_csv(OUT/'by_strength.csv')
    assert len(frame)==576
    atomic_json(OUT/'completion.json',{'status':'COMPLETE','conditions':576,'datasets':72,
        'LP_queries':int(frame.LP_queries.sum()),'oracle_bound_not_a_performance_result':True,
        'actual_ABCD_predictions_unchanged':True,'Markdown_bytes':(source.original.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size})
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    # Mixed expert decisions can be correct even if no individual argmax is.
    x=np.array([[.4,.25,.35],[.25,.4,.35]])
    assert feasible_correct(x,2)[0]
    assert not feasible_correct(np.array([[.8,.1,.1],[.7,.2,.1]]),2)[0]
    main()
