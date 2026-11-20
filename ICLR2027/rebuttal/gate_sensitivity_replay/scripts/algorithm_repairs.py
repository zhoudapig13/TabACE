"""Source-calibrated high-confidence repair and median-centered output safeguard.

Repairs are new algorithm variants. Source calibration uses dataset-level outer
training folds only; unchanged outer-test labels are used solely for evaluation.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import torch
from replay_diagnostics import OUT,ROOT,cells,classify,regress,softmax,MAGNITUDES,write_json

REPAIR=OUT/'algorithm_repairs_v1'


def prepare():
    plan=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        parent_plan_sha256=hashlib.sha256((OUT/'frozen_plan.json').read_bytes()).hexdigest(),
        classification=dict(base_confidence_threshold=.9,alternative_confidence_threshold=.6,
            eligibility='Base and aggregate top classes differ, with the two confidence thresholds satisfied',
            gate='alpha_new=max(alpha_original, .99*sigmoid(features @ theta)) on eligible rows; otherwise alpha_original',
            features=['intercept','alternative_confidence-minus-Base_confidence','alternative_vote_fraction',
                      'context_support','Base_aggregate_total_variation','alternative_margin-minus-Base_margin'],
            calibration='Minimize weighted source-calibration log loss; one head per dataset-level outer fold',
            regularization=.01,initial_intercept=-4.,all_other_initial_coefficients=0.,
            coefficient_bounds=[-10.,10.],hyperparameter_search=False,
            original_weighting_coefficients_frozen=True),
        regression=dict(guard='Before fusion, clip each branch to query_median +/-3*source_std',
            width_source_sd=3.,median_computed_without_labels=True,
            Base_output_clipped=False,recompute_weights_support_and_gate=True,
            stress_magnitudes_source_sd=MAGNITUDES,each_branch_and_both_signs=True),
        new_algorithms_separate_from_original=True,backbone_calls=0,evaluation_labels_used_for_fitting=False,
        input_scope='Appendix-rule reconstructed caches, not recovered original paper archives',markdown_written=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=REPAIR/'frozen_plan.json'
    if not path.exists():write_json(path,plan)
    print('REPAIR_PLAN_FROZEN',flush=True)


def components(base,bank,theta):
    classes=base.shape[1];a=np.clip(bank,1e-12,1.)
    q=(1+(a*np.log(a)).sum(2)/np.log(classes)).T
    s=(1-.5*np.abs(bank-bank.mean(0)[None]).sum(2)).T
    weights=softmax(theta[:5]+q*theta[5:10]+s*theta[10:15],1)
    aggregate=np.einsum('nm,mnc->nc',weights,bank)
    support=(weights*q*s).sum(1)
    b=np.clip(base,1e-12,1.);u=-(b*np.log(b)).sum(1)/np.log(classes)
    alpha=theta[-1]*u*support
    cb=base.max(1);ca=aggregate.max(1)
    mb=np.sort(base,axis=1)[:,-1]-np.sort(base,axis=1)[:,-2]
    ma=np.sort(aggregate,axis=1)[:,-1]-np.sort(aggregate,axis=1)[:,-2]
    choice=aggregate.argmax(1)
    vote=(bank.argmax(2)==choice[None]).mean(0)
    distance=np.clip(.5*np.abs(base-aggregate).sum(1),0.,1.)
    features=np.c_[np.ones(len(base)),ca-cb,vote,support,distance,ma-mb]
    eligible=(cb>=.9)&(ca>=.6)&(base.argmax(1)!=choice)
    return aggregate,alpha,features,eligible


def fit_head(features,base_true,alternative_true,old_alpha,weights):
    initial=np.r_[-4.,np.zeros(features.shape[1]-1)]
    if not len(features):return initial,dict(n_calibration_disagreements=0,optimizer_success=True)
    weight=weights/weights.sum()
    X,B,A,L,W=[torch.as_tensor(x,dtype=torch.float64) for x in
               [features,base_true,alternative_true,old_alpha,weight]]
    prior=torch.as_tensor(initial,dtype=torch.float64)
    def objective(parameters):
        c=torch.tensor(parameters,dtype=torch.float64,requires_grad=True)
        alpha=torch.maximum(L,.99*torch.sigmoid(X@c))
        p=(1-alpha)*B+alpha*A
        loss=-(W*torch.log(p.clamp(min=1e-12))).sum()+.01*((c-prior)**2).mean()
        loss.backward();return float(loss.detach()),c.grad.detach().numpy()
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(-10,10)]*len(initial),
                    options={'maxiter':200,'ftol':1e-10})
    assert np.isfinite(result.x).all()
    return result.x,dict(n_calibration_disagreements=len(features),optimizer_success=bool(result.success),
                        optimizer_message=str(result.message),weighted_regularized_loss=float(result.fun))


def classification(backbone,track):
    dataset=[]
    for _,meta,base,bank,y,sm,ss,theta,fold,reference,path in cells(backbone,track,'classification'):
        dataset.append(dict(meta=meta,base=base,bank=bank,y=y,fold=fold,theta=theta,reference=reference))
    parent=json.loads((OUT/'frozen_plan.json').read_text())
    case=next(c for c in parent['cases'] if (c['backbone'],c['track'],c['task'])==(backbone,track,'classification'))
    counts=[];performance=[];head_reports=[]
    for record in case['policies']:
        policy=json.loads((ROOT/record['path']).read_text());theta=np.asarray(policy['parameters'])
        train_ids=set(policy['train_data_ids']);test_ids=set(policy['test_data_ids'])
        assert not train_ids.intersection(test_ids)
        train=[c for c in dataset if c['meta']['data_id'] in train_ids]
        test=[c for c in dataset if c['meta']['data_id'] in test_ids]
        condition_counts=defaultdict(int)
        for c in train:condition_counts[c['meta']['data_id']]+=1
        parts=[]
        for c in train:
            ag,alpha,f,eligible=components(c['base'],c['bank'],theta)
            y=c['y'].astype(int);n=len(y)
            w=np.full(n,1./(len(train_ids)*condition_counts[c['meta']['data_id']]*n))
            parts.append((f[eligible],c['base'][np.arange(n),y][eligible],
                          ag[np.arange(n),y][eligible],alpha[eligible],w[eligible]))
        inputs=[np.concatenate([p[i] for p in parts],axis=0) for i in range(5)]
        coefficients,audit=fit_head(*inputs)
        head=dict(backbone=backbone,track=track,outer_fold=policy['fold'],source_data_ids=sorted(train_ids),
                  evaluation_data_ids=sorted(test_ids),head_coefficients=coefficients.tolist(),
                  original_policy_sha256=record['sha256'],calibration=audit,
                  evaluation_labels_used_for_fit=False,thresholds_fixed_before_calibration=True)
        write_json(REPAIR/backbone/track/'classification'/f'fold_{policy["fold"]}_repair_head.json',head)
        head_reports.append(head)
        for c in test:
            m,b,k,y=c['meta'],c['base'],c['bank'],c['y']
            ag,alpha,f,eligible=components(b,k,theta)
            original,_=classify(b,k,theta)
            assert abs(np.mean(original.argmax(1)==y)-c['reference']['accuracy'])<1e-12
            repaired_alpha=alpha.copy()
            repaired_alpha[eligible]=np.maximum(alpha[eligible],.99*expit(f[eligible]@coefficients))
            repaired=(1-repaired_alpha[:,None])*b+repaired_alpha[:,None]*ag
            repaired=np.maximum(repaired,1e-12);repaired/=repaired.sum(1,keepdims=True)
            assert np.array_equal(repaired[~eligible],original[~eligible])
            before=b.argmax(1);cb=b.max(1);correct=before==y
            for method,p in [('Original',original),('HC_repair',repaired)]:
                pred=p.argmax(1);right=pred==y
                performance.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                    outer_fold=policy['fold'],method=method,n=len(y),base_accuracy=float(correct.mean()),
                    accuracy=float(right.mean()),log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y.astype(int)],1e-12)).mean()),
                    gate_eligible_queries=int(eligible.sum()),low_confidence_outputs_unchanged=True))
                for threshold in [0.,.8,.9,.95,.99]:
                    selected=cb>=threshold;wrong=selected&~correct;good=selected&correct
                    counts.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                        method=method,confidence_threshold=threshold,n_queries=len(y),n_high_confidence=int(selected.sum()),
                        base_errors=int(wrong.sum()),corrected_errors=int((wrong&right).sum()),base_correct=int(good.sum()),
                        new_errors=int((good&~right).sum()),changed_predictions=int((selected&(before!=pred)).sum())))
        print('HC_REPAIR_FOLD',backbone,track,policy['fold'],audit,flush=True)
    dest=REPAIR/backbone/track/'classification';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(counts).to_csv(dest/'confidence_counts.csv',index=False)
    pd.DataFrame(performance).to_csv(dest/'performance.csv',index=False)
    write_json(dest/'audit.json',dict(dataset_folds_disjoint=True,all_low_confidence_outputs_bitwise_unchanged=True,
        n_datasets=len({c['meta']['data_id'] for c in dataset}),episodes=len(dataset),heads=head_reports,
        original_policy_replays_verified=True,evaluation_labels_used_for_fitting=False))
    print('HC_REPAIR_COMPLETE',backbone,track,flush=True)


def guarded(base,bank,theta,sm,ss):
    center=np.median(bank,axis=0)
    bounded=center[None]+np.clip(bank-center[None],-3*ss,3*ss)
    p,weights,alpha=regress(base,bounded,theta,sm,ss)
    assert np.all(np.abs(bounded-center[None])<=3*ss+1e-8*max(1,ss))
    return p,bounded


def regression(backbone,track):
    samples=defaultdict(list);scores=[];change_counts=[];condition_stress=[]
    for i,m,b,k,y,sm,ss,theta,fold,ref,path in cells(backbone,track,'regression'):
        original=regress(b,k,theta,sm,ss)[0];repaired,bounded=guarded(b,k,theta,sm,ss)
        scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        assert abs(np.sqrt(np.mean((original-y)**2))/scale-ref['nrmse'])<1e-8*max(1,ref['nrmse'])
        for name,p in [('Original',original),('Median_guard3sd',repaired)]:
            scores.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                method=name,n=len(y),r2=1-float(np.sum((p-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                nrmse=float(np.sqrt(np.mean((p-y)**2))/scale)))
        change_counts.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],n=len(y),
            alternative_values=k.size,clipped_values=int(np.count_nonzero(k!=bounded)),
            changed_final_predictions=int(np.count_nonzero(np.abs(repaired-original)>1e-10*ss))))
        median=np.median(k,axis=0)
        for magnitude in MAGNITUDES:
            worst=np.zeros(len(y))
            for branch in range(5):
                for sign in [-1.,1.]:
                    modified=k.copy();modified[branch]=median+sign*magnitude*ss
                    p,_=guarded(b,modified,theta,sm,ss)
                    worst=np.maximum(worst,np.abs(p-repaired)/ss)
            samples[magnitude].append(worst)
            condition_stress.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                method='Median_guard3sd',magnitude_source_sd=magnitude,n=len(y),
                mean_displacement_source_sd=float(worst.mean()),p95_displacement_source_sd=float(np.quantile(worst,.95)),
                max_displacement_source_sd=float(worst.max())))
        if i%100==0:print('MEDIAN_GUARD',backbone,track,i,flush=True)
    dest=REPAIR/backbone/track/'regression';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(scores).to_csv(dest/'performance.csv',index=False)
    pd.DataFrame(change_counts).to_csv(dest/'clipping_counts.csv',index=False)
    pd.DataFrame(condition_stress).to_csv(dest/'stress_by_condition.csv',index=False)
    pooled=[]
    for magnitude,parts in samples.items():
        d=np.concatenate(parts)
        pooled.append(dict(backbone=backbone,track=track,method='Median_guard3sd',magnitude_source_sd=magnitude,
            n_query_conditions=len(d),mean_displacement_source_sd=float(d.mean()),median_displacement_source_sd=float(np.median(d)),
            p95_displacement_source_sd=float(np.quantile(d,.95)),p99_displacement_source_sd=float(np.quantile(d,.99)),
            max_displacement_source_sd=float(d.max())))
    pd.DataFrame(pooled).to_csv(dest/'stress_pooled.csv',index=False)
    write_json(dest/'audit.json',dict(source_scale_only=True,target_labels_used_for_guard=False,
        guard_width=3.,guard_location='alternative predictions before weighting',Base_unclipped=True,
        query_median_computed_without_labels=True,original_policy_replay_verified=True))
    print('MEDIAN_GUARD_COMPLETE',backbone,track,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--stage',choices=['classification','regression']);p.add_argument('--backbone');p.add_argument('--track')
    a=p.parse_args();torch.set_num_threads(1)
    if a.prepare:prepare()
    elif a.stage=='classification':classification(a.backbone,a.track)
    elif a.stage=='regression':regression(a.backbone,a.track)
