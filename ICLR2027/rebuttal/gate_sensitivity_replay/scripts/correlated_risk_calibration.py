"""Source-calibrated attenuation under correlated alternative prediction errors.

The gate uses branch error second moments estimated on source-calibration data,
and a small tree model. Its exposure to common prediction bias is isolated by a
no-augmentation control. No query features or labels are added at inference.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse,hashlib,json,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from replay_diagnostics import OUT,ROOT,cells,regress,softmax,write_json

DEST=OUT/'correlated_risk_v4'
LAMBDAS=[0.,.03,.1]
GAMMAS=[0.,.25,.5,1.]
CAL_MAGNITUDES=[.5,1.,2.,5.]
RADIUS=12.
FEATURES=['original_alpha','log_absolute_disagreement','signed_disagreement','Base_location',
    'aggregate_location','log_abs_Base','log_abs_aggregate','support','log_branch_MAD',
    'log_branch_SD','log_max_residual','max_weight','weight_concentration',
    'relative_aggregate_error_second_moment','Base_aggregate_error_dependence','second_moment_optimal_alpha']

def prepare():
    path=DEST/'frozen_plan.json'
    if path.exists():raise RuntimeError('Do not overwrite a frozen evaluated plan')
    write_json(path,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        purpose='Test correlated common errors directly and improve the robustness/natural-performance tradeoff',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_calibration_queries_per_condition_maximum=96,features=FEATURES,
        covariance='Weighted source-calibration second moments of Base and five alternative errors; full cross terms retained',
        gate='alpha_new=alpha_original*((1-gamma)+gamma*clip(head(features),0,1))*min(1,(12/d)^2)',
        radius_source_sd=RADIUS,augmentation_weights=LAMBDAS,gain_mixture_candidates=GAMMAS,
        augmentation='One label-free shared +/-bias draw per source query, fixed seed; alternatives only, Base and labels unchanged',
        calibration_bias_magnitudes=CAL_MAGNITUDES,
        selection='Inner source datasets: minimize average common-bias validation normalized MSE subject to natural normalized MSE <= original +0.001; fallback to original if no feasible candidate',
        no_augmentation_control='Fit lambda=0 and choose gamma using natural source validation loss only',
        learner=dict(type='HistGradientBoostingRegressor',max_iter=100,max_leaf_nodes=15,max_depth=4,
            min_samples_leaf=150,l2_regularization=5.,learning_rate=.07,early_stopping=False,random_state=0),
        evaluation_bias_magnitudes=[.25,.5,1.,2.,3.,5.,8.,10.],
        stress_dependence_patterns=['independent_random_signs','shared_bias_3of5','shared_bias_5of5'],
        matched_per_affected_branch_bias_magnitude=True,stress_sign_selection_uses_labels=False,
        held_out_labels_used_for_parameter_fitting=False,contexts_and_backbones_unchanged=True,
        original_fusion_parameters_frozen=True,backbone_calls=0,markdown_written=False,
        evaluation_status='Development evaluation on previously inspected reconstructed banks; not an untouched confirmatory test'))

def load(backbone,track):
    dataset=[]
    rng=np.random.default_rng(0)
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        idx=np.sort(rng.choice(len(y),min(96,len(y)),replace=False))
        dataset.append(dict(id=int(m['data_id']),meta=m,b=(b-sm)/ss,k=((k-sm)/ss).T,y=(y-sm)/ss,
            sm=sm,ss=ss,fold=fold,ref=ref,idx=idx))
    return dataset

def pack(dataset,track):
    counts=defaultdict(float)
    for c in dataset:counts[c['id']]+=(5. if track=='five' else 7.) if c['meta']['rho']==0 else 1.
    parts=[]
    for c in dataset:
        idx=c['idx'];cw=(5. if track=='five' else 7.) if c['meta']['rho']==0 else 1.
        weight=cw/(len(counts)*counts[c['id']]*len(idx)*max(float(c['y'].var()),1e-12))
        parts.append((c['b'][idx],c['k'][idx],c['y'][idx],np.full(len(idx),weight)))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))

def second_moment(data):
    b,k,y,w=data
    errors=np.c_[b-y,k-y[:,None]]
    moment=errors.T@(errors*w[:,None])/w.sum()
    moment+=np.eye(6)*max(1e-12,float(np.trace(moment))*1e-8)
    assert np.linalg.eigvalsh(moment).min()>0
    return moment

def state(b,k,theta,moment):
    median=np.median(k,axis=1);residual=np.abs(k-median[:,None]);tau=np.exp(theta[-2])
    agreement=np.exp(-residual/tau)
    weights=softmax(theta[:5]+agreement*theta[5:10],1)
    ag=(weights*k).sum(1);support=(weights*agreement).sum(1)
    difference=ag-b;distance=np.abs(difference)
    alpha=theta[-1]*(1-np.exp(-distance/tau))*support
    factor=np.minimum(1.,(RADIUS/np.maximum(distance,1e-12))**2)
    delta=alpha*factor*difference
    vbase=moment[0,0]
    valt=np.einsum('ni,ij,nj->n',weights,moment[1:,1:],weights)
    cross=weights@moment[1:,0]
    optimal=np.clip((vbase-cross)/np.maximum(vbase+valt-2*cross,1e-12),0,1)
    feature=np.c_[alpha,np.log1p(distance),np.tanh(difference/3.),np.clip(b,-30,30),np.clip(ag,-30,30),
        np.log1p(np.abs(b)),np.log1p(np.abs(ag)),support,np.log1p(np.median(residual,axis=1)),
        np.log1p(k.std(1)),np.log1p(residual.max(1)),weights.max(1),(weights**2).sum(1),
        np.log1p(valt/max(vbase,1e-12)),cross/np.sqrt(np.maximum(valt*vbase,1e-24)),optimal]
    assert feature.shape[1]==len(FEATURES) and np.isfinite(feature).all()
    return feature.astype(np.float32),delta,b+alpha*difference

def fit(data,theta,moment,lam):
    b,k,y,w=data
    x,delta,_=state(b,k,theta,moment)
    target=np.clip((y-b)/np.where(np.abs(delta)>1e-12,delta,1.),0,1)
    weight=w*delta**2
    if lam:
        rng=np.random.default_rng(0)
        bias=rng.choice(CAL_MAGNITUDES,len(b))*rng.choice([-1.,1.],len(b))
        xa,da,_=state(b,k+bias[:,None],theta,moment)
        ta=np.clip((y-b)/np.where(np.abs(da)>1e-12,da,1.),0,1)
        x=np.r_[x,xa];target=np.r_[target,ta];weight=np.r_[weight,lam*w*da**2]
    selected=weight>1e-16
    weight=weight[selected];weight=weight/weight.mean()
    model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,max_depth=4,min_samples_leaf=150,
        l2_regularization=5.,learning_rate=.07,early_stopping=False,random_state=0)
    model.fit(x[selected],target[selected],sample_weight=weight)
    return model

def outputs(b,k,theta,head):
    if not head['enabled']:
        return state(b,k,theta,head['moment'])[2]
    x,delta,_=state(b,k,theta,head['moment'])
    r=np.clip(head['model'].predict(x),0,1)
    factor=(1-head['gamma'])+head['gamma']*r
    out=b+delta*factor
    assert np.isfinite(out).all() and np.max(np.abs(out-b))<=RADIUS+1e-7
    return out

def weighted_loss(y,p,w):return float(np.sum(w*(p-y)**2))

def candidate_scores(model,moment,data,theta):
    b,k,y,w=data
    x,delta,old=state(b,k,theta,moment);r=np.clip(model.predict(x),0,1)
    scores={g:dict(natural=weighted_loss(y,b+delta*((1-g)+g*r),w),stress=0.) for g in GAMMAS}
    for magnitude in CAL_MAGNITUDES:
        for sign in [-1,1]:
            x,d,_=state(b,k+sign*magnitude,theta,moment)
            r=np.clip(model.predict(x),0,1)
            for gamma in GAMMAS:
                scores[gamma]['stress']+=weighted_loss(y,b+d*((1-gamma)+gamma*r),w)/(2*len(CAL_MAGNITUDES))
    return scores,weighted_loss(y,old,w)

def policy_records(backbone,track):
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    return next(c for c in plan['cases'] if (c['backbone'],c['track'],c['task'])==(backbone,track,'regression'))['policies']

def train(backbone,track):
    dataset=load(backbone,track)
    for record in policy_records(backbone,track):
        started=time.monotonic();policy=json.loads((ROOT/record['path']).read_text());theta=np.asarray(policy['parameters'])
        train_ids=set(policy['train_data_ids']);test_ids=set(policy['test_data_ids']);assert train_ids.isdisjoint(test_ids)
        outer_train=[c for c in dataset if c['id'] in train_ids]
        inner_fold=min(c['fold'] for c in outer_train)
        inner_train=[c for c in outer_train if c['fold']!=inner_fold]
        inner_val=[c for c in outer_train if c['fold']==inner_fold]
        assert {c['id'] for c in inner_train}.isdisjoint(c['id'] for c in inner_val)
        fitdata=pack(inner_train,track);valdata=pack(inner_val,track);moment=second_moment(fitdata)
        candidates=[];control_gamma=None
        for lam in LAMBDAS:
            model=fit(fitdata,theta,moment,lam);scores,baseline=candidate_scores(model,moment,valdata,theta)
            if lam==0:control_gamma=min(GAMMAS,key=lambda g:scores[g]['natural'])
            for gamma,v in scores.items():
                candidates.append(dict(augmentation_weight=lam,gamma=gamma,**v,baseline_natural=baseline,
                    feasible=v['natural']<=baseline+.001))
            print('RISK_CANDIDATE',backbone,track,policy['fold'],lam,scores,flush=True)
        feasible=[c for c in candidates if c['feasible']]
        choice=min(feasible,key=lambda c:(c['stress'],c['natural'],c['augmentation_weight'],c['gamma'])) if feasible else None
        data=pack(outer_train,track);moment=second_moment(data)
        models={0.:fit(data,theta,moment,0.)}
        if choice and choice['augmentation_weight'] not in models:
            lam=choice['augmentation_weight'];models[lam]=fit(data,theta,moment,lam)
        control=dict(enabled=True,model=models[0.],moment=moment,gamma=control_gamma)
        chosen=dict(enabled=choice is not None,model=models[choice['augmentation_weight']] if choice else None,
            moment=moment,gamma=choice['gamma'] if choice else 0.)
        dest=DEST/backbone/track;dest.mkdir(parents=True,exist_ok=True)
        joblib.dump(dict(theta=theta,control=control,chosen=chosen),dest/f'fold_{policy["fold"]}.joblib')
        write_json(dest/f'fold_{policy["fold"]}.json',dict(backbone=backbone,track=track,outer_fold=policy['fold'],
            source_data_ids=sorted(train_ids),evaluation_data_ids=sorted(test_ids),
            inner_training_ids=sorted({c['id'] for c in inner_train}),inner_validation_ids=sorted({c['id'] for c in inner_val}),
            original_policy_sha256=record['sha256'],selection_candidates=candidates,selected=choice,
            no_augmentation_control_gamma=control_gamma,held_out_labels_used_for_fit=False,
            error_second_moment=moment.tolist(),seconds=time.monotonic()-started,
            note='Inner head selection is source-only; the frozen original fusion policy was previously calibrated on all outer-training datasets'))
        print('RISK_FOLD_COMPLETE',backbone,track,policy['fold'],choice,'seconds',time.monotonic()-started,flush=True)
    print('RISK_TRAIN_COMPLETE',backbone,track,flush=True)

def evaluate(backbone,track):
    dest=DEST/backbone/track
    heads={f:joblib.load(dest/f'fold_{f}.joblib') for f in range(5)}
    rows=[];condition_stress=[];pool=defaultdict(list)
    for i,m,rawb,rawk,y,sm,ss,theta,fold,ref,path in cells(backbone,track,'regression'):
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T;head=heads[fold]
        assert np.array_equal(theta,head['theta'])
        def predict_all(matrix):
            old=regress(b,matrix.T,theta,0.,1.)[0]
            return dict(Original=old,Risk_no_aug=outputs(b,matrix,theta,head['control']),
                Risk_shared_aug=outputs(b,matrix,theta,head['chosen']))
        initial=predict_all(k)
        scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        original=initial['Original']*ss+sm
        assert abs(float(np.sqrt(np.mean((original-y)**2))/scale)-ref['nrmse'])<1e-8*max(1,ref['nrmse'])
        for method,p in initial.items():
            pred=p*ss+sm
            rows.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                outer_fold=fold,method=method,n=len(y),r2=1-float(np.sum((pred-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                nrmse=float(np.sqrt(np.mean((pred-y)**2))/scale)))
        # The same fixed random signs/subsets are used for all methods/magnitudes.
        rng=np.random.default_rng(0)
        sign=rng.choice([-1.,1.],len(b))
        independent=rng.choice([-1.,1.],k.shape)
        subset=np.argsort(rng.random(k.shape),axis=1)[:,:3]
        mask=np.zeros_like(k);np.put_along_axis(mask,subset,sign[:,None],axis=1)
        patterns=dict(independent_random_signs=independent,shared_bias_3of5=mask,shared_bias_5of5=sign[:,None]*np.ones_like(k))
        for magnitude in [.25,.5,1.,2.,3.,5.,8.,10.]:
            for pattern,shift in patterns.items():
                pred=predict_all(k+magnitude*shift)
                for method,p in pred.items():
                    d=np.abs(p-initial[method]);pool[(pattern,magnitude,method)].append(d)
                    output=p*ss+sm
                    condition_stress.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                        method=method,pattern=pattern,magnitude_source_sd=magnitude,n=len(y),
                        r2=1-float(np.sum((output-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                        mean_displacement_source_sd=float(d.mean()),p95_displacement_source_sd=float(np.quantile(d,.95))))
        if i%100==0:print('RISK_EVALUATION',backbone,track,i,flush=True)
    pd.DataFrame(rows).to_csv(dest/'performance.csv',index=False)
    pd.DataFrame(condition_stress).to_csv(dest/'stress_by_condition.csv',index=False)
    records=[]
    for (pattern,magnitude,method),parts in pool.items():
        d=np.concatenate(parts)
        records.append(dict(backbone=backbone,track=track,pattern=pattern,magnitude_source_sd=magnitude,method=method,
            n_query_conditions=len(d),mean_displacement_source_sd=float(d.mean()),
            p95_displacement_source_sd=float(np.quantile(d,.95)),max_displacement_source_sd=float(d.max())))
    pd.DataFrame(records).to_csv(dest/'stress_pooled.csv',index=False)
    write_json(dest/'evaluation_audit.json',dict(original_frozen_metrics_reproduced=True,
        stress_bias_and_signs_selected_without_labels=True,stress_does_not_change_Base_or_labels=True,
        query_count_includes_repeated_conditions=True,benchmark_unchanged=True,seed=0))
    print('RISK_EVALUATION_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['train','evaluate'])
    p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    elif a.stage=='train':train(a.backbone,a.track)
    else:evaluate(a.backbone,a.track)
