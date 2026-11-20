"""Source-only residual calibration with a common-mode sensitivity penalty."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,ROOT,cells,write_json
from calibrated_regression_repairs import components
from correlated_risk_calibration import load,pack,policy_records

DEST=OUT/'common_mode_v7'
RIDGES=[.001,.01,.1]
SHARED=[0.,.01,.1,1.]
BLENDS=[.25,.5,1.]
CAP=6.

def prepare():
    p=DEST/'frozen_plan.json'
    if p.exists():raise RuntimeError('Do not overwrite evaluated plan')
    hashes=[]
    for b in ['tabicl','tabpfn']:
        for t in ['five','seven_reimplemented']:
            for fold in range(5):
                path=OUT/'algorithm_repairs_v3'/b/t/f'fold_{fold}.json'
                hashes.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(p,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_gate_hashes=hashes,
        purpose='Improve natural errors and shared-error recovery via calibration rather than relying solely on update saturation',
        residual_features=['intercept','standardized_Base','alternative0-minus-Base','alternative1-minus-Base',
            'alternative2-minus-Base','alternative3-minus-Base','alternative4-minus-Base'],
        residual_objective='Weighted squared residual + ridge + lambda_common*(mean_common_sensitivity + sum(alternative residual coefficients))^2',
        mean_common_sensitivity='Label-free finite difference of the frozen evidence update under +/-0.05 source SD common shifts',
        ridges=RIDGES,shared_penalties=SHARED,blends=BLENDS,cap_source_sd=CAP,
        selection='Source inner datasets: require nonclean normalized MSE <= Original+0.001 and shared-bias validation MSE <= Original; minimize nonclean loss +0.02 stress loss +0.2 max(clean_loss-original_clean_loss,0)',
        fallback='Frozen evidence head with smooth cap6',calibration_stress_magnitudes=[.5,1.,2.,5.],
        head_training_and_selection_use_outer_training_datasets_only=True,held_out_labels_used_for_fit=False,
        original_predictions_contexts_benchmarks_unchanged=True,backbone_calls=0,markdown_written=False,
        evaluation_status='Reconstructed development evaluation; all attempted settings retained, not a confirmatory independent test'))

def state(b,k,t,beta):
    ag,a,f=components(b,k,t)
    delta=np.clip(a*(1+.5*np.tanh(f@beta)),0,1)*(ag-b)
    x=np.c_[np.ones(len(b)),b,k-b[:,None]]
    return delta,x

def fit(data,t,beta,ridge,shared):
    b,k,y,w=data;d,x=state(b,k,t,beta)
    rms=np.sqrt(np.sum(w[:,None]*x*x,axis=0)/w.sum()).clip(min=1e-6);rms[0]=1.
    z=x/rms;target=y-b-d
    a=z.T@(w[:,None]*z)+ridge*np.eye(z.shape[1])
    rhs=z.T@(w*target)
    dp=state(b,k+.05,t,beta)[0];dm=state(b,k-.05,t,beta)[0]
    sensitivity=float(np.average((dp-dm)/.1,weights=w))
    direction=np.r_[0.,0.,np.ones(5)]/rms
    a+=shared*np.outer(direction,direction);rhs-=shared*sensitivity*direction
    coef=np.linalg.solve(a,rhs)/rms
    assert np.isfinite(coef).all()
    return coef

def predict(b,k,t,beta,coef=None,blend=0.):
    d,x=state(b,k,t,beta)
    if coef is not None:d=d+blend*(x@coef)
    result=b+CAP*np.tanh(d/CAP)
    assert np.isfinite(result).all() and np.max(np.abs(result-b))<=CAP+1e-7
    return result

def original(b,k,t):
    ag,a,f=components(b,k,t);return b+a*(ag-b)

def loss(data,t,beta,coef,blend):
    b,k,y,w=data;p=predict(b,k,t,beta,coef,blend)
    return float(np.sum(w*(p-y)**2))

def oldloss(data,t):
    b,k,y,w=data;return float(np.sum(w*(original(b,k,t)-y)**2))

def stressloss(data,t,beta,coef,blend,old=False):
    b,k,y,w=data;score=0.
    for magnitude in [.5,1.,2.,5.]:
        for sign in [-1.,1.]:
            p=original(b,k+sign*magnitude,t) if old else predict(b,k+sign*magnitude,t,beta,coef,blend)
            score+=float(np.sum(w*(p-y)**2))/8
    return score

def train(backbone,track):
    dataset=load(backbone,track)
    for record in policy_records(backbone,track):
        parent=json.loads((ROOT/record['path']).read_text());t=np.asarray(parent['parameters'])
        train_ids=set(parent['train_data_ids']);test_ids=set(parent['test_data_ids']);assert train_ids.isdisjoint(test_ids)
        prior=json.loads((OUT/'algorithm_repairs_v3'/backbone/track/f'fold_{parent["fold"]}.json').read_text())
        beta=np.asarray(prior['parameters']);outer=[c for c in dataset if c['id'] in train_ids]
        inner=min(c['fold'] for c in outer);innertrain=[c for c in outer if c['fold']!=inner];val=[c for c in outer if c['fold']==inner]
        fitdata=pack(innertrain,track);natural=pack([c for c in val if c['meta']['rho']!=0],track)
        clean=pack([c for c in val if c['meta']['rho']==0],track)
        baseline=oldloss(natural,t);cleanbase=oldloss(clean,t);stressbase=stressloss(natural,t,beta,None,0,old=True)
        candidates=[]
        for ridge in RIDGES:
            for shared in SHARED:
                coef=fit(fitdata,t,beta,ridge,shared)
                for blend in BLENDS:
                    normal=loss(natural,t,beta,coef,blend);cl=loss(clean,t,beta,coef,blend)
                    st=stressloss(natural,t,beta,coef,blend)
                    candidates.append(dict(ridge=ridge,shared_penalty=shared,blend=blend,natural=normal,clean=cl,stress=st,
                        baseline_natural=baseline,baseline_clean=cleanbase,baseline_stress=stressbase,
                        feasible=normal<=baseline+.001 and st<=stressbase,
                        score=normal+.02*st+.2*max(cl-cleanbase,0.)))
        good=[c for c in candidates if c['feasible']];choice=min(good,key=lambda c:c['score']) if good else None
        coef=fit(pack(outer,track),t,beta,choice['ridge'],choice['shared_penalty']) if choice else None
        write_json(DEST/backbone/track/f'fold_{parent["fold"]}.json',dict(backbone=backbone,track=track,outer_fold=parent['fold'],
            source_data_ids=sorted(train_ids),evaluation_data_ids=sorted(test_ids),
            evidence_parameters=beta.tolist(),residual_coefficients=coef.tolist() if coef is not None else None,
            blend=choice['blend'] if choice else 0.,selected=choice,source_candidates=candidates,
            held_out_labels_used_for_fit=False,original_policy_sha256=record['sha256']))
        print('COMMON_MODE_FIT',backbone,track,parent['fold'],choice,flush=True)
    print('COMMON_MODE_TRAIN_COMPLETE',backbone,track,flush=True)

def evaluate(backbone,track):
    dest=DEST/backbone/track;heads={f:json.loads((dest/f'fold_{f}.json').read_text()) for f in range(5)}
    performance=[];cohorts=[]
    for i,m,rawb,rawk,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T;head=heads[fold];beta=np.asarray(head['evidence_parameters'])
        coef=np.asarray(head['residual_coefficients']) if head['residual_coefficients'] is not None else None
        def predictions(matrix):
            return dict(Original=original(b,matrix,t),Smooth6=predict(b,matrix,t,beta),
                Common_mode_calibrated=predict(b,matrix,t,beta,coef,head['blend']))
        p=predictions(k);scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        for method,v in p.items():
            output=v*ss+sm
            performance.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                method=method,n=len(y),r2=1-float(np.sum((output-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                nrmse=float(np.sqrt(np.mean((output-y)**2))/scale)))
        old=p['Original']*ss+sm
        assert abs(float(np.sqrt(np.mean((old-y)**2))/scale)-ref['nrmse'])<1e-8*max(1,ref['nrmse'])
        if m['rho']==0:continue
        yn=(y-sm)/ss;targetscale=max(float(yn.std()),1e-12)
        rng=np.random.default_rng(0);sign=rng.choice([-1.,1.],len(b))
        for magnitude in [0.,1.,3.,5.,10.]:
            shifted=k+sign[:,None]*magnitude
            pred=predictions(shifted);err=(shifted-yn[:,None])/targetscale
            base_se=((b-yn)/targetscale)**2;old_se=((pred['Original']-yn)/targetscale)**2
            for threshold in [0.,1.,3.]:
                joint=(err>threshold).all(1)|(err<-threshold).all(1)
                old_harm=np.maximum(old_se-base_se,0.)
                for method,v in pred.items():
                    se=((v-yn)/targetscale)**2
                    cohorts.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                        method=method,magnitude_source_sd=magnitude,threshold_evaluation_sd=threshold,n=len(y),n_joint=int(joint.sum()),
                        sum_normalized_se=float(se.sum()),joint_sum_normalized_se=float(se[joint].sum()),
                        joint_old_excess_loss=float(old_harm[joint].sum()),
                        joint_new_excess_loss=float(np.maximum(se-base_se,0.)[joint].sum()),
                        improved_queries=int((joint&(se<old_se-1e-10)).sum()),worsened_queries=int((joint&(se>old_se+1e-10)).sum())))
        if i%200==0:print('COMMON_MODE_EVAL',backbone,track,i,flush=True)
    pd.DataFrame(performance).to_csv(dest/'performance.csv',index=False)
    pd.DataFrame(cohorts).to_csv(dest/'joint_error_records.csv',index=False)
    write_json(dest/'evaluation_audit.json',dict(original_frozen_metrics_reproduced=True,source_test_disjoint=True,
        labels_used_for_scoring_and_cohort_statistics_only=True,identical_stress_directions_for_methods=True))
    print('COMMON_MODE_EVALUATION_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['train','evaluate'])
    p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    elif a.stage=='train':train(a.backbone,a.track)
    else:evaluate(a.backbone,a.track)
