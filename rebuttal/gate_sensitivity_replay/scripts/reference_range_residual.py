"""Calibrate residual corrections only beyond the normal update reference range."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,ROOT,cells,write_json
from calibrated_regression_repairs import components
import common_mode_residual as core

DEST=OUT/'reference_range_v9'
THRESHOLD=.05
_raw_state=core.state

def state(b,k,t,beta):
    evidence,x=_raw_state(b,k,t,beta)
    ag,a,f=components(b,k,t);old=a*(ag-b)
    activity=1-np.exp(-(np.maximum(np.abs(old)-THRESHOLD,0.)/THRESHOLD)**2)
    return old+a*a*(evidence-old),x*activity[:,None]

core.state=state
core.DEST=DEST

def prepare():
    p=DEST/'frozen_plan.json'
    if p.exists():raise RuntimeError('Do not overwrite evaluated plan')
    write_json(p,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_implementation_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        purpose='Preserve normal-update behavior while enabling stronger calibration outside a source-clean reference range',
        reference='95th percentile of absolute Original updates on source-clean predictions, minimum0.05 source SD',
        activity='1-exp(-(max(abs(original_update)-reference,0)/reference)^2)',
        formula='delta=delta_original+alpha_original^2*(delta_evidence-delta_original)+activity*blend*X@beta; final=Base+6*source_SD*tanh(delta/(6*source_SD))',
        reference_uses_labels=False,ridges=core.RIDGES,shared_penalties=core.SHARED,blends=core.BLENDS,
        selection='Source inner datasets: nonclean loss <=Original+0.001, clean loss <=Original+0.005, stress loss <=Original; minimize nonclean loss+0.1 max(clean loss-original clean loss,0)',
        held_out_labels_used_for_fit_or_selection=False,benchmark_unchanged=True,backbone_calls=0,markdown_written=False,
        evaluation_status='Final development attempt in this cycle; all versions retained, not a confirmatory independent test'))

def reference(clean,t):
    b,k,y,w=clean;ag,a,f=components(b,k,t)
    return max(.05,float(np.quantile(np.abs(a*(ag-b)),.95)))

def train(backbone,track):
    global THRESHOLD
    dataset=core.load(backbone,track)
    for record in core.policy_records(backbone,track):
        parent=json.loads((ROOT/record['path']).read_text());t=np.asarray(parent['parameters'])
        train_ids=set(parent['train_data_ids']);test_ids=set(parent['test_data_ids']);assert train_ids.isdisjoint(test_ids)
        prior=json.loads((OUT/'algorithm_repairs_v3'/backbone/track/f'fold_{parent["fold"]}.json').read_text());beta=np.asarray(prior['parameters'])
        outer=[c for c in dataset if c['id'] in train_ids];inner=min(c['fold'] for c in outer)
        innertrain=[c for c in outer if c['fold']!=inner];val=[c for c in outer if c['fold']==inner]
        THRESHOLD=reference(core.pack([c for c in innertrain if c['meta']['rho']==0],track),t)
        inner_reference=THRESHOLD;fitdata=core.pack(innertrain,track)
        natural=core.pack([c for c in val if c['meta']['rho']!=0],track);clean=core.pack([c for c in val if c['meta']['rho']==0],track)
        baseline=core.oldloss(natural,t);cleanbase=core.oldloss(clean,t);stressbase=core.stressloss(natural,t,beta,None,0,old=True)
        candidates=[]
        for ridge in core.RIDGES:
            for shared in core.SHARED:
                coef=core.fit(fitdata,t,beta,ridge,shared)
                for blend in core.BLENDS:
                    normal=core.loss(natural,t,beta,coef,blend);cl=core.loss(clean,t,beta,coef,blend);st=core.stressloss(natural,t,beta,coef,blend)
                    candidates.append(dict(ridge=ridge,shared_penalty=shared,blend=blend,natural=normal,clean=cl,stress=st,
                        baseline_natural=baseline,baseline_clean=cleanbase,baseline_stress=stressbase,
                        feasible=normal<=baseline+.001 and cl<=cleanbase+.005 and st<=stressbase,
                        score=normal+.1*max(cl-cleanbase,0.)))
        good=[c for c in candidates if c['feasible']];choice=min(good,key=lambda c:c['score']) if good else None
        THRESHOLD=reference(core.pack([c for c in outer if c['meta']['rho']==0],track),t)
        coef=core.fit(core.pack(outer,track),t,beta,choice['ridge'],choice['shared_penalty']) if choice else None
        write_json(DEST/backbone/track/f'fold_{parent["fold"]}.json',dict(backbone=backbone,track=track,outer_fold=parent['fold'],
            source_data_ids=sorted(train_ids),evaluation_data_ids=sorted(test_ids),evidence_parameters=beta.tolist(),
            residual_coefficients=coef.tolist() if coef is not None else None,blend=choice['blend'] if choice else 0.,
            update_threshold=THRESHOLD,inner_update_threshold=inner_reference,selected=choice,source_candidates=candidates,
            held_out_labels_used_for_fit=False,original_policy_sha256=record['sha256']))
        print('REFERENCE_FIT',backbone,track,parent['fold'],THRESHOLD,choice,flush=True)
    print('REFERENCE_TRAIN_COMPLETE',backbone,track,flush=True)

def evaluate(backbone,track):
    global THRESHOLD
    dest=DEST/backbone/track;heads={f:json.loads((dest/f'fold_{f}.json').read_text()) for f in range(5)}
    performance=[];cohorts=[];budgets=[]
    for i,m,rawb,rawk,rawy,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T;y=(rawy-sm)/ss;h=heads[fold];THRESHOLD=h['update_threshold']
        beta=np.asarray(h['evidence_parameters']);coef=np.asarray(h['residual_coefficients']) if h['residual_coefficients'] is not None else None
        def predictions(matrix):
            return dict(Original=core.original(b,matrix,t),Reference_calibrated=core.predict(b,matrix,t,beta,coef,h['blend']))
        pred=predictions(k);scale=max(float(rawy.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        for method,v in pred.items():
            output=v*ss+sm
            performance.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],method=method,n=len(y),
                r2=1-float(np.sum((output-rawy)**2))/max(float(np.sum((rawy-rawy.mean())**2)),1e-12),
                nrmse=float(np.sqrt(np.mean((output-rawy)**2))/scale)))
        original=pred['Original']*ss+sm
        assert abs(float(np.sqrt(np.mean((original-rawy)**2))/scale)-ref['nrmse'])<1e-8*max(1,ref['nrmse'])
        if m['rho']==0:continue
        sd=max(float(y.std()),1e-12);base_se=((b-y)/sd)**2;old_se=((pred['Original']-y)/sd)**2;errors=(k-y[:,None])/sd
        masks={'all':np.ones(len(y),bool)}
        for threshold in [1,3]:
            joint=(errors>threshold).all(1)|(errors<-threshold).all(1)
            masks[f'joint{threshold}']=joint;masks[f'harmful_joint{threshold}']=joint&(old_se>base_se+.01)
        for method,v in pred.items():
            se=((v-y)/sd)**2
            for name,mask in masks.items():
                budgets.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],method=method,
                    cohort=name,n_total=len(y),n_cohort=int(mask.sum()),sum_normalized_loss=float(se[mask].sum()),
                    sum_excess_vs_Base=float(np.maximum(se-base_se,0.)[mask].sum()),
                    no_longer_materially_worse_than_Base=int((mask&(se<=base_se+.01)).sum()),
                    improved_vs_original=int((mask&(se<old_se-1e-10)).sum()),worsened_vs_original=int((mask&(se>old_se+1e-10)).sum())))
        rng=np.random.default_rng(0);sign=rng.choice([-1.,1.],len(y))
        for magnitude in [0.,1.,3.,5.,10.]:
            matrix=k+sign[:,None]*magnitude;values=predictions(matrix);err=(matrix-y[:,None])/sd
            for threshold in [1.,3.]:
                mask=(err>threshold).all(1)|(err<-threshold).all(1)
                for method,v in values.items():
                    se=((v-y)/sd)**2
                    cohorts.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],method=method,
                        magnitude_source_sd=magnitude,threshold_evaluation_sd=threshold,n=len(y),n_joint=int(mask.sum()),
                        joint_sum_normalized_se=float(se[mask].sum())))
        if i%200==0:print('REFERENCE_EVAL',backbone,track,i,flush=True)
    pd.DataFrame(performance).to_csv(dest/'performance.csv',index=False)
    pd.DataFrame(budgets).to_csv(dest/'condition_budgets.csv',index=False)
    pd.DataFrame(cohorts).to_csv(dest/'joint_error_records.csv',index=False)
    write_json(dest/'evaluation_audit.json',dict(original_frozen_metrics_reproduced=True,source_test_disjoint=True,
        labels_used_for_scoring_and_cohort_statistics_only=True,update_threshold_from_source_clean_predictions_only=True))
    print('REFERENCE_EVALUATION_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['train','evaluate'])
    p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    elif a.stage=='train':train(a.backbone,a.track)
    else:evaluate(a.backbone,a.track)
