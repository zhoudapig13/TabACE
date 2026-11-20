"""Explicit follow-up safeguards; fixed choices, no held-out outcome tuning."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,ROOT,cells,classify,regress,MAGNITUDES,write_json


def prepare():
    plan=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_plan_sha256=hashlib.sha256((OUT/'frozen_plan.json').read_bytes()).hexdigest(),
        purpose='Follow up the observed saturated-classification gate and extreme regression branch sensitivity',
        classification_gate='Dual_gate: U_new=max(normalized Base entropy, total_variation(Base, alternative aggregate)); keep all other learned parameters fixed',
        classification_gate_strength=1.,classification_confidence_thresholds=[0.,.5,.8,.9,.95,.99],
        regression_clipping_widths_source_sd=[3.,5.],regression_injection_magnitudes_source_sd=MAGNITUDES,
        label_use='Evaluation counts and losses only; no fitting or outcome-based candidate selection',
        original_method_outputs_retained=True,all_new_safeguards_labelled_as_variants=True,
        no_claim_of_original_method_equivalence=True,backbone_calls=0,markdown_written=False)
    p=OUT/'supplementary_variants_frozen_plan.json'
    if not p.exists():write_json(p,plan)
    print('FOLLOWUP_PLAN_FROZEN',flush=True)


def classification(backbone,track):
    rows=[];performance=[]
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'classification'):
        pred0=b.argmax(1);conf=b.max(1);correct=pred0==y
        original,_=classify(b,k,t)
        assert abs(np.mean(original.argmax(1)==y)-ref['accuracy'])<1e-12
        for method,flags in [('Full',{}),('No_entropy_gate',dict(disable_entropy=True)),
                             ('Dual_gate_variant',dict(disagreement_override=True))]:
            p,alpha=classify(b,k,t,**flags);pred=p.argmax(1);right=pred==y
            performance.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                method=method,n=len(y),accuracy=float(right.mean()),base_accuracy=float(correct.mean()),
                log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y.astype(int)],1e-12)).mean())))
            for threshold in [0.,.5,.8,.9,.95,.99]:
                selected=conf>=threshold;wrong=selected&~correct;good=selected&correct
                rows.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                    method=method,confidence_threshold=threshold,n_queries=len(y),
                    n_high_confidence=int(selected.sum()),base_errors=int(wrong.sum()),corrected_errors=int((wrong&right).sum()),
                    base_correct=int(good.sum()),new_errors=int((good&~right).sum()),
                    changed_predictions=int((selected&(pred0!=pred)).sum()),
                    alpha_sum_high_confidence=float(alpha[selected].sum())))
        if i%300==0:print('FOLLOWUP_CLASSIFICATION',backbone,track,i,flush=True)
    dest=OUT/backbone/track/'classification'
    pd.DataFrame(rows).to_csv(dest/'followup_counts.csv',index=False)
    pd.DataFrame(performance).to_csv(dest/'gate_variant_performance.csv',index=False)
    print('FOLLOWUP_CLASSIFICATION_COMPLETE',backbone,track,flush=True)


def regression(backbone,track):
    metrics=[];stress=[];samples=defaultdict(list);clipping=[]
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        evalscale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        original=regress(b,k,t,sm,ss)[0]
        baselines={width:regress(b,k,t,sm,ss,clip=width)[0] for width in [3.,5.]}
        for width,p in baselines.items():
            method=f'Full_clip{int(width)}sd'
            clipping.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                method=method,clipped_branch_values=int(np.count_nonzero(k!=np.clip(k,sm-width*ss,sm+width*ss))),
                alternative_values=k.size,changed_predictions=int(np.count_nonzero(np.abs(p-original)>1e-10*ss)),n=len(y)))
            metrics.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                method=method,n=len(y),nrmse=float(np.sqrt(np.mean((p-y)**2))/evalscale),
                r2=1-float(np.sum((p-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12)))
        median=np.median(k,axis=0)
        for magnitude in MAGNITUDES:
            worst={width:np.full(len(y),-1.) for width in baselines}
            for branch in range(5):
                for sign in [-1.,1.]:
                    modified=k.copy();modified[branch]=median+sign*magnitude*ss
                    for width in baselines:
                        p=regress(b,modified,t,sm,ss,clip=width)[0]
                        worst[width]=np.maximum(worst[width],np.abs(p-baselines[width])/ss)
            for width,delta in worst.items():
                name=f'Full_clip{int(width)}sd';samples[(magnitude,name)].append(delta)
                stress.append(dict(backbone=backbone,track=track,data_id=m['data_id'],family=m['family'],rho=m['rho'],
                    method=name,magnitude_source_sd=magnitude,n=len(y),mean_displacement_source_sd=float(delta.mean()),
                    p95_displacement_source_sd=float(np.quantile(delta,.95)),max_displacement_source_sd=float(delta.max())))
        if i%100==0:print('FOLLOWUP_REGRESSION',backbone,track,i,flush=True)
    dest=OUT/backbone/track/'regression'
    pd.DataFrame(metrics).to_csv(dest/'clip_width_performance.csv',index=False)
    pd.DataFrame(clipping).to_csv(dest/'clip_width_change_counts.csv',index=False)
    pd.DataFrame(stress).to_csv(dest/'clip_width_stress_by_condition.csv',index=False)
    pooled=[]
    for (magnitude,name),values in samples.items():
        delta=np.concatenate(values)
        pooled.append(dict(backbone=backbone,track=track,method=name,magnitude_source_sd=magnitude,n_query_conditions=len(delta),
            mean_displacement_source_sd=float(delta.mean()),median_displacement_source_sd=float(np.median(delta)),
            p95_displacement_source_sd=float(np.quantile(delta,.95)),p99_displacement_source_sd=float(np.quantile(delta,.99)),
            max_displacement_source_sd=float(delta.max())))
    pd.DataFrame(pooled).to_csv(dest/'clip_width_stress_pooled.csv',index=False)
    print('FOLLOWUP_REGRESSION_COMPLETE',backbone,track,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
    p.add_argument('--stage',choices=['classification','regression']);p.add_argument('--backbone');p.add_argument('--track')
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.stage=='classification':classification(a.backbone,a.track)
    elif a.stage=='regression':regression(a.backbone,a.track)
