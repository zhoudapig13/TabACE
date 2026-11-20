"""Report all prespecified shared-error thresholds and actual fusion harm."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,cells,regress,write_json

DEST=OUT/'joint_harm_frequency'
THRESHOLDS=[0.,.5,1.,2.,3.]

def prepare():
    p=DEST/'frozen_plan.json'
    if p.exists():raise RuntimeError('Do not overwrite diagnostic')
    write_json(p,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        thresholds_evaluation_label_sd=THRESHOLDS,
        joint='All five alternatives err in the same direction, each beyond the stated threshold',
        material_harm='Original squared loss exceeds Base squared loss by >0.01 evaluation target variance',
        actual_gate='Also report material-harm queries with original alpha >=0.1',
        labels_used_for_descriptive_statistics_only=True,all_thresholds_reported=True,
        no_comparison_to_unmatched_random_contexts=True,markdown_written=False))

def run(backbone,track):
    rows=[]
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        if m['rho']==0:continue
        p,w,alpha=regress(b,k,t,sm,ss)
        scale=max(float(y.std()),1e-12)
        error=(k-y[None])/scale
        base_se=((b-y)/scale)**2;original_se=((p-y)/scale)**2
        excess=np.maximum(original_se-base_se,0.)
        for threshold in THRESHOLDS:
            joint=(error>threshold).all(0)|(error<-threshold).all(0)
            harmful=joint&(original_se>base_se+.01)
            active=harmful&(alpha>=.1)
            rows.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                threshold_evaluation_sd=threshold,n_queries=len(y),n_joint=int(joint.sum()),
                n_joint_material_harm=int(harmful.sum()),n_joint_active_material_harm=int(active.sum()),
                original_squared_error_sum=float(original_se.sum()),joint_squared_error_sum=float(original_se[joint].sum()),
                excess_loss_sum=float(excess.sum()),joint_excess_loss_sum=float(excess[joint].sum()),
                joint_active_excess_loss_sum=float(excess[active].sum())))
    dest=DEST/backbone/track;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'condition_counts.csv',index=False)
    pooled=[]
    for threshold,g in pd.DataFrame(rows).groupby('threshold_evaluation_sd'):
        s=g.select_dtypes('number').sum();n=int(s.n_queries)
        pooled.append(dict(backbone=backbone,track=track,threshold_evaluation_sd=threshold,n_queries=n,
            joint_percent=100*s.n_joint/n,joint_material_harm_percent=100*s.n_joint_material_harm/n,
            joint_active_material_harm_percent=100*s.n_joint_active_material_harm/n,
            joint_error_mass_percent=100*s.joint_squared_error_sum/s.original_squared_error_sum,
            joint_share_of_excess_fusion_loss_percent=100*s.joint_excess_loss_sum/max(s.excess_loss_sum,1e-12)))
    pd.DataFrame(pooled).to_csv(dest/'summary.csv',index=False)
    print('JOINT_HARM_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.backbone,a.track)
