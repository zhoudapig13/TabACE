"""Quantify natural shared-error loss budgets on fixed, original-defined cohorts."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,cells,write_json
import gate_preserving_residual as implementation

DEST=OUT/'gated_error_budget'

def prepare():
    p=DEST/'frozen_plan.json'
    if p.exists():raise RuntimeError('Do not overwrite diagnostic')
    write_json(p,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        thresholds_evaluation_sd=[1.,3.],joint='All five original alternatives err in the same direction beyond threshold',
        harmful='Joint event AND Original loss exceeds Base loss by >0.01 target variance',
        fixed_cohorts=['all','joint1','joint3','harmful_joint1','harmful_joint3'],
        total_error_recovery='(sum original normalized SE - sum new normalized SE) / sum original normalized SE',
        excess_loss_recovery='Reduction of positive excess loss relative to Base, on the same fixed cohort',
        count_recovery='Original harmful queries no longer materially worse than Base',
        labels_used_for_diagnostics_only=True,model_parameters_frozen=True,markdown_written=False))

def run(backbone,track):
    folder=OUT/'gate_preserving_v8'/backbone/track
    heads={f:json.loads((folder/f'fold_{f}.json').read_text()) for f in range(5)}
    records=[];incidence=[]
    for i,m,rawb,rawk,rawy,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        if m['rho']==0:continue
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T;y=(rawy-sm)/ss;sd=max(float(y.std()),1e-12)
        h=heads[fold];beta=np.asarray(h['evidence_parameters'])
        coef=np.asarray(h['residual_coefficients']) if h['residual_coefficients'] is not None else None
        predictions=dict(Original=implementation.core.original(b,k,t),
            Gate_preserving=implementation.core.predict(b,k,t,beta,coef,h['blend']))
        base=((b-y)/sd)**2;old=((predictions['Original']-y)/sd)**2;errors=(k-y[:,None])/sd
        cohorts={'all':np.ones(len(y),bool)}
        for threshold in [1,3]:
            joint=(errors>threshold).all(1)|(errors<-threshold).all(1)
            harmful=joint&(old>base+.01)
            cohorts[f'joint{threshold}']=joint;cohorts[f'harmful_joint{threshold}']=harmful
        for method,p in predictions.items():
            se=((p-y)/sd)**2
            for threshold in [1,3]:
                joint=cohorts[f'joint{threshold}'];harm=joint&(se>base+.01)
                incidence.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                    method=method,threshold_evaluation_sd=threshold,n=len(y),n_joint=int(joint.sum()),n_harm=int(harm.sum())))
            for name,mask in cohorts.items():
                records.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                    method=method,cohort=name,n_total=len(y),n_cohort=int(mask.sum()),
                    sum_normalized_loss=float(se[mask].sum()),sum_excess_vs_Base=float(np.maximum(se-base,0.)[mask].sum()),
                    no_longer_materially_worse_than_Base=int((mask&(se<=base+.01)).sum()),
                    improved_vs_original=int((mask&(se<old-1e-10)).sum()),worsened_vs_original=int((mask&(se>old+1e-10)).sum())))
    dest=DEST/backbone/track;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(dest/'condition_budgets.csv',index=False)
    pd.DataFrame(incidence).to_csv(dest/'condition_incidence.csv',index=False)
    print('GATED_BUDGET_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.backbone,a.track)
