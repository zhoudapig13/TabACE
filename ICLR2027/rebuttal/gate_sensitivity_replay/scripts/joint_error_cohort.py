"""Label-informed diagnostic cohort: all affected alternatives jointly wrong.

Evaluation labels define the diagnostic cohort and score errors, never select
predictions, fit parameters, choose cap settings, or choose perturbation signs.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,cells,write_json
from smooth_update_guard import predict

DEST=OUT/'joint_error_cohort'

def prepare():
    path=DEST/'frozen_plan.json'
    if path.exists():raise RuntimeError('Do not overwrite evaluated diagnostic')
    write_json(path,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cohort='All affected alternatives have the same signed error, each exceeding 1 source-label SD',
        affected_branches=[3,5],magnitudes_source_sd=[0.,1.,3.,5.,10.],nonclean_conditions_only=True,
        metrics=['Conditional source-standardized MSE','Fraction improved/worsened relative to Original','Cohort frequency'],
        labels_used_for_diagnostic_cohort_and_scoring_only=True,labels_not_used_to_select_directions=True,
        model_and_cap_settings_frozen=True,markdown_written=False))

def run(backbone,track):
    heads={f:json.loads((OUT/'algorithm_repairs_v3'/backbone/track/f'fold_{f}.json').read_text()) for f in range(5)}
    rows=[]
    for i,m,rawb,rawk,rawy,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        if m['rho']==0:continue
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T;y=(rawy-sm)/ss;beta=np.asarray(heads[fold]['parameters'])
        rng=np.random.default_rng(0);sign=rng.choice([-1.,1.],len(b));rng.choice([-1.,1.],k.shape)
        subset=np.argsort(rng.random(k.shape),axis=1)[:,:3]
        three=np.zeros_like(k);np.put_along_axis(three,subset,sign[:,None],axis=1)
        for affected,pattern in [(3,three),(5,sign[:,None]*np.ones_like(k))]:
            selected=np.abs(pattern)>0
            for magnitude in [0.,1.,3.,5.,10.]:
                changed=k+magnitude*pattern
                error=changed-y[:,None]
                mask=((selected&(error>1.)).sum(1)==affected)|((selected&(error<-1.)).sum(1)==affected)
                predictions=predict(b,changed,t,beta);predictions['Base']=b
                old_error=(predictions['Original']-y)**2
                for method,p in predictions.items():
                    se=(p-y)**2
                    rows.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                        affected_branches=affected,magnitude_source_sd=magnitude,method=method,n_queries=len(y),n_cohort=int(mask.sum()),
                        squared_error_sum_source_sd=float(se[mask].sum()),
                        improved_vs_original=int((mask&(se<old_error-1e-10)).sum()),
                        worsened_vs_original=int((mask&(se>old_error+1e-10)).sum())))
    dest=DEST/backbone/track;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'condition_records.csv',index=False)
    df=pd.DataFrame(rows)
    pooled=[]
    for (affected,magnitude,method),g in df.groupby(['affected_branches','magnitude_source_sd','method']):
        n=int(g.n_cohort.sum());total=int(g.n_queries.sum())
        pooled.append(dict(backbone=backbone,track=track,affected_branches=affected,magnitude_source_sd=magnitude,method=method,
            n_queries=total,n_cohort=n,cohort_percent=100*n/total,
            conditional_mse_source_sd=float(g.squared_error_sum_source_sd.sum()/max(n,1)),
            improved_percent=100*g.improved_vs_original.sum()/max(n,1),worsened_percent=100*g.worsened_vs_original.sum()/max(n,1)))
    pd.DataFrame(pooled).to_csv(dest/'pooled_metrics.csv',index=False)
    write_json(dest/'audit.json',dict(cohort_common_for_all_methods=True,
        evaluation_labels_not_used_for_prediction_selection=True,parameters_and_stress_signs_frozen=True))
    print('JOINT_COHORT_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.backbone,a.track)
