"""Rescore the existing, unchanged joint-error cohort on an R2-compatible scale.

Source-label SD can be near zero after label collapse. Pooled errors divided by
that SD are therefore unstable. Normalize errors by the evaluation target
variance for scoring, as in R2, and average datasets equally. Cohort membership,
predictions, and signs are unchanged, and the original score is retained.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,cells,write_json

DEST=OUT/'common_error_study_summary'

def main():
    records=[];details=[];scale_audit=[]
    for b in ['tabicl','tabpfn']:
        for t in ['five','seven_reimplemented']:
            lookup={}
            for i,m,base,bank,y,sm,ss,theta,fold,ref,path in cells(b,t,'regression'):
                rawvar=float(y.var());source_var=rawvar/(ss*ss)
                lookup[(int(m['data_id']),m['family'],m['rho'])]=max(source_var,1e-12)
                if m['rho']!=0 and ss/max(float(y.std()),1e-12)<1e-6:
                    scale_audit.append(dict(backbone=b,track=t,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                        source_sd=ss,evaluation_sd=float(y.std())))
            p=OUT/'joint_error_cohort'/b/t/'condition_records.csv'
            frame=pd.read_csv(p)
            divisor=np.asarray([lookup[(r.data_id,r.family,r.rho)] for r in frame.itertuples()])
            frame['normalized_squared_error_sum']=frame.squared_error_sum_source_sd/divisor
            frame['conditional_normalized_mse']=frame.normalized_squared_error_sum/frame.n_cohort.clip(lower=1)
            frame['score_status']='Rescored with R2 target-variance normalization; original source-SD score retained'
            details.append(frame)
            for (affected,magnitude),group in frame.groupby(['affected_branches','magnitude_source_sd']):
                summary=[]
                for method,rows in group.groupby('method'):
                    task=rows.groupby('data_id')[['normalized_squared_error_sum','n_cohort']].sum()
                    valid=task.n_cohort>0
                    metric=float((task.loc[valid,'normalized_squared_error_sum']/task.loc[valid,'n_cohort']).mean())
                    summary.append(dict(backbone=b,track=t,affected_branches=affected,magnitude_source_sd=magnitude,
                        method=method,datasets_with_cohort=int(valid.sum()),n_cohort=int(rows.n_cohort.sum()),
                        conditional_normalized_mse=metric,
                        improved_percent=100*rows.improved_vs_original.sum()/max(rows.n_cohort.sum(),1),
                        worsened_percent=100*rows.worsened_vs_original.sum()/max(rows.n_cohort.sum(),1)))
                original=next(r for r in summary if r['method']=='Original')
                for row in summary:
                    row['mse_reduction_percent']=100*(1-row['conditional_normalized_mse']/original['conditional_normalized_mse'])
                    records.append(row)
    pd.DataFrame(records).to_csv(DEST/'joint_error_conditional_metrics.csv',index=False)
    pd.concat(details).to_csv(DEST/'joint_error_rescored_conditions.csv',index=False)
    write_json(DEST/'joint_error_scoring_audit.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        issue='Near-zero source-label SD in label-collapse conditions makes the initial pooled source-SD MSE unsuitable for cross-dataset comparison',
        fix='Normalize each condition by its evaluation target variance, as in R2; equal-dataset mean conditional error',
        changed_predictions=False,changed_cohort=False,changed_stress_signs=False,
        evaluation_labels_used_for_scoring_only=True,all_original_scores_retained=True,
        near_zero_source_scale_conditions=scale_audit))
    d=pd.DataFrame(records)
    print(d[(d.affected_branches==5)&d.magnitude_source_sd.isin([0,1,3,5,10])&(d.method=='Smooth_cap6')].to_string(index=False))

if __name__=='__main__':main()
