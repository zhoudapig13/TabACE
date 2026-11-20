"""Measure naturally occurring failures; labels are used for diagnostics only."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,cells,regress,write_json
from algorithm_repairs import guarded
from correlated_repair import trust

DEST=OUT/'natural_failure_audit'

def run(backbone,track):
    records=[];scores=[]
    for i,m,b,k,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        original,w,a=regress(b,k,t,sm,ss)
        clipped,bounded=guarded(b,k,t,sm,ss)
        trusted,multiplier=trust(b,k,w,a,ss)
        center=np.median(k,axis=0)
        distance=np.abs(k-center[None])/ss
        # Compare residuals with the clipping bound, NOT exact floating-point
        # equality after reconstructing median + residual.
        exceeds=distance>3.
        error=(k-y[None])/ss
        row=dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
            n_queries=len(y),n_branch_values=k.size,clipped_branch_values=int(exceeds.sum()),
            any_extreme_branch_queries=int(exceeds.any(0).sum()),single_extreme_branch_queries=int((exceeds.sum(0)==1).sum()),
            changed_after_clipping=int((np.abs(original-clipped)>1e-10*ss).sum()),
            trust_gate_activated=int((multiplier<1).sum()),
            clipping_absolute_displacement_sum_source_sd=float((np.abs(original-clipped)/ss).sum()),
            trust_absolute_displacement_sum_source_sd=float((np.abs(original-trusted)/ss).sum()))
        for threshold in [1.,2.,3.]:
            positive=(error>threshold).sum(0);negative=(error<-threshold).sum(0)
            for count in [3,5]:
                mask=(positive>=count)|(negative>=count)
                tag=f'common_wrong_{count}of5_{int(threshold)}sd'
                row[tag]=int(mask.sum())
                row[tag+'_original_squared_error_source_sd']=float((((original-y)/ss)**2)[mask].sum())
        records.append(row)
        scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        for method,p in [('Original',original),('Clip3_no_refit',clipped),('Trust3_no_refit',trusted)]:
            scores.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                method=method,n=len(y),r2=1-float(np.sum((p-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                nrmse=float(np.sqrt(np.mean((p-y)**2))/scale)))
    dest=DEST/backbone/track;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(dest/'condition_counts.csv',index=False)
    pd.DataFrame(scores).to_csv(dest/'performance.csv',index=False)
    write_json(dest/'audit.json',dict(labels_used_only_for_diagnostic_error_frequency=True,
        labels_not_used_for_clipping_or_trust_gate=True,definition_common_wrong='At least 3 or all 5 alternative errors exceed the same signed threshold in source-label SD units',
        definition_single_extreme='Exactly one alternative lies outside query-median +/-3 source-label SD',
        clipping_trigger_comparison='absolute residual >3 source SD; floating-point roundoff excluded',
        no_refitting=True,benchmark_unchanged=True))
    print('NATURAL_FAILURE_AUDIT_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    run(a.backbone,a.track)
