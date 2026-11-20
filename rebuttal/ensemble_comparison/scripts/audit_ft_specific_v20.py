"""Audit frozen controls and paired dataset deltas for the Specific development run."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'results'
OUT=ROOT/'ft_specific_context_adaptation_v20'
keys=['task','block','task_id','family','rho']
paths=list(OUT.glob('jobs/*/*/*/query_detail.csv'))
new=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
old=pd.read_csv(ROOT/'finetune_delivery_v17/all_condition_results.csv')
shared=new[new.method=='FT+Shared_frozen'].merge(old[old.method=='FT+Shared_dev'],on=keys,suffixes=('_new','_old'),validate='one_to_one')
clean=new[(new.block=='clean')&(new.method=='FT+Specific_adapted')].merge(old[(old.block=='clean')&(old.method=='FT+Specific_dev')],on=keys,suffixes=('_new','_old'),validate='one_to_one')
errors={}
for name,frame in [('Shared',shared),('Clean_Specific',clean)]:
    for field in ['accuracy','auc','r2','nrmse']:
        a,b=frame[field+'_new'],frame[field+'_old'];mask=a.notna()&b.notna()
        if mask.any():errors[name+'_'+field]=float(np.max(np.abs(a[mask]-b[mask])))
assert all(v<1e-12 for v in errors.values()),errors
rows=[];rng=np.random.default_rng(20261112)
for (task,block),g in new.groupby(['task','block']):
    fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
    for field in fields:
        d=g.groupby(['task_id','method'])[field].mean().unstack()
        delta=(d['FT+Specific_adapted']-d['FT+Shared_frozen']).to_numpy()
        boot=np.mean(rng.choice(delta,(10000,len(delta)),replace=True),axis=1)
        rows.append(dict(task=task,block=block,metric=field,datasets=len(delta),
            Shared=float(d['FT+Shared_frozen'].mean()),Specific=float(d['FT+Specific_adapted'].mean()),
            Specific_minus_Shared=float(delta.mean()),paired_dataset_bootstrap_low=float(np.quantile(boot,.025)),
            paired_dataset_bootstrap_high=float(np.quantile(boot,.975)),
            improved_datasets=int(np.sum(delta<0) if field=='nrmse' else np.sum(delta>0)),
            uncertainty_scope='dataset resampling within single seed; not seed variance'))
pd.DataFrame(rows).to_csv(OUT/'paired_dataset_gaps.csv',index=False)
audit=dict(status='PASS',completed_jobs=len(paths),expected_jobs=175,frozen_control_max_absolute_errors=errors,
    Shared_conditions_checked=len(shared),Clean_Specific_conditions_checked=len(clean),
    Markdown_bytes=Path('/mnt/public/zhouyibo/ICLR2027/REBUTTAL_EXPERIMENTS.md').stat().st_size,
    inference_budget_changed=True,context_cap_Specific=2048,context_cap_Shared=256,
    result_scope='adapted development experiment; not original method or matched-budget comparison')
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2));print(pd.DataFrame(rows).to_string(index=False))
