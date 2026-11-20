"""Verify the adapted replacement table against frozen Base and FT evidence."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'results';OUT=ROOT/'pretrained_tabace_adaptation_v22'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((OUT/'plan.json').read_text())
assert sha(ROOT/'finetune_delivery_v17/averages.csv')==plan['frozen_FT_report_sha256']
assert sha(ROOT/'ft_specific_context_adaptation_v20/averages.csv')==plan['frozen_FT_Specific_report_sha256']
paths=list(OUT.glob('jobs/*/*/*/completion.json'));assert len(paths)==175,len(paths)
assert not list(OUT.glob('jobs/*/*/*/error.json'))
d=pd.read_csv(OUT/'query_detail.csv');original=pd.read_csv(ROOT/'pretrained_tabace_rerun_v21/query_detail.csv')
keys=['task','block','task_id','family','rho']
paired=d[d.method=='Base'].merge(original[original.method=='Base'],on=keys,suffixes=('_new','_old'),validate='one_to_one')
assert len(paired)==4151
maximum={}
for f in ['accuracy','auc','r2','nrmse']:
    a,b=paired[f+'_new'],paired[f+'_old'];take=a.notna()&b.notna();maximum[f]=float(np.max(np.abs(a[take]-b[take])))
    assert maximum[f]<1e-12,(f,maximum[f])
records=[json.loads(p.read_text()) for p in OUT.glob('jobs/*/*/*/query/*/*.json')]
assert len(records)==4151,len(records)
assert all(r['pretrained_weights_only'] and not r['final_query_labels_used_by_constructors'] for r in records)
assert all(r['reference_query_and_calibration_overlap']==0 for r in records)
assert max(c['context_rows'] for r in records for c in r['reference_cost'])<=256
assert max(c['context_rows'] for r in records for c in r['extra_cost'])<=2048
combined=pd.read_csv(OUT/'replacement_comparison_long.csv');assert len(combined)==36
ref=pd.read_csv(ROOT/'finetune_delivery_v17/averages.csv');ref=ref[ref.method=='FT']
further=pd.read_csv(ROOT/'ft_specific_context_adaptation_v20/averages.csv')
further['method']=further.method.map({'FT+Shared_frozen':'FT+Shared','FT+Specific_adapted':'FT+Specific'})
for _,r in pd.concat([ref,further],ignore_index=True).iterrows():
    q=combined[(combined.task==r.task)&(combined.block==r.block)&(combined.method==r.method)].iloc[0]
    for f in ['accuracy','auc','r2','nrmse']:
        if pd.notna(r[f]):assert abs(r[f]-q[f])<1e-12
rows=[];rng=np.random.default_rng(20261112)
for (task,block),g in d[d.method.isin(['Shared','Specific'])].groupby(['task','block']):
    for f in (['accuracy','auc'] if task=='classification' else ['r2','nrmse']):
        a=g.groupby(['task_id','method'])[f].mean().unstack();delta=(a.Specific-a.Shared).to_numpy()
        boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
        rows.append(dict(task=task,block=block,metric=f,Specific_minus_Shared=float(delta.mean()),
            bootstrap_low=float(np.quantile(boot,.025)),bootstrap_high=float(np.quantile(boot,.975)),
            uncertainty='single-seed dataset bootstrap, not seed variance'))
pd.DataFrame(rows).to_csv(OUT/'paired_dataset_gaps.csv',index=False)
result=dict(status='PASS',jobs=175,final_conditions=4151,Base_max_metric_change=maximum,FT_series_unchanged=True,
    pretrained_TFM_weights_only=True,context_adaptation_and_clean_TRAIN_labels=True,
    Shared_context_cap=256,Specific_context_cap=2048,seed=0,
    final_query_labels_for_policy_fit_or_selection=False,
    Markdown_bytes=Path('/mnt/public/zhouyibo/ICLR2027/REBUTTAL_EXPERIMENTS.md').stat().st_size)
(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
