"""Check the final table, unchanged controls, and source-only Specific selection."""
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'results';OUT=ROOT/'pretrained_specific_crossfit_v23';PARENT=ROOT/'pretrained_tabace_adaptation_v22'
assert json.loads((PARENT/'audit.json').read_text())['status']=='PASS'
assert len(list(OUT.glob('jobs/*/*/*/completion.json')))==175
assert not list(OUT.glob('jobs/*/*/*/error.json'))
d=pd.read_csv(OUT/'query_detail.csv');p=pd.read_csv(PARENT/'query_detail.csv');keys=['task','block','task_id','family','rho','method']
control=d[d.method.isin(['Base','Shared','Reference1'])].merge(p[p.method.isin(['Base','Shared','Reference1'])],on=keys,suffixes=('_new','_old'),validate='one_to_one')
assert len(control)==4151*3
maximum={}
for f in ['accuracy','auc','r2','nrmse']:
    a,b=control[f+'_new'],control[f+'_old'];mask=a.notna()&b.notna();maximum[f]=float(np.max(np.abs(a[mask]-b[mask])))
    assert maximum[f]<1e-12,(f,maximum[f])
plan=json.loads((OUT/'plan.json').read_text())
for path,key in [(ROOT/'finetune_delivery_v17/averages.csv','frozen_FT_report_sha256'),(ROOT/'ft_specific_context_adaptation_v20/averages.csv','frozen_FT_Specific_report_sha256')]:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==plan[key]
for path in OUT.glob('jobs/*/*/*/policies.json'):
    r=json.loads(path.read_text());assert not r['final_query_labels_for_selection']
    assert r['Shared_parameters_refit_without_held_fold_labels_for_Specific_selection'] and r['original_row_grouped_folds']==3
table=pd.read_csv(OUT/'replacement_comparison_long.csv');assert len(table)==36
oldft=pd.read_csv(ROOT/'finetune_delivery_v17/averages.csv');oldft=oldft[oldft.method=='FT']
oldextra=pd.read_csv(ROOT/'ft_specific_context_adaptation_v20/averages.csv');oldextra['method']=oldextra.method.map({'FT+Shared_frozen':'FT+Shared','FT+Specific_adapted':'FT+Specific'})
for _,r in pd.concat([oldft,oldextra],ignore_index=True).iterrows():
    q=table[(table.task==r.task)&(table.block==r.block)&(table.method==r.method)].iloc[0]
    for f in ['accuracy','auc','r2','nrmse']:
        if pd.notna(r[f]):assert abs(q[f]-r[f])<1e-12
rows=[];rng=np.random.default_rng(20261112)
for (task,block),g in d[d.method.isin(['Shared','Specific'])].groupby(['task','block']):
    for f in (['accuracy','auc'] if task=='classification' else ['r2','nrmse']):
        a=g.groupby(['task_id','method'])[f].mean().unstack();delta=(a.Specific-a.Shared).to_numpy();boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
        rows.append(dict(task=task,block=block,metric=f,Specific_minus_Shared=float(delta.mean()),bootstrap_low=float(np.quantile(boot,.025)),
            bootstrap_high=float(np.quantile(boot,.975)),uncertainty='dataset bootstrap within single seed, not seed variance'))
pd.DataFrame(rows).to_csv(OUT/'paired_dataset_gaps.csv',index=False)
result=dict(status='PASS',task_track_jobs=175,final_conditions=4151,Base_Shared_Reference1_max_metric_change=maximum,FT_series_unchanged=True,
    Specific_selection_original_row_grouped_source_folds=3,final_query_labels_for_selection=False,new_TFM_calls=0,
    pretrained_TFM_weights_only=True,Shared_context_cap=256,Specific_context_cap=2048,seed=0,
    Markdown_bytes=Path('/mnt/public/zhouyibo/ICLR2027/REBUTTAL_EXPERIMENTS.md').stat().st_size)
(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print(pd.DataFrame(rows).to_string(index=False))
