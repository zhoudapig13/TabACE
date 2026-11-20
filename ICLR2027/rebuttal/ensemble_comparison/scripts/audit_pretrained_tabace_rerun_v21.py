"""Audit completeness, paired inputs, frozen FT rows, and fresh Base agreement."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'results'
OUT=ROOT/'pretrained_tabace_rerun_v21'
keys=['task','block','task_id','family','rho']
fresh=pd.read_csv(OUT/'query_detail.csv')
previous=pd.read_csv(ROOT/'finetune_delivery_v17/all_condition_results.csv')
paired=fresh[fresh.method=='Base'].merge(previous[previous.method=='Base'],on=keys,suffixes=('_new','_old'),validate='one_to_one')
assert len(paired)==4151,len(paired)
differences={}
for field in ['accuracy','auc','r2','nrmse']:
    a,b=paired[field+'_new'],paired[field+'_old'];mask=a.notna()&b.notna()
    differences[field]=float(np.max(np.abs(a[mask]-b[mask])))
    assert differences[field]<1e-5,(field,differences[field])
records=[]
for path in OUT.glob('jobs/*/*/*/query/*/*.json'):records.append(json.loads(path.read_text()))
assert len(records)==4151,len(records)
assert all(r['pretrained_weights_only'] and not r['FT_delta_loaded'] and r['paired_FT_input_hashes_match'] for r in records)
assert max(max(r['member_rows'].values()) for r in records)<=256
assert len(list(OUT.glob('jobs/*/*/*/bank_completion.json')))==175
assert len(fresh)==4151*3
comparison=pd.read_csv(OUT/'replacement_comparison_long.csv')
assert len(comparison)==36
oldft=pd.read_csv(ROOT/'finetune_delivery_v17/averages.csv');oldft=oldft[oldft.method=='FT']
newft=comparison[comparison.method=='FT']
for _,row in oldft.iterrows():
    new=newft[(newft.task==row.task)&(newft.block==row.block)].iloc[0]
    for f in ['accuracy','auc','r2','nrmse']:
        if pd.notna(row[f]):assert abs(new[f]-row[f])<1e-12
oldcombined=pd.read_csv(ROOT/'ft_specific_context_adaptation_v20/averages.csv')
for _,row in oldcombined.iterrows():
    method={'FT+Shared_frozen':'FT+Shared','FT+Specific_adapted':'FT+Specific'}[row.method]
    new=comparison[(comparison.task==row.task)&(comparison.block==row.block)&(comparison.method==method)].iloc[0]
    for f in ['accuracy','auc','r2','nrmse']:
        if pd.notna(row[f]):assert abs(new[f]-row[f])<1e-12
result=dict(status='PASS',new_pretrained_methods=['Base','Shared','Specific'],task_track_jobs=175,
    paired_final_conditions=4151,prediction_metric_rows=len(fresh),fresh_Base_max_absolute_metric_differences=differences,
    all_final_inputs_paired_to_FT=True,FT_series_metrics_unchanged=True,pretrained_max_context=256,
    FT_Specific_max_context=2048,seed=0,
    Markdown_bytes=Path('/mnt/public/zhouyibo/ICLR2027/REBUTTAL_EXPERIMENTS.md').stat().st_size)
(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
