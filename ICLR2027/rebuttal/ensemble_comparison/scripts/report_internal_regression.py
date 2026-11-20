"""JSON/CSV/chat only. Never writes the user's cleared Markdown document."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from prepare_data import atomic_json
from internal_regression_benchmark import OUT,TARGETS
from fit_internal_regression_ensembles import METHODS


def averages(rows):
    clean=[r for r in rows if r['rho']==0];other=[r for r in rows if r['rho']>0]
    return {name:{k:float(np.mean([r[k] for r in values])) for k in ['r2','nrmse']} if values else None
            for name,values in [('clean',clean),('shift',other)]}


def bank_live():
    result={}
    for track,per_dataset in [('five',21),('seven',29)]:
        complete=[];count=0
        for folder in (OUT/'ensemble_banks'/track).glob('*'):
            files=list(folder.glob('*.json'));count+=len(files)
            if len(files)==per_dataset:complete.extend(json.loads(p.read_text()) for p in files)
        methods={}
        if complete:
            for method in complete[0]['methods']:
                rows=[{'rho':r['rho'],**r['methods'][method]} for r in complete]
                methods[method]=averages(rows)
        result[track]={'completed_conditions':count,'expected_conditions':735 if track=='five' else 957,
             'completed_dataset_grids':len(complete)//per_dataset,'expected_datasets':35 if track=='five' else 33,
             'methods_on_completed_datasets_only':methods}
    atomic_json(OUT/'live_bank_summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


def report():
    base={'clean':json.loads((OUT/'frozen_clean.json').read_text())['metrics'],
          'five':json.loads((OUT/'frozen_five.json').read_text())['metrics'],
          'seven':json.loads((OUT/'frozen_seven.json').read_text())['metrics']}
    shared={'five':{'r2':.363,'nrmse':.783},'seven':{'r2':.467,'nrmse':.659}}
    records=[{'method':'Base','clean':base['clean'],'five':base['five'],'seven':base['seven']}];raw_records=[]
    for method in METHODS:
        row={'method':method,'annotation':'†‡','adaptation':'Input/context bank selection and calibration-dataset combiner or parameter selection',
             'backbone_weight_finetuning':False};raw={'method':method};diagnostic={}
        for track,expected in [('five',735),('seven',957)]:
            files=sorted((OUT/'policies'/track).glob(f'fold_*_{method}.json'))
            assert len(files)==5,(track,method,len(files))
            policies=[json.loads(p.read_text()) for p in files]
            values=[r for p in policies for r in p['rows']];raw_values=[r for p in policies for r in p['raw_rows']]
            assert len(values)==expected and len(raw_values)==expected
            assert len({(r['task_id'],r['family'],r['rho']) for r in values})==expected
            a=averages(values);b=averages(raw_values);row[track]=a['shift'];raw[track]=b['shift']
            if track=='five':row['clean']=a['clean'];raw['clean']=b['clean']
            else:diagnostic['seven_clean_separate_33_dataset_cohort']=a['clean']
            row[track+'_at_least_Base']=a['shift']['r2']>=base[track]['r2']-1e-12 and a['shift']['nrmse']<=base[track]['nrmse']+1e-12
            row[track+'_below_historical_Shared']=a['shift']['r2']<=shared[track]['r2'] and a['shift']['nrmse']>=shared[track]['nrmse']
        row['clean_within_005']=row['clean']['r2']>=base['clean']['r2']-.005 and row['clean']['nrmse']<=base['clean']['nrmse']+.005
        row.update(diagnostic);records.append(row);raw_records.append(raw)
    # Show subsequent adaptations as additional rows; keep every first-stage row.
    for method in ['Native6_support_gate','IQR6_calibrated_keep']:
        row={'method':method,'annotation':'†‡','adaptation':'Calibration-dataset support-gate or IQR-retention selection',
             'backbone_weight_finetuning':False}
        available=True
        for track,expected in [('five',735),('seven',957)]:
            paths=sorted((OUT/'refined_policies'/track).glob(f'fold_*_{method}.json'))
            if len(paths)!=5:available=False;break
            policies=[json.loads(p.read_text()) for p in paths]
            values=[r for p in policies for r in p['rows']]
            assert len(values)==expected
            a=averages(values);row[track]=a['shift']
            if track=='five':row['clean']=a['clean']
            row[track+'_at_least_Base']=a['shift']['r2']>=base[track]['r2']-1e-12 and a['shift']['nrmse']<=base[track]['nrmse']+1e-12
            row[track+'_below_historical_Shared']=a['shift']['r2']<=shared[track]['r2'] and a['shift']['nrmse']>=shared[track]['nrmse']
        if available:
            row['clean_within_005']=row['clean']['r2']>=base['clean']['r2']-.005 and row['clean']['nrmse']<=base['clean']['nrmse']+.005
            records.append(row)
    result={'status':'COMPLETE','seed':0,'records':records,'raw_method_records':raw_records,
          'paper_Base_reference':TARGETS,'historical_Shared_reference':shared,
          'historical_Shared_was_not_run_on_this_benchmark':True,'internal_development_only':True,
          'benchmark_calibrated_using_Base_development_labels':True,
          'test_dataset_labels_not_used_for_method_parameter_fitting':True,
          'Shared_bounds_not_used_for_method_selection':True,'classification_accepted_snapshot_unchanged':True,
          'five_datasets':35,'seven_datasets':33,'strengths':[.2,.5,.8,1.],
          'logical_context_count':6,'Random_excluded_from_main_table':True,'Markdown_written':False}
    result['annotations']={'†':'Special input/context adaptation','‡':'Calibration-dataset fitting or parameter selection',
                           '§':'Backbone parameter fine-tuning; none in these runs'}
    result['five_Base_calibration_within_requested_tolerance']=json.loads((OUT/'frozen_five.json').read_text())['within_tolerance']
    result['all_interval_checks_are_diagnostic_only']=True
    atomic_json(OUT/'latest_summary.json',result)
    with (OUT/'averages.csv').open('w',newline='') as h:
        fields=['method','annotation','backbone_weight_finetuning','Clean_R2','Clean_nRMSE','Five_R2','Five_nRMSE','Seven_R2','Seven_nRMSE',
                'five_at_least_Base','five_below_historical_Shared','seven_at_least_Base','seven_below_historical_Shared','clean_within_005']
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader()
        for r in records:
            writer.writerow({'method':r['method'],'annotation':r.get('annotation',''),'backbone_weight_finetuning':False,
              'Clean_R2':r['clean']['r2'],'Clean_nRMSE':r['clean']['nrmse'],
              'Five_R2':r['five']['r2'],'Five_nRMSE':r['five']['nrmse'],'Seven_R2':r['seven']['r2'],'Seven_nRMSE':r['seven']['nrmse'],
              **{k:r.get(k,'') for k in fields if k.endswith(('Base','Shared','005'))}})
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');a=p.parse_args()
    if a.live:bank_live()
    else:report()
