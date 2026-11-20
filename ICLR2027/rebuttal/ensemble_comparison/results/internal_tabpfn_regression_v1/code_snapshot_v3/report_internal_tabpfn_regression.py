"""Actual means and diagnostic historical-reference checks; no score filtering."""
import csv
import argparse
import json
import numpy as np
from prepare_data import atomic_json
from internal_tabpfn_regression import OUT,TARGETS
from fit_internal_regression_ensembles import METHODS


def averages(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in ['r2','nrmse']}


def live():
    result={}
    for track,size,expected in [('five',21,735),('seven',29,957)]:
        complete=[];count=0
        for folder in (OUT/'ensemble_banks'/track).glob('*'):
            paths=list(folder.glob('*.json'));count+=len(paths)
            if len(paths)==size:complete.extend(json.loads(p.read_text()) for p in paths)
        means={}
        if complete:
            for method in complete[0]['methods']:
                means[method]={block:averages([r['methods'][method] for r in complete if (r['rho']==0)==is_clean])
                    for block,is_clean in [('clean',True),('shift',False)]}
        refs={}
        for method in ['PHE6','RF_PFN6']:
            files=list((OUT/'references'/method/track).glob('*/*.json'))
            rows=[json.loads(p.read_text()) for p in files]
            refs[method]={'completed_conditions':len(rows),'expected_conditions':expected}
            if len(rows)==expected:
                refs[method]['means']={block:averages([r['metrics'] for r in rows if (r['rho']==0)==is_clean])
                    for block,is_clean in [('clean',True),('shift',False)]}
        result[track]={'completed_bank_conditions':count,'expected_bank_conditions':expected,
            'complete_dataset_grids':len(complete)//size,'means_on_complete_dataset_grids_only':means,'references':refs}
    atomic_json(OUT/'live_summary.json',result);print(json.dumps(result,indent=2))


def report():
    base={t:json.loads((OUT/f'frozen_{t}.json').read_text())['metrics'] for t in ['clean','five','seven']}
    shared={'five':{'r2':.237,'nrmse':.834},'seven':{'r2':.460,'nrmse':.656}}
    records=[{'method':'Base','annotation':'','backbone_weight_finetuning':False,**base}];raw=[]
    for method in METHODS+['PHE6','RF_PFN6']:
        row={'method':method,'annotation':'†‡','backbone_weight_finetuning':False};raw_row={'method':method}
        for track,expected in [('five',735),('seven',957)]:
            if method in METHODS:
                paths=sorted((OUT/'policies'/track).glob(f'fold_*_{method}.json'));assert len(paths)==5
                policies=[json.loads(p.read_text()) for p in paths];values=[r for p in policies for r in p['rows']]
                raw_values=[r for p in policies for r in p['raw_rows']]
            else:
                files=list((OUT/'references'/method/track).glob('*/*.json'))
                values=[{'rho':r['rho'],**r['metrics']} for p in files for r in [json.loads(p.read_text())]]
                raw_values=values
            assert len(values)==len(raw_values)==expected,(method,track,len(values))
            row[track]=averages([r for r in values if r['rho']>0]);raw_row[track]=averages([r for r in raw_values if r['rho']>0])
            if track=='five':
                row['clean']=averages([r for r in values if r['rho']==0]);raw_row['clean']=averages([r for r in raw_values if r['rho']==0])
            row[track+'_at_least_Base']=row[track]['r2']>=base[track]['r2']-1e-12 and row[track]['nrmse']<=base[track]['nrmse']+1e-12
            row[track+'_below_historical_Shared']=row[track]['r2']<=shared[track]['r2'] and row[track]['nrmse']>=shared[track]['nrmse']
        row['clean_within_005']=row['clean']['r2']>=base['clean']['r2']-.005 and row['clean']['nrmse']<=base['clean']['nrmse']+.005
        row['budget_group']='six trees, variable node contexts and prediction calls' if method=='RF_PFN6' else 'six logical candidates/configurations'
        row['adaptation']='Official source holdout selection in a controlled six-seed portfolio' if method=='PHE6' else 'Official adaptive forest; six trees, depth two and source-only pruning' if method=='RF_PFN6' else 'Input/context bank and combiner calibration using other dataset folds'
        records.append(row);raw.append(raw_row)
    result={'status':'COMPLETE','seed':0,'records':records,'raw_method_records':raw,'paper_Base_reference':TARGETS,
        'historical_Shared_reference':shared,'historical_Shared_not_evaluated_on_new_benchmark':True,
        'internal_development_only':True,'paper_protocol_reproduction':False,'cross_backbone_benchmarks_differ':True,
        'benchmark_calibrated_using_Base_development_labels':True,'Shared_bounds_not_used_for_selection':True,
        'interval_flags_diagnostic_only':True,'strengths':[.2,.5,.8,1.],'five_datasets':35,'seven_datasets':33,
        'annotations':{'†':'Dedicated input/context/budget adaptation','‡':'Calibration learning or source-only selection',
                       '§':'Backbone weight fine-tuning; none performed'},'all_raw_variants_preserved':True,'Markdown_written':False}
    atomic_json(OUT/'latest_summary.json',result)
    with (OUT/'averages.csv').open('w',newline='') as h:
        fields=['method','annotation','budget_group','Clean_R2','Clean_nRMSE','Five_R2','Five_nRMSE','Seven_R2','Seven_nRMSE',
                'five_at_least_Base','five_below_historical_Shared','seven_at_least_Base','seven_below_historical_Shared','clean_within_005']
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader()
        for r in records:
            writer.writerow({k:r.get(k,'') for k in fields if '_' not in k or k.startswith(('five_','seven_','clean_','budget_'))}|
                {name+'_'+metric:r[track][key] for track,name in [('clean','Clean'),('five','Five'),('seven','Seven')]
                 for metric,key in [('R2','r2'),('nRMSE','nrmse')]})
    for r in records:print(r['method'],*[f"{r[t]['r2']:.6f}/{r[t]['nrmse']:.6f}" for t in ['clean','five','seven']],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--live',action='store_true');a=p.parse_args()
    if a.live:live()
    else:report()
