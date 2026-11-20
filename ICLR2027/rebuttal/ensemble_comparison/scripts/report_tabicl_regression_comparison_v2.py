"""Report safety adaptations alongside preserved raw/first-stage comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from prepare_data import RUN,ROOT,atomic_json
from tabicl_regression_comparison_v2 import OUT,BENCH
from fit_tabicl_v2_calibration_gate import METHODS,predict
import fit_internal_regression_ensembles as original
from run_tabicl import metrics

SHARED={'five':{'r2':.363,'nrmse':.783},'seven':{'r2':.467,'nrmse':.659}}


def averages(rows,clean):
    values=[r for r in rows if (r['rho']==0)==clean]
    return {k:float(np.mean([r[k] for r in values])) for k in ['r2','nrmse']}


def flags(row,base):
    for track in ['five','seven']:
        row[track+'_at_least_Base']=row[track]['r2']>=base[track]['r2']-1e-12 and row[track]['nrmse']<=base[track]['nrmse']+1e-12
        row[track+'_below_historical_Shared']=row[track]['r2']<SHARED[track]['r2'] and row[track]['nrmse']>SHARED[track]['nrmse']
    row['Clean_within_005']=row['clean']['r2']>=base['clean']['r2']-.005 and row['clean']['nrmse']<=base['clean']['nrmse']+.005
    row['all_interval_checks_pass']=all(row[k] for k in ['five_at_least_Base','five_below_historical_Shared',
        'seven_at_least_Base','seven_below_historical_Shared','Clean_within_005'])


def write_csv(path,rows,gate):
    fields=['method','variant','annotation','Clean_R2','Clean_nRMSE','Five_R2','Five_nRMSE','Seven_R2','Seven_nRMSE']
    if gate:fields+=['five_Base_fallback_folds','seven_Base_fallback_folds','all_interval_checks_pass']
    with path.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for row in rows:
            record={'method':row['method'],'variant':row.get('variant','Base' if row['method']=='Base' else 'raw_defaults'),
                    'annotation':row.get('annotation','')}
            for track,title in [('clean','Clean'),('five','Five'),('seven','Seven')]:
                record[title+'_R2']=row[track]['r2'];record[title+'_nRMSE']=row[track]['nrmse']
            if gate:record.update({k:row.get(k,'') for k in fields[-3:]})
            w.writerow(record)


def report():
    assert (OUT/'first_stage_prediction_audit.json').exists()
    base={track:json.loads((OUT/f'frozen_{track}.json').read_text())['metrics'] for track in ['clean','five','seven']}
    first=json.loads((OUT/'first_stage_summary.json').read_text());records=[{'method':'Base','annotation':'',**base}]
    for method in METHODS:
        row={'method':method,'variant':'cross_fitted_calibration_safety_gate','annotation':'†‡',
            'backbone_weights_updated':False,'candidate_budget_maximum':6}
        for track,expected in [('five',735),('seven',957)]:
            policies=[json.loads(p.read_text()) for p in sorted((OUT/'safety_policies'/track).glob(f'fold_*_{method}.json'))]
            assert len(policies)==5
            values=[r for p in policies for r in p['rows']];assert len(values)==expected
            assert len({(r['task_id'],r['family'],r['rho']) for r in values})==expected
            row[track]=averages(values,False)
            row[track+'_Base_fallback_folds']=sum(p['Base_fallback'] for p in policies)
            row[track+'_Base_fallback_fold_ids']=[p['fold'] for p in policies if p['Base_fallback']]
            if track=='five':row['clean']=averages(values,True)
        flags(row,base);records.append(row)
    result={'status':'COMPUTATION_COMPLETE_AUDIT_PENDING','seed':0,'records':records,
        'default_variant':'explicitly_adapted_calibration_safety_gate','original_method_results':'first_stage_summary.json',
        'raw_default_results':'raw_defaults_averages.csv','first_stage_method_records':first['records'],
        'historical_Shared_reference':SHARED,'Shared_was_not_run_on_this_benchmark':True,
        'interval_checks_diagnostic_only':True,'Shared_bounds_not_used_for_parameter_selection':True,
        'benchmark_frozen_and_unchanged':True,'benchmark_role':'internal Base-calibrated development benchmark',
        'not_original_paper_protocol_reproduction':True,'outer_test_labels_not_used_for_method_fitting':True,
        'fallback_is_ensemble_gain':False,'Native_fallback_deploys_one_Base_instead_of_native_six':True,
        'six_is_candidate_budget_not_guaranteed_nonzero_member_count':True,
        'all_recommended_safety_versions_meet_diagnostic_intervals':all(r['all_interval_checks_pass'] for r in records[1:]),
        'all_original_methods_meet_intervals':all(all(r[k] for k in ['five_at_least_Base','five_below_historical_Shared',
            'seven_at_least_Base','seven_below_historical_Shared','clean_within_005']) for r in first['records'][1:]),
        'all_variants_retained':True,'backbone_parameter_finetuning':False,'Markdown_written':False,
        'annotations':{'†':'Dedicated context/calibration safety-gate adaptation','‡':'Calibration-dataset fitting/selection',
                       '§':'Backbone parameter fine-tuning; none performed'}}
    atomic_json(OUT/'latest_summary.json',result);write_csv(OUT/'averages.csv',records,True)
    raw=[records[0]]+[r|{'annotation':'' if r['method']=='Native6' else '†‡'} for r in first['raw_method_records']]
    write_csv(OUT/'raw_defaults_averages.csv',raw,False)
    for row in records:
        print(row['method'],*[f"{row[t]['r2']:.6f}/{row[t]['nrmse']:.6f}" for t in ['clean','five','seven']],
              'fallback',row.get('five_Base_fallback_folds'),row.get('seven_Base_fallback_folds'),flush=True)


def audit():
    summary=json.loads((OUT/'latest_summary.json').read_text());assert summary['default_variant']=='explicitly_adapted_calibration_safety_gate'
    assert json.loads((OUT/'first_stage_prediction_audit.json').read_text())['status']=='PASS'
    original.OUT=OUT;count=0;means={}
    for track,expected in [('five',735),('seven',957)]:
        cells,_=original.load(track);bykey={(int(c['meta']['task_id']),c['meta']['family'],c['meta']['rho']):c for c in cells}
        assert len(bykey)==expected
        for c in cells:c['iqr_order']=np.argsort(c['iqr'].mean(1),kind='stable')
        for method in METHODS:
            rows=[]
            for path in sorted((OUT/'safety_policies'/track).glob(f'fold_*_{method}.json')):
                policy=json.loads(path.read_text());cal=set(policy['calibration_data_ids']);test=set(policy['test_data_ids'])
                assert cal.isdisjoint(test) and not policy['test_labels_used_for_fitting'] and not policy['Shared_bounds_used_for_selection']
                assert not policy['benchmark_changed'] and not policy['backbone_weights_updated']
                assert np.isclose(policy['z_critical'],norm.ppf(1-.05/(2*policy['candidate_count'])),rtol=1e-12,atol=1e-12)
                for split in policy['inner_splits']:
                    tr=set(split['train_data_ids']);val=set(split['validation_data_ids'])
                    assert tr.isdisjoint(val) and tr|val==cal and (tr|val).isdisjoint(test)
                if not policy['Base_fallback']:assert policy['selected']['evidence']['passes']
                for row in policy['rows']:
                    c=bykey[(int(row['task_id']),row['family'],row['rho'])]
                    assert int(c['meta']['data_id']) in test and c['fold']==policy['fold']
                    p=OUT/'safety_predictions'/track/method/str(row['task_id'])/f"{row['family']}_{row['rho']}.npz"
                    with np.load(p) as z:prediction=z['prediction'];assert np.array_equal(z['y_target'],c['y'])
                    reconstructed=predict(c,policy['model']);assert np.allclose(prediction,reconstructed,rtol=1e-11,atol=1e-11)
                    if policy['Base_fallback']:assert np.array_equal(prediction,c['base'])
                    actual=metrics(c['y'],prediction,'regression',c['meta']['scale'])
                    for k in ['r2','nrmse']:assert np.isclose(actual[k],row[k],rtol=1e-10,atol=1e-10)
                    rows.append({'rho':row['rho'],**actual});count+=1
            assert len(rows)==expected;means[(method,track)]=averages(rows,False)
            if track=='five':means[(method,'clean')]=averages(rows,True)
    for row in summary['records'][1:]:
        for track in ['clean','five','seven']:
            for k in ['r2','nrmse']:assert np.isclose(row[track][k],means[(row['method'],track)][k],rtol=1e-10,atol=1e-10)
    lock=json.loads((OUT/'input_and_previous_results_lock.json').read_text())
    for path,sha in lock['sha256'].items():assert hashlib.sha256((RUN/path).read_bytes()).hexdigest()==sha,path
    assert json.loads((BENCH/'final_audit.json').read_text())['all_six_absolute_differences_at_most_005']
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    result={'status':'PASS','safety_prediction_vectors_recomputed':count,'original_prediction_audit':'first_stage_prediction_audit.json',
        'dataset_outer_and_inner_fit_validation_disjointness_verified':True,'predictions_reconstructed_from_models':True,
        'all_Base_fallbacks_equal_saved_actual_Base_predictions':True,'reported_means_verified':True,
        'benchmark_and_previous_completed_results_unchanged':True,'Shared_bounds_not_used_for_selection':True,
        'backbone_weights_updated':False,'Markdown_bytes':0}
    atomic_json(OUT/'safety_prediction_audit.json',result)
    summary['status']='COMPLETE_AND_AUDITED';summary['safety_prediction_audit']=result
    atomic_json(OUT/'latest_summary.json',summary)
    print('V2_SAFETY_PREDICTION_AUDIT_PASS',count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['report','audit']);a=p.parse_args()
    if a.phase=='report':report()
    else:audit()
