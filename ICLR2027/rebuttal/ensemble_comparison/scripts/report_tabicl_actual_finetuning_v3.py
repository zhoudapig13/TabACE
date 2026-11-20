"""Actual weight-fine-tuning results and independent artifact/data-flow checks."""
import argparse
import csv
import hashlib
import json
import numpy as np
import torch
from prepare_data import RUN,ROOT,atomic_json,protocol
from internal_regression_benchmark import manifest
from recalibrate_tabicl_regression_v2 import OUT as BENCH
from train_tabicl_robust_regression_v3 import OUT,CKPT
from evaluate_tabicl_finetuned_ensembles_v3 import METHODS,prediction
from tabicl_regression_comparison_v2 import load_condition
from run_tabicl import metrics


def average(rows,clean):
    values=[r for r in rows if (r['rho']==0)==clean]
    return {k:float(np.mean([r[k] for r in values])) for k in ['r2','nrmse']}


def report(tag):
    records={};base={t:json.loads((BENCH/f'frozen_{t}.json').read_text())['metrics'] for t in ['clean','five','seven']}
    for method in METHODS:records[method]={'method':method,'annotation':'§' if method in ['FT_single','Native6','Random6'] else '†‡§',
        'extra_clean_TRAIN_supervision':True,'raw_Base_fallback':False}
    for track,expected in [('five',735),('seven',957)]:
        rows=[json.loads(p.read_text()) for p in (OUT/'benchmark_predictions'/tag/track).glob('*/*.json')]
        assert len(rows)==expected,(track,len(rows),expected)
        assert len({(r['task_id'],r['family'],r['rho']) for r in rows})==expected
        for method in METHODS:
            values=[{'rho':r['rho'],**r['methods'][method]} for r in rows]
            result=average(values,False);row=records[method];row[track]=result
            row[track+'_R2_gain']=result['r2']-base[track]['r2'];row[track+'_gain_at_least_002']=row[track+'_R2_gain']>=.02-1e-12
            if track=='five':row['clean']=average(values,True)
    for r in records.values():r['Clean_in_desired_soft_band']=.71<=r['clean']['r2']<=.74
    summary={'status':'COMPUTATION_COMPLETE_AUDIT_PENDING','tag':tag,'seed':0,
        'records':[{'method':'Original_Base','annotation':'','extra_clean_TRAIN_supervision':False,**base},*records.values()],
        'all_ensemble_methods_gain_at_least_002_both_environments':all(r['five_gain_at_least_002'] and r['seven_gain_at_least_002'] for m,r in records.items() if m!='FT_single'),
        'fine_tuned_single_control_reported':True,'backbone_weight_updates_verified':True,
        'additional_supervision':'Authorized pristine labels from official TRAIN source, with disjoint train/validation subsets',
        'benchmark_query_labels_used_for_fit_selection_or_stopping':False,'raw_Base_fallback_used':False,
        'Clean_band_is_soft_only':True,'benchmark_unchanged':True,'historical_Shared_not_rerun':True,
        'historical_Shared_reference':{'five':{'r2':.363,'nrmse':.783},'seven':{'r2':.467,'nrmse':.659}},
        'candidate_budget':6,'single_estimator_control_budget':1,'Markdown_written':False,
        'annotation_legend':{'§':'Actual updated backbone checkpoint; additional clean TRAIN supervision',
                             '†':'Context/view or IQR budget adaptation','‡':'TRAIN-validation combiner fitting/selection'},
        'internal_Base_calibrated_benchmark_not_original_protocol_reproduction':True}
    dest=OUT/'reports'/tag;dest.mkdir(parents=True,exist_ok=True)
    atomic_json(dest/'latest_summary.json',summary)
    with (dest/'averages.csv').open('w',newline='') as h:
        fields=['method','annotation','Clean_R2','Clean_nRMSE','Five_R2','Five_nRMSE','Five_R2_gain','Seven_R2','Seven_nRMSE','Seven_R2_gain','extra_clean_TRAIN_supervision']
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for row in summary['records']:
            d={k:row.get(k,'') for k in ['method','annotation','extra_clean_TRAIN_supervision']}
            for track,name in [('clean','Clean'),('five','Five'),('seven','Seven')]:
                d[name+'_R2']=row[track]['r2'];d[name+'_nRMSE']=row[track]['nrmse']
                if track!='clean':d[name+'_R2_gain']=row.get(track+'_R2_gain','')
            w.writerow(d)
    for r in summary['records']:
        print(r['method'],*[f"{r[t]['r2']:.6f}/{r[t]['nrmse']:.6f}" for t in ['clean','five','seven']],
            'gain',r.get('five_R2_gain'),r.get('seven_R2_gain'),flush=True)
    print('ALL_ENSEMBLE_R2_GAIN_TARGETS',summary['all_ensemble_methods_gain_at_least_002_both_environments'],flush=True)


def audit(tag):
    dest=OUT/'reports'/tag;summary=json.loads((dest/'latest_summary.json').read_text());rows_by_method={};count=0
    original=torch.load(CKPT,map_location='cpu',weights_only=False)['state_dict'];checkpoint_checks=[]
    for m in manifest():
        folder=OUT/'checkpoints'/tag/str(m['task_id']);training=json.loads((folder/'training_result.json').read_text())
        assert training['actual_backbone_weights_updated'] and not training['benchmark_query_labels_used_for_training_or_selection']
        weights=torch.load(folder/'best.ckpt',map_location='cpu',weights_only=False)['state_dict'];changed=[]
        for key in original:
            if not torch.equal(original[key],weights[key]):changed.append(key)
        assert changed and all(k.startswith('icl_predictor.') for k in changed)
        assert not any(k.startswith(('col_embedder.','row_interactor.')) for k in changed)
        split=json.loads((folder/'source_split.json').read_text());tr=set(split['train_original_row_ids']);val=set(split['validation_original_row_ids'])
        assert tr.isdisjoint(val)
        with np.load(RUN/'cache/five/regression'/f"{m['task_id']}.npz") as z:
            assert not (tr|val).intersection(set(z['target_indices'].tolist()))
        checkpoint_checks.append({'task_id':m['task_id'],'changed_tensors':len(changed),'checkpoint_sha256':hashlib.sha256((folder/'best.ckpt').read_bytes()).hexdigest()})
        policy=json.loads((OUT/'combiner_policies'/tag/f"{m['task_id']}.json").read_text())
        assert not policy['benchmark_query_labels_used'] and not policy['original_Base_fallback']
        assert policy['checkpoint_sha256']==checkpoint_checks[-1]['checkpoint_sha256']
        for p in (OUT/'source_validation_banks'/tag/str(m['task_id'])).glob('*.json'):
            meta=json.loads(p.read_text());assert set(meta['context_row_ids'])<=tr and set(meta['query_row_ids'])<=val
            assert not meta['benchmark_query_labels_used']
    bytask={m['task_id']:m for m in manifest()}
    for track,expected in [('five',735),('seven',957)]:
        paths=list((OUT/'benchmark_predictions'/tag/track).glob('*/*.npz'));assert len(paths)==expected
        for p in paths:
            meta=json.loads(p.with_suffix('.json').read_text());m=bytask[meta['task_id']]
            data,info,base=load_condition(track,m,meta['family'],meta['rho'])
            assert meta['input_hashes']=={k:protocol.array_sha256(data[k]) for k in ['X_source','y_source','X_target','y_target']}
            policy=json.loads((OUT/'combiner_policies'/tag/f"{m['task_id']}.json").read_text())
            with np.load(p) as z:
                b={k:z[k] for k in ['feature','bootstrap','iqr','single','native']};t=z['y_target']
                assert np.array_equal(t,data['y_target']) and np.array_equal(z['original_Base'],base)
                assert b['feature'].shape==b['bootstrap'].shape==b['iqr'].shape==(6,len(t))
                assert np.array_equal(b['feature'][0],b['single'])
                baseline=metrics(t,base,'regression',info['scale']);rows_by_method.setdefault(('Original_Base',track),[]).append({'rho':meta['rho'],**baseline})
                for method in METHODS:
                    pred=z['prediction__'+method];rebuilt=prediction(b,method,policy['models'][method])
                    assert np.allclose(pred,rebuilt,rtol=1e-10,atol=1e-10)
                    actual=metrics(t,pred,'regression',info['scale'])
                    for k in ['r2','nrmse']:assert np.isclose(actual[k],meta['methods'][method][k],rtol=1e-10,atol=1e-10)
                    rows_by_method.setdefault((method,track),[]).append({'rho':meta['rho'],**actual});count+=1
    for r in summary['records']:
        for track in ['five','seven']:
            observed=average(rows_by_method[(r['method'],track)],False)
            for k in observed:assert np.isclose(observed[k],r[track][k],rtol=1e-10,atol=1e-10)
        observed=average(rows_by_method[(r['method'],'five')],True)
        for k in observed:assert np.isclose(observed[k],r['clean'][k],rtol=1e-10,atol=1e-10)
    for path,sha in json.loads((OUT/'benchmark_lock.json').read_text()).items():assert hashlib.sha256((RUN/path).read_bytes()).hexdigest()==sha
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    result={'status':'PASS','checkpoint_weight_changes_verified':checkpoint_checks,
        'prediction_vectors_recomputed':count,'all_original_benchmark_input_hashes_verified':1692,
        'pristine_training_and_validation_rows_disjoint_from_benchmark_queries':True,
        'no_raw_Base_fallback':True,'predictions_rebuilt_from_updated_checkpoint_banks_and_calibrated_models':True,
        'benchmark_frozen_files_unchanged':True,'Markdown_bytes':0}
    atomic_json(dest/'audit.json',result);summary['status']='COMPLETE_AND_AUDITED';summary['audit']='audit.json';atomic_json(dest/'latest_summary.json',summary)
    print('TRUE_FINETUNING_ENSEMBLE_AUDIT_PASS',count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['report','audit']);p.add_argument('--tag',default='initial128');a=p.parse_args()
    torch.set_num_threads(1)
    if a.phase=='report':report(a.tag)
    else:audit(a.tag)
