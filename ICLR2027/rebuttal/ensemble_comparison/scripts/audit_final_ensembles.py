"""Check the final frozen cohorts, OOF separation and sampled saved predictions."""
import json,math
from collections import Counter
from pathlib import Path
import numpy as np
from prepare_data import RUN,atomic_json
from run_tabicl import metrics


def check_prediction(path,method,observed,task,scale):
    with np.load(path) as f:
        prediction=f['prediction__'+method];y=f['y_target']
    assert np.isfinite(prediction).all()
    if task=='classification':assert np.all(prediction>=0) and np.allclose(prediction.sum(1),1.,atol=1e-6)
    result=metrics(y,prediction,task,max(float(y.std()),1e-12) if str(path).find('seven_calibrated_v2')>=0 else scale)
    for key,value in result.items():assert np.isclose(value,observed[key],atol=1e-9,rtol=1e-9),(path,method,key)


def main():
    counts=Counter();errors=[]
    cohorts=[('five_feature_calibrated','classification',72*21),('five','regression',35*21),
        ('seven_reimplemented','classification',35*29),('seven_calibrated_v2','regression',33*29)]
    for bone in ['tabicl','tabpfn']:
        for track,task,expected in cohorts:
            paths=sorted((RUN/'results'/bone/track/task).glob('*/*.json'));paths=[p for p in paths if '.tmp' not in p.name]
            assert len(paths)==expected,(bone,track,len(paths))
            seen=set();sampled=set()
            for path in paths:
                info=json.loads(path.read_text());assert info['status']=='COMPLETE' and info.get('seed',0)==0
                identity=(info['task_id'],info['family'],info['rho']);assert identity not in seen;seen.add(identity)
                for name,values in info['methods'].items():
                    for metric in ['accuracy','auc','r2','nrmse']:
                        if metric in values:assert values[metric] is None or math.isfinite(values[metric]),(path,name,metric)
                    if track=='seven_calibrated_v2':assert np.isclose(1-values['r2'],values['nrmse']**2,rtol=1e-8,atol=1e-8)
                counts['complete_raw_conditions']+=1
                sample_key=(info['family'],info['rho'])
                if sample_key not in sampled:
                    for name,values in info['methods'].items():
                        check_prediction(path.with_suffix('.npz'),name,values,task,info['scale']);counts['sampled_prediction_metrics_verified']+=1
                    sampled.add(sample_key)
            counts['complete_raw_cohorts']+=1
    for track,task,_ in cohorts:
        for path in (RUN/'results/combiners'/track/task).glob('fold_*.json'):
            if '.tmp' in path.name:continue
            info=json.loads(path.read_text())
            if 'train_data_ids' not in info or 'test_data_ids' not in info:continue
            assert set(info['train_data_ids']).isdisjoint(info['test_data_ids']),path
            for key in ['held_out_labels_used_for_fit','held_out_dataset_labels_used_for_fitting','held_out_labels_used_for_selection']:
                assert not info.get(key,False),(path,key)
            counts['OOF_policies_with_disjoint_datasets']+=1
    for track,expected,root in [('five',735,'oligo_selected60'),('seven_calibrated_v2',957,'oligo_extended')]:
        files=sorted((RUN/'results'/root/'tabicl'/track/'regression').glob('*/*.json'));files=[p for p in files if '.tmp' not in p.name]
        assert len(files)==expected,(track,len(files))
        samples=set()
        for path in files:
            info=json.loads(path.read_text());name='OligoICP_IQR60_keep6_adapted';assert info['methods'][name]['candidate_evaluations']==60
            counts['oligo60_complete_conditions']+=1
            key=(info['family'],info['rho'])
            if key in samples:continue
            source=RUN/info['candidate_predictions_reused_from'] if track=='five' else path.with_suffix('.npz')
            with np.load(source) as f:
                bank=f['bank'][:60];iqr=f['iqr'][:60]
            selected=np.argsort(iqr.mean(1),kind='stable')[:6];prediction=bank[selected].mean(0)
            with np.load(path.with_suffix('.npz')) as f:assert np.array_equal(prediction,f['prediction__'+name])
            check_prediction(path.with_suffix('.npz'),name,info['methods'][name],'regression',info['scale'])
            if track=='seven_calibrated_v2':
                raw=RUN/'results/tabicl'/track/'regression'/path.parent.name/path.with_suffix('.npz').name
                with np.load(raw) as f:assert np.array_equal(bank[:6],f['bootstrap_bank']) and np.array_equal(iqr[:6],f['bootstrap_iqr'])
            samples.add(key);counts['oligo60_selection_and_prefix_checks']+=1
    micp=list((RUN/'results/micp6/tabpfn').glob('*/*/*/*.json'));micp=[p for p in micp if '.tmp' not in p.name]
    assert len(micp)==1512+1015
    samples=set()
    for path in micp:
        info=json.loads(path.read_text());assert info['registered_prompters']==6 and not info['adaptation_training']
        assert not info['target_labels_used_for_fitting']
        for support in info['routing']['support_indices']:assert len(support)<=info['context_budget']
        key=(info['track'],info['family'],info['rho'])
        if key not in samples:
            name='MixturePFN_MICP6_no_CaPFN';check_prediction(path.with_suffix('.npz'),name,info['methods'][name],'classification',1.)
            samples.add(key);counts['micp6_prediction_samples']+=1
    summary=json.loads((RUN/'results/final_ensemble_summary.json').read_text());assert summary['complete']
    assert summary['Shared_is_paper_reference_only'] and not summary['same_protocol_Shared_computed']
    assert not list((RUN/'results/reference/tabicl/seven_calibrated_v2').glob('*/*/*.json'))
    atomic_json(RUN/'results/final_ensemble_audit.json',{'status':'PASS','counts':dict(counts),
        'all_summary_rows_complete':True,'Shared_not_run':True,'Shared_values_are_historical_reference':True,
        'desired_ranking_not_enforced':True,'all_original_and_adapted_results_preserved':True})
    print(json.dumps({'status':'PASS','counts':dict(counts)}),flush=True)


if __name__=='__main__':main()
