"""Replay nonlinear readout fitting and verify unchanged upstream TFM data."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import pandas as pd
import heavy_nonlinear_probe_v350 as run
from prepare_data import RUN,ROOT,atomic_json,protocol
from run_tabicl import metrics


def main(pilot=False):
    source=run.source;plan=json.loads((run.OUT/'plan.json').read_text())
    assert source.sha(run.__file__)==plan['script_sha256'] and source.sha(run.probe.__file__)==plan['probe_sha256']
    assert source.sha(RUN/'scripts/linear_probe_adapter_v344.py')==plan['scoring_helper_sha256']
    assert source.sha(run.features.OUT/'plan.json')==plan['features_plan_sha256']
    # Feature provenance was independently reconstructed in the linear audit.
    upstream=RUN/'results/heavy_supervised_probe_v345'/('pilot_independent_audit.json' if pilot else 'independent_audit.json')
    audit=json.loads(upstream.read_text());assert audit['status']=='PASS'
    covered=pd.read_csv(upstream.with_name(upstream.name.replace('independent_audit.json','condition_metrics.csv')))
    keys=set(covered[['task_id','rho']].itertuples(index=False,name=None))
    fp=json.loads((run.features.OUT/'plan.json').read_text());cases=run.features.schedule.endpoint.endpoint_cases()
    if pilot:cases=[c for c in cases if c['task_id'] in fp['pilot_task_ids']]
    rows=[];count=0
    for c in cases:
        tid=c['task_id'];assert source.sha(c['reference_file'])==c['reference_sha256']
        with np.load(c['reference_file']) as z:Ry,Cy,rids,cids=(z[k] for k in ('y_train_reference','y_calibration','train_reference_indices','calibration_indices'))
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:apos,epos,ay,aids,eids=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y','adaptation_indices','evaluation_indices'))
        assert set(np.r_[rids,aids]).isdisjoint(eids) and set(np.r_[rids,aids]).isdisjoint(cids)
        for rho in source.novel.STRENGTHS:
            assert (tid,rho) in keys
            path=run.OUT/'predictions'/str(tid)/f'cell_heavy_tail_{rho}.npz'
            meta=json.loads(path.with_suffix('.json').read_text());assert source.sha(path)==meta['output_sha256']
            assert meta['reference_sha256']==c['reference_sha256'] and meta['adaptation_sha256']==c['adaptation_sha256']
            assert meta['adaptation_positions']==apos.tolist() and meta['evaluation_positions']==epos.tolist() and not meta['evaluation_labels_opened']
            assert protocol.array_sha256(np.r_[rids,aids])==meta['training_row_ids_sha256']
            assert protocol.array_sha256(cids)==meta['validation_source_row_ids_sha256'] and protocol.array_sha256(eids)==meta['evaluation_row_ids_sha256']
            f=run.features.OUT/'features'/str(tid)/path.name
            fm=json.loads(f.with_suffix('.json').read_text());assert source.sha(f)==meta['features_sha256']==fm['output_sha256']
            assert not fm['target_labels_opened'] and fm['target_labels_in_context']==0
            with np.load(f) as feature,np.load(path) as actual:
                offset=len(Ry)+len(Cy)
                for name in ['Native','Shared']:
                    prediction,policy,oof=run.probe.adapt(feature[name+'_features'],feature[name+'_raw'],Ry,Cy,ay,apos,epos)
                    saved=meta[name+'_policy'];assert policy['kind']==saved['kind'] and policy['min_samples_leaf']==saved['min_samples_leaf'] and policy['alpha']==saved['alpha']
                    np.testing.assert_allclose(policy['score'],saved['score'],atol=1e-12,rtol=0)
                    np.testing.assert_allclose(prediction,actual[name+'_probe'],atol=1e-12,rtol=0)
                    if saved['model'] is not None:
                        assert source.sha(saved['model']['file'])==saved['model']['sha256']
                        model=joblib.load(saved['model']['file']);assert model.classes_.tolist()==saved['model']['classes']
                        params=model.get_params();assert params['n_estimators']==128 and params['min_samples_leaf']==saved['min_samples_leaf']
                        assert params['max_features']=='sqrt' and params['random_state']==0 and params['n_jobs']==1
                        F=feature[name+'_features'][offset+epos];K=feature[name+'_raw'].shape[1]
                        np.testing.assert_allclose(run.probe.predict(model,F,K),run.probe.predict(policy['model'],F,K),atol=1e-12,rtol=0)
                    else:assert policy['model'] is None
                    for leaf,value in oof.items():
                        np.testing.assert_allclose(value['target'],actual[f'{name}_cv_target_{leaf}'],atol=1e-12,rtol=0)
                        np.testing.assert_allclose(value['source_validation'],actual[f'{name}_cv_source_{leaf}'],atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(actual[name+'_raw'],feature[name+'_raw'][offset+epos])
                original=source.OUT/'predictions'/'perturbation'/'classification'/str(tid)/path.name
                assert source.sha(original)==meta['original_prediction_sha256']
                with np.load(original) as z:np.testing.assert_array_equal(actual['Base'],z['Base'][epos])
                assert source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
                with np.load(c['heldout_truth_file']) as z:truth=z['y_target'];np.testing.assert_array_equal(z['target_indices'],eids)
                for method in ['Base','Native_raw','Shared_raw','Native_probe','Shared_probe']:
                    rows.append({'task_id':tid,'data_id':c['data_id'],'rho':rho,'method':method,**metrics(truth,actual[method],'classification',1.)})
            count+=1
        print('NONLINEAR_PROBE_AUDIT_CASE',tid,flush=True)
    assert count==4*len(cases)
    prefix='pilot_' if pilot else '';df=pd.DataFrame(rows);df.to_csv(run.OUT/f'{prefix}condition_metrics.csv',index=False)
    ds=df.groupby(['data_id','method'])[['accuracy','auc']].mean().reset_index()
    table=ds.groupby('method')[['accuracy','auc']].mean().reset_index();table['datasets']=len(cases)
    table.to_csv(run.OUT/f'{prefix}family_averages.csv',index=False)
    get=lambda name:float(table.loc[table.method==name,'accuracy'].iloc[0]);gain=get('Shared_probe')-get('Base')
    atomic_json(run.OUT/f'{prefix}numerical_acceptance.json',{'pilot_only':pilot,'active_endpoint':['perturbation','classification','cell_heavy_tail'],
        'conditions':count,'datasets':len(cases),'Base':get('Base'),'Shared_nonlinear_probe':get('Shared_probe'),'gain':gain,
        'gain_vs_Native_same_supervision':get('Shared_probe')-get('Native_probe'),'required_gain_strictly_greater_than':.05,
        'active_endpoint_pass':not pilot and count==256 and gain>.05,'all_twelve_pass':False,'extra_source_and_test_subset_supervision':True})
    assert run.features.schedule.verify_frozen()==plan['frozen_registry_sha256']
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(run.OUT/f'{prefix}independent_audit.json',{'status':'PASS','pilot_only':pilot,'conditions':count,
        'upstream_TFM_feature_audit_sha256':source.sha(upstream),'all_readout_models_and_OOF_predictions_independently_refitted':True,
        'saved_models_reloaded_and_probabilities_checked':True,'fixed_eval_row_ids_and_original_Base_checked':True,
        'frozen_passed_endpoints_unchanged':True,'Markdown_bytes':0,'script_sha256':source.sha(__file__)})
    print(table.to_string(index=False),flush=True);print('GAIN',gain,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)
