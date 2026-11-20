"""Check reference/target splits, TFM feature provenance and readout refits."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import torch
from tabicl import TabICLClassifier
import column_desync_supervised_probe_v375 as run
from prepare_data import RUN,ROOT,atomic_json,protocol
from run_tabicl import metrics


def main(pilot=False,replay=False):
    source=run.source;plan=json.loads((run.OUT/'plan.json').read_text());fp=json.loads((run.features.OUT/'plan.json').read_text())
    assert source.sha(run.__file__)==plan['script_sha256'] and source.sha(run.probe.__file__)==plan['probe_sha256']
    assert source.sha(run.features.OUT/'plan.json')==plan['features_plan_sha256']
    for module,key in [(run.features,'script_sha256'),(run.features.representation,'representation_sha256'),
                       (source.novel,'generator_sha256'),(source.constructors,'registry_sha256'),(source.fusion,'fusion_sha256')]:
        assert source.sha(module.__file__)==fp[key]
    cases=run.features.schedule.endpoint.endpoint_cases()
    if pilot:cases=[c for c in cases if c['task_id'] in fp['pilot_task_ids']]
    if replay:torch.set_num_threads(1);torch.manual_seed(0);assert int(os.environ['CUDA_VISIBLE_DEVICES']) in range(6)
    rows=[];count=0;contexts=0;replays=[]
    for c in cases:
        tid=c['task_id'];assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:X0,y0,Q0,num,sids,qids=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        assert source.sha(c['reference_file'])==c['reference_sha256']
        with np.load(c['reference_file']) as z:RX,Ry,CX,Cy,rids,cids=(z[k] for k in ('X_train_reference','y_train_reference','X_calibration','y_calibration','train_reference_indices','calibration_indices'))
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:apos,epos,ay,aids,eids=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y','adaptation_indices','evaluation_indices'))
        assert set(apos).isdisjoint(epos) and set(np.r_[rids,cids,sids]).isdisjoint(qids)
        assert set(np.r_[rids,aids]).isdisjoint(eids) and set(np.r_[rids,aids]).isdisjoint(cids)
        cpdir=run.features.schedule.endpoint.oldbank.training.OUT/'checkpoints'/'perturbation'/'classification'/str(tid)
        cp=json.loads((cpdir/'result.json').read_text());ca=json.loads((cpdir/'input_audit.json').read_text())
        assert not cp['target_adaptation_labels_used_for_weights'] and ca['source_train_validation_disjoint_from_entire_original_query']
        assert source.sha(cp['checkpoint_file'])==cp['checkpoint_sha256']
        pfile=source.OUT/'policies'/'classification'/f'{c["fold"]}.json';gate=json.loads(pfile.read_text());assert c['data_id'] not in gate['training_data_ids']
        allQ=np.r_[RX,CX,Q0];offset=len(Ry)+len(Cy)
        for rho in source.novel.STRENGTHS:
            path=run.OUT/'predictions'/str(tid)/f'column_desynchronization_{rho}.npz'
            meta=json.loads(path.with_suffix('.json').read_text());assert source.sha(path)==meta['output_sha256']
            assert meta['reference_sha256']==c['reference_sha256'] and meta['adaptation_sha256']==c['adaptation_sha256'] and not meta['evaluation_labels_opened']
            assert meta['adaptation_positions']==apos.tolist() and meta['evaluation_positions']==epos.tolist()
            assert protocol.array_sha256(np.r_[rids,aids])==meta['training_row_ids_sha256']
            assert protocol.array_sha256(cids)==meta['validation_source_row_ids_sha256'] and protocol.array_sha256(eids)==meta['evaluation_row_ids_sha256']
            f=run.features.OUT/'features'/str(tid)/path.name
            fm=json.loads(f.with_suffix('.json').read_text());assert source.sha(f)==fm['output_sha256']==meta['features_sha256']
            assert fm['checkpoint_sha256']==cp['checkpoint_sha256'] and fm['input_sha256']==c['input_sha256'] and fm['reference_sha256']==c['reference_sha256']
            assert fm['policy_sha256']==source.sha(pfile) and not fm['target_labels_opened'] and fm['target_labels_in_context']==0 and fm['context_rows']==256
            assert fm['reference_rows']==len(Ry) and fm['calibration_rows']==len(Cy) and fm['original_query_rows']==len(Q0)
            assert protocol.array_sha256(np.r_[rids,cids,qids])==fm['prediction_row_ids_sha256']
            X,y,Q,_=source.novel.generate(X0,y0,Q0,num,'classification','column_desynchronization',rho,tid)
            for v,key in [(X,'observed_source_X_sha256'),(y,'observed_source_y_sha256'),(allQ,'unlabeled_prediction_X_sha256')]:assert protocol.array_sha256(v)==fm[key]
            views=source.constructors.shared_contexts(X,y,Q,len(y),'classification',None)
            order=source.constructors.specific.importance(X,y.astype(float));columns=np.array([j for j in range(X.shape[1]) if j!=int(order[0])])
            if not len(columns):columns=np.array([0])
            for (name,A,b,T),saved in zip(views,fm['contexts']):
                predictX=allQ[:,columns] if name=='feature_subspace' else allQ
                assert name==saved['name'] and len(b)==saved['rows'] and len(b)<=256
                for value,key in [(A,'X_sha256'),(b,'y_sha256'),(predictX,'prediction_X_sha256')]:assert protocol.array_sha256(value)==saved[key]
                contexts+=1
            with np.load(f) as features,np.load(path) as actual:
                dim=fm['projected_representation_dimension'];native,shared,bank=features['Native_raw'],features['Shared_raw'],features['members']
                assert len(native)==len(allQ)
                shared_expected=source.fusion.predict({'task':'classification','base':native,'bank':bank,'mu':float(y.mean()),'sigma':max(float(y.std()),1e-12)},gate['Shared'])
                np.testing.assert_allclose(shared,shared_expected,atol=1e-12,rtol=0)
                np.testing.assert_array_equal(features['Native_features'][:,:dim],features['Shared_features'][:,:dim])
                log=lambda p:np.log(np.clip(p,1e-12,1.))
                np.testing.assert_allclose(features['Native_features'][:,dim:],log(native),atol=2e-6,rtol=2e-6)
                expected=np.c_[log(native),log(shared),log(bank).transpose(1,0,2).reshape(len(allQ),-1)]
                np.testing.assert_allclose(features['Shared_features'][:,dim:],expected,atol=2e-6,rtol=2e-6)
                if replay and rho==source.novel.STRENGTHS[0]:
                    predictor=run.features.representation.LatentPredictor('classification','cuda:0')
                    predictor.models[1]=TabICLClassifier(n_estimators=1,model_path=cp['checkpoint_file'],allow_auto_download=False,
                        device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
                    p,phi=predictor.core(X,y,allQ,len(np.unique(y0)))
                    projected=run.features.projection(phi)
                    np.testing.assert_allclose(p,native,atol=2e-6,rtol=2e-6)
                    np.testing.assert_allclose(projected,features['Native_features'][:,:dim],atol=2e-6,rtol=2e-6)
                    replays.append({'task_id':tid,'rho':rho,'prediction_max_abs_difference':float(np.max(np.abs(p-native))),
                        'embedding_max_abs_difference':float(np.max(np.abs(projected-features['Native_features'][:,:dim])))})
                    del predictor,phi;torch.cuda.empty_cache()
                for name in ['Native','Shared']:
                    result,policy,oof=run.probe.adapt(features[name+'_features'],features[name+'_raw'],Ry,Cy,ay,apos,epos,source_prior=np.bincount(y0.astype(int))/len(y0))
                    np.testing.assert_allclose(result,actual[name+'_probe'],atol=1e-10,rtol=1e-10)
                    saved=meta[name+'_policy'];assert policy['kind']==saved['kind'] and policy['C']==saved['C'] and policy['alpha']==saved['alpha']
                    np.testing.assert_allclose(policy['score'],saved['score'],atol=1e-10,rtol=1e-10)
                    if policy['model'] is not None:
                        for key in ['mean','scale']+(['coef','intercept'] if policy['model']['kind']=='logistic' else []):
                            np.testing.assert_allclose(policy['model'][key],saved['model'][key],atol=1e-10,rtol=1e-10)
                    for C,value in oof.items():
                        np.testing.assert_allclose(value['target'],actual[f'{name}_cv_target_{C}'],atol=1e-10,rtol=1e-10)
                        np.testing.assert_allclose(value['source_validation'],actual[f'{name}_cv_source_{C}'],atol=1e-10,rtol=1e-10)
                    np.testing.assert_array_equal(actual[name+'_raw'],features[name+'_raw'][offset+epos])
                original=source.OUT/'predictions'/'perturbation'/'classification'/str(tid)/path.name
                assert source.sha(original)==meta['original_prediction_sha256']
                with np.load(original) as z:np.testing.assert_array_equal(actual['Base'],z['Base'][epos])
                assert source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
                with np.load(c['heldout_truth_file']) as truth_np:truth=truth_np['y_target'];np.testing.assert_array_equal(truth_np['target_indices'],eids)
                for method in ['Base','Native_raw','Shared_raw','Native_probe','Shared_probe']:
                    rows.append({'task_id':tid,'data_id':c['data_id'],'rho':rho,'method':method,**metrics(truth,actual[method],'classification',1.)})
            count+=1
        print('HEAVY_PROBE_AUDIT_CASE',tid,flush=True)
    assert count==4*len(cases)
    prefix='pilot_' if pilot else ''
    df=pd.DataFrame(rows);df.to_csv(run.OUT/f'{prefix}condition_metrics.csv',index=False)
    ds=df.groupby(['data_id','method'])[['accuracy','auc']].mean().reset_index()
    table=ds.groupby('method')[['accuracy','auc']].mean().reset_index();table['datasets']=len(cases)
    table.to_csv(run.OUT/f'{prefix}family_averages.csv',index=False)
    get=lambda method:float(table.loc[table.method==method,'accuracy'].iloc[0]);gain=get('Shared_probe')-get('Base')
    atomic_json(run.OUT/f'{prefix}numerical_acceptance.json',{'pilot_only':pilot,'active_endpoint':list(run.features.schedule.endpoint.ENDPOINT),
        'conditions':count,'datasets':len(cases),'Base':get('Base'),'Shared_supervised_probe':get('Shared_probe'),'gain':gain,
        'gain_vs_Native_same_supervision':get('Shared_probe')-get('Native_probe'),
        'required_gain_strictly_greater_than':.03,'active_endpoint_pass':not pilot and count==288 and gain>.03,
        'all_twelve_pass':False,'extra_source_and_test_subset_supervision_explicit':True})
    assert run.features.schedule.verify_frozen()==plan['frozen_registry_sha256']
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(run.OUT/f'{prefix}independent_audit.json',{'status':'PASS','pilot_only':pilot,'conditions':count,
        'TFM_contexts_reconstructed':contexts,'reference_and_target_and_eval_row_ids_checked':True,
        'TFM_feature_extraction_uses_no_target_labels':True,'linear_heads_and_four_fold_validation_predictions_independently_refitted':True,
        'all_target_adaptation_rows_used_only_as_training_or_validation_and_excluded_from_evaluation':True,
        'original_frozen_Base_predictions_bitwise_preserved':True,'frozen_endpoint_files_unchanged':True,
        'GPU_representation_spot_replays':replays,'not_every_neural_prediction_recomputed':True,
        'Markdown_bytes':0,'script_sha256':source.sha(__file__)})
    print(table.to_string(index=False),flush=True);print('GAIN',gain,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args();main(a.pilot,a.replay)
