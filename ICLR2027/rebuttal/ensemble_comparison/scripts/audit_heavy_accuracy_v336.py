"""Audit target-label gradient splits, checkpoint tensors and paired inference."""
import argparse
import gc
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import torch
import heavy_accuracy_inference_v335 as run
from audit_dataset_finetuning_v308 import state_digest
from prepare_data import ROOT,atomic_json,protocol
from run_tabicl import metrics


def main(pilot=False):
    source=run.source;training=run.training
    tp=json.loads((training.OUT/'plan.json').read_text());ip=json.loads((run.OUT/'plan.json').read_text())
    assert source.sha(training.__file__)==tp['script_sha256']
    assert source.sha(training.implementation.__file__)==tp['implementation_sha256']
    assert source.sha(training.implementation.compatible.__file__)==tp['compatibility_sha256']
    for module,key in [(run,'script_sha256'),(source.novel,'generator_sha256'),(source.constructors,'registry_sha256'),(source.fusion,'fusion_sha256')]:
        assert source.sha(module.__file__)==ip[key]
    cases=json.loads((training.OUT/'case_manifest.json').read_text())
    if pilot:cases=[c for c in cases if c['task_id'] in tp['pilot_task_ids']]
    rows=[];checkpoints=[];count=0;context_count=0
    for c in cases:
        tid=c['task_id'];assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:X0,y0,Q0,num,sid,qid=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        assert source.sha(c['reference_file'])==c['reference_sha256']
        with np.load(c['reference_file']) as z:RX,Ry,rid,CX,Cy,cid=(z[k] for k in ('X_train_reference','y_train_reference','train_reference_indices','X_calibration','y_calibration','calibration_indices'))
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        nval=max(4,len(apos)//4);ta,va=apos[:-nval],apos[-nval:]
        ids=np.r_[sid,rid,np.tile(qid[ta],4)];vids=np.r_[cid,qid[va]]
        assert set(ids).isdisjoint(vids) and set(np.r_[ids,vids]).isdisjoint(qid[epos])
        assert set(np.r_[sid,rid,cid]).isdisjoint(qid) and set(cid).isdisjoint(np.r_[sid,rid])
        dest=training.OUT/'checkpoints'/str(tid)
        result=json.loads((dest/'result.json').read_text());a=json.loads((dest/'input_audit.json').read_text())
        assert a['target_training_positions']==ta.tolist() and a['target_validation_positions']==va.tolist() and a['heldout_evaluation_positions']==epos.tolist()
        assert protocol.array_sha256(ids)==a['gradient_training_row_ids_sha256'] and protocol.array_sha256(vids)==a['validation_row_ids_sha256']
        assert a['source_training_rows']==len(y0)+len(Ry) and a['source_reserved_validation_rows']==len(Cy)
        assert a['unique_target_gradient_training_rows']==len(ta) and a['target_validation_rows']==len(va) and a['target_training_repetition_factor']==4
        assert not a['heldout_eval_labels_opened'] and not result['heldout_eval_labels_opened']
        X,y,_,_=source.novel.generate(X0,y0,Q0,num,'classification','cell_heavy_tail',80,tid)
        VX,Vy,record=training.endpoint.reserve_target('canonical',X,y,Q0,Q0[ta],ay[:-nval],'classification')
        assert protocol.array_sha256(VX)==a['validation_context_X_sha256'] and protocol.array_sha256(Vy)==a['validation_context_y_sha256']
        assert record==a['validation_context'] and len(Vy)<=256
        assert source.sha(result['checkpoint_file'])==result['checkpoint_sha256']
        tensor=torch.load(result['checkpoint_file'],map_location='cpu',weights_only=False)
        digest=state_digest(tensor['state_dict']);del tensor;gc.collect()
        assert digest==result['selected_model_sha256']
        warm=json.loads((training.endpoint.oldbank.training.OUT/'checkpoints'/'perturbation'/'classification'/str(tid)/'result.json').read_text())
        assert result['initial_model_sha256']==warm['selected_model_sha256']
        assert result['selected_weights_changed']==(digest!=result['initial_model_sha256']) and result['training_loss_calls']>0
        trace=result['validation_trace'];best=result['selected_validation_score']
        assert best>=max(v['primary'] for v in trace)-1.0001e-4
        tied=[v['secondary']['equal_domain_log_loss'] for v in trace if abs(v['primary']-best)<1e-12]
        assert result['selected_validation_NLL']<=min(tied)+1.1e-6
        for v in trace:
            s=v['secondary'];np.testing.assert_allclose(v['primary'],.5*(s['source_accuracy']+s['target_accuracy']),atol=1e-12,rtol=0)
            assert v['context_rows']<=256
        checkpoints.append({'task_id':tid,'weights_changed':result['selected_weights_changed'],'training_loss_calls':result['training_loss_calls'],
            'validation_gain':best-trace[0]['primary'],'target_gradient_rows':len(ta),'target_validation_rows':len(va)})
        pfile=source.OUT/'policies'/'classification'/f'{c["fold"]}.json';gate=json.loads(pfile.read_text())
        assert c['data_id'] not in gate['training_data_ids']
        for rho in source.novel.STRENGTHS:
            path=run.OUT/'predictions'/str(tid)/f'cell_heavy_tail_{rho}.npz'
            meta=json.loads(path.with_suffix('.json').read_text());assert source.sha(path)==meta['output_sha256']
            assert meta['input_sha256']==c['input_sha256'] and meta['adaptation_sha256']==c['adaptation_sha256']
            assert meta['evaluation_positions']==epos.tolist() and meta['target_adaptation_positions']==apos.tolist()
            assert meta['checkpoint_sha256']['Warm']==warm['checkpoint_sha256'] and meta['checkpoint_sha256']['Adapted']==result['checkpoint_sha256']
            assert meta['policy_sha256']==source.sha(pfile) and meta['actual_GPU'] in range(6) and not meta['eval_labels_opened']
            X,y,Qfull,_=source.novel.generate(X0,y0,Q0,num,'classification','cell_heavy_tail',rho,tid)
            Q=Qfull[epos];views=[('canonical',X,y,Q)]+source.constructors.shared_contexts(X,y,Q,len(y),'classification',None)
            order=source.constructors.specific.importance(X,y.astype(float));columns=np.array([j for j in range(X.shape[1]) if j!=int(order[0])])
            if not len(columns):columns=np.array([0])
            for (name,A,b,T),saved in zip(views,meta['contexts']):
                V,labels,record=training.endpoint.reserve_target(name,A,b,T,Qfull[apos],ay,'classification',columns)
                assert len(labels)<=256 and name==saved['name'] and record['target_rows']==len(apos)==saved['target_rows']
                assert record['retained_candidate_positions']==saved['retained_candidate_positions']
                assert protocol.array_sha256(V)==saved['X_sha256'] and protocol.array_sha256(labels)==saved['y_sha256'] and protocol.array_sha256(T)==saved['query_X_sha256']
                np.testing.assert_array_equal(labels[-len(apos):],ay)
                if name=='canonical':mu,sigma=float(labels.mean()),max(float(labels.std()),1e-12)
                context_count+=1
            with np.load(path) as pred:
                original=source.OUT/'predictions'/'perturbation'/'classification'/str(tid)/path.name
                assert source.sha(original)==meta['original_prediction_sha256']
                with np.load(original) as z:np.testing.assert_array_equal(pred['Base'],z['Base'][epos])
                for version in ['Warm','Adapted']:
                    shared=source.fusion.predict({'task':'classification','base':pred[version+'_Native'],
                        'bank':pred[version+'_members'],'mu':mu,'sigma':sigma},gate['Shared'])
                    np.testing.assert_allclose(shared,pred[version+'_Shared'],atol=1e-12,rtol=0)
                assert source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
                with np.load(c['heldout_truth_file']) as z:truth=z['y_target'];np.testing.assert_array_equal(z['target_indices'],qid[epos])
                for method in ['Base','Warm_Native','Warm_Shared','Adapted_Native','Adapted_Shared']:
                    rows.append({'task_id':tid,'data_id':c['data_id'],'family':'cell_heavy_tail','rho':rho,'method':method,
                        **metrics(truth,pred[method],'classification',1.)})
            count+=1
    assert count==4*len(cases)
    prefix='pilot_' if pilot else ''
    df=pd.DataFrame(rows);df.to_csv(run.OUT/f'{prefix}condition_metrics.csv',index=False)
    ds=df.groupby(['method','data_id'])[['accuracy','auc']].mean().reset_index()
    table=ds.groupby('method')[['accuracy','auc']].mean().reset_index();table['datasets']=len(cases)
    table.to_csv(run.OUT/f'{prefix}family_averages.csv',index=False)
    get=lambda method:float(table.loc[table.method==method,'accuracy'].iloc[0])
    gain=get('Adapted_Shared')-get('Base')
    atomic_json(run.OUT/f'{prefix}numerical_acceptance.json',{'pilot_only':pilot,'active_endpoint':list(training.endpoint.ENDPOINT),
        'conditions':count,'datasets':len(cases),'Base':get('Base'),'Adapted_Shared':get('Adapted_Shared'),'gain':gain,
        'gain_vs_Warm_Shared_same_contexts':get('Adapted_Shared')-get('Warm_Shared'),
        'gain_vs_Adapted_Native_same_supervision':get('Adapted_Shared')-get('Adapted_Native'),
        'required_gain_strictly_greater_than':.05,'active_endpoint_pass':not pilot and count==256 and gain>.05,'all_twelve_pass':False})
    assert training.verify_frozen()==tp['frozen_registry_sha256']==ip['frozen_registry_sha256']
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(run.OUT/f'{prefix}independent_audit.json',{'status':'PASS','pilot_only':pilot,'conditions':count,
        'checkpoints':checkpoints,'checkpoint_tensor_digests_recomputed':True,'target_gradient_vs_validation_vs_evaluation_split_reconstructed':True,
        'all_eval_row_ids_disjoint_from_gradient_and_validation_labels':True,'contexts_reconstructed':context_count,
        'Warm_and_Adapted_contexts_identical':True,'frozen_Shared_fusion_replayed':True,'previous_passed_endpoints_unchanged':True,
        'Markdown_bytes':0,'script_sha256':source.sha(__file__)})
    print(table.to_string(index=False),flush=True);print('GAIN',gain,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)
