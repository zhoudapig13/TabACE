"""Reconstruct every supervised context and cross-fit assignment, then score."""
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import torch
from tabicl import TabICLClassifier
import interaction_sampling_context_adaptation_v455 as run
from prepare_data import RUN,ROOT,atomic_json,protocol
from run_tabicl import Predictor,metrics
from serial_endpoint_state_v371 import verify_registry


def main():
    source=run.source
    plan=json.loads((run.OUT/'plan.json').read_text())
    for module,key in [(run,'script_sha256'),(run.geometry,'geometry_sha256'),(run.gate,'gate_sha256'),
                       (source.constructors,'constructor_sha256'),(source.fusion,'fusion_sha256'),(source.novel,'generator_sha256')]:
        assert source.sha(module.__file__)==plan[key]
    assert source.sha(run.split64.OUT/'plan.json')==plan['split_plan_sha256']
    torch.set_num_threads(1);torch.manual_seed(0)
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) in range(6)
    cases=run.split64.endpoint_cases();assert len(cases)==35
    replay_ids={cases[i]['task_id'] for i in (0,len(cases)//2,len(cases)-1)}
    rows=[];count=0;context_count=0;replays=[]
    for c in cases:
        tid=c['task_id']
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            X0,y0,Q0,num,sids,qids=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        assert source.sha(c['reference_file'])==c['reference_sha256']
        with np.load(c['reference_file']) as z:RX,Ry,rids=(z[k] for k in ('X_train_reference','y_train_reference','train_reference_indices'))
        assert set(rids).isdisjoint(np.r_[sids,qids])
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:
            a,e,ay,aids,eids=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y','adaptation_indices','evaluation_indices'))
        assert len(a)==len(e)==64 and set(a).isdisjoint(e) and set(aids).isdisjoint(eids)
        np.testing.assert_array_equal(aids,qids[a]);np.testing.assert_array_equal(eids,qids[e])
        folds=np.arange(64)%4
        cpdir=run.oldbank.training.OUT/'checkpoints'/'shift'/'classification'/str(tid)
        cp=json.loads((cpdir/'result.json').read_text())
        ca=json.loads((cpdir/'input_audit.json').read_text())
        assert not cp['target_adaptation_labels_used_for_weights'] and ca['source_train_validation_disjoint_from_entire_original_query']
        assert source.sha(cp['checkpoint_file'])==cp['checkpoint_sha256']
        pfile=source.OUT/'policies'/'classification'/f'{c["fold"]}.json'
        snapshot=json.loads(pfile.read_text());assert c['data_id'] not in snapshot['training_data_ids']
        replayed=False
        for rho in source.novel.STRENGTHS:
            path=run.OUT/'predictions'/str(tid)/f'interaction_sampling_{rho}.npz'
            meta=json.loads(path.with_suffix('.json').read_text());assert source.sha(path)==meta['output_sha256']
            assert meta['original_input_sha256']==c['input_sha256'] and meta['reference_sha256']==c['reference_sha256']
            assert meta['adaptation_sha256']==c['adaptation_sha256'] and not meta['evaluation_labels_opened']
            assert meta['checkpoint_sha256']==cp['checkpoint_sha256'] and meta['policy_sha256']==source.sha(pfile)
            assert meta['adaptation_positions']==a.tolist() and meta['evaluation_positions']==e.tolist() and meta['actual_GPU'] in range(6)
            X,y,Q,generator=source.novel.generate(X0,y0,Q0,num,'classification','interaction_sampling',rho,tid)
            poolX,pooly=np.r_[X,RX],np.r_[y,Ry]
            observed=sids[np.asarray(generator['selected_source_positions'])] if generator.get('source_rows_resampled') else sids
            poolids=np.r_[observed,rids]
            assert set(poolids).isdisjoint(qids)
            for value,key in [(poolX,'pool_X_sha256'),(pooly,'pool_y_sha256'),(poolids,'pool_row_ids_sha256')]:assert protocol.array_sha256(value)==meta[key]
            groups=run.geometry.construct(poolX,pooly,Q,'classification')
            original=source.OUT/'predictions'/'shift'/'classification'/str(tid)/path.name
            assert source.sha(original)==meta['original_prediction_sha256']
            with np.load(path) as pred,np.load(original) as old:
                np.testing.assert_array_equal(pred['Base'],old['Base'][e])
                coverage={'crossfit':np.zeros(64,dtype=int),'final':np.zeros(64,dtype=int)}
                records=meta['context_records'];expected_records=[]
                for group in groups:
                    qi,rr=group['query_positions'],group['pool_positions']
                    CX,Cy=poolX[rr],pooly[rr]
                    views=source.constructors.shared_contexts(CX,Cy,Q[qi],len(Cy),'classification',None)
                    order=source.constructors.specific.importance(CX,Cy.astype(float))
                    columns=np.asarray([j for j in range(CX.shape[1]) if j!=int(order[0])])
                    if not len(columns):columns=np.array([0])
                    def check(mode,fold,train,wanted,output):
                        nonlocal context_count,replayed
                        if not len(wanted):return
                        matched=[r for r in records if r['group']==group['group'] and r['mode']==mode and r['fold']==fold]
                        assert len(matched)==1
                        record=matched[0];expected_records.append((mode,fold,group['group']))
                        train_positions=a[train]
                        for v,key in [(rr,'pool_positions'),(train_positions,'target_training_positions'),(wanted,'query_positions'),(output,'output_positions')]:assert v.tolist()==record[key]
                        assert set(train_positions).isdisjoint(wanted) and set(qids[train_positions]).isdisjoint(qids[wanted])
                        assert len(train)==(48 if mode=='crossfit' else 64)
                        local={int(v):i for i,v in enumerate(qi)}
                        take=np.asarray([local[int(v)] for v in wanted],dtype=int)
                        assert len(record['contexts'])==6
                        mu=sigma=None
                        native=pred['Native_cv_probe'] if mode=='crossfit' else pred['Native_context_raw']
                        bank=pred['cv_members'] if mode=='crossfit' else pred['members']
                        shared=pred['Shared_cv_probe'] if mode=='crossfit' else pred['Shared_context_raw']
                        for (name,A,b,Tall),saved in zip([('canonical',CX,Cy,Q[qi])]+views,record['contexts']):
                            T=Tall[take]
                            V,labels,info=run.reserve_target(name,A,b,T,Q[train_positions],ay[train],'classification',columns)
                            assert len(labels)<=256 and name==saved['name']
                            for key in ('source_rows','target_rows','retained_candidate_positions'):assert info[key]==saved[key]
                            np.testing.assert_array_equal(labels[-len(train):],ay[train])
                            for v,key in [(V,'X_sha256'),(labels,'y_sha256'),(T,'query_X_sha256')]:assert protocol.array_sha256(v)==saved[key]
                            if name=='canonical':
                                mu,sigma=float(labels.mean()),max(float(labels.std()),1e-12)
                                if tid in replay_ids and rho==20 and not replayed:
                                    predictor=Predictor('classification','cuda:0')
                                    predictor.models[1]=TabICLClassifier(n_estimators=1,model_path=cp['checkpoint_file'],allow_auto_download=False,
                                        device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
                                    actual=predictor(V,labels,T,members=1,random_state=0,n_classes=len(np.unique(y0)))[0]
                                    np.testing.assert_allclose(actual,native[output],atol=2e-6,rtol=2e-6)
                                    replays.append({'task_id':tid,'rho':rho,'mode':mode,'max_abs_difference':float(np.max(np.abs(actual-native[output])))})
                                    del predictor;torch.cuda.empty_cache();replayed=True
                            context_count+=1
                        np.testing.assert_allclose([mu,sigma],[record['source_mean'],record['source_scale']],atol=1e-12,rtol=0)
                        expected=source.fusion.predict({'task':'classification','base':native[output],'bank':bank[:,output],'mu':mu,'sigma':sigma},snapshot['Shared'])
                        np.testing.assert_allclose(expected,shared[output],atol=1e-12,rtol=0)
                        coverage[mode][output]+=1
                    for fold in range(4):
                        out=np.flatnonzero((folds==fold)&np.isin(a,qi))
                        check('crossfit',fold,np.flatnonzero(folds!=fold),a[out],out)
                    out=np.flatnonzero(np.isin(e,qi))
                    check('final',None,np.arange(64),e[out],out)
                assert len(expected_records)==len(records)
                for value in coverage.values():np.testing.assert_array_equal(value,np.ones(64,dtype=int))
                for name in ('Native','Shared'):
                    policy=run.gate.select(ay,old['Base'][a],pred[name+'_cv_probe'])
                    saved=meta[name+'_outer_gate']
                    assert policy['beta']==saved['beta']
                    np.testing.assert_allclose([policy['accuracy'],policy['nll']],[saved['accuracy'],saved['nll']],atol=1e-12,rtol=0)
                    expected=run.gate.apply(policy,old['Base'][e],pred[name+'_context_raw'])
                    np.testing.assert_allclose(expected,pred[name+'_probe'],atol=1e-12,rtol=0)
                assert source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
                with np.load(c['heldout_truth_file']) as z:truth=z['y_target'];np.testing.assert_array_equal(z['target_indices'],qids[e])
                for name in ('Base','Native_probe','Shared_probe','Native_context_raw','Shared_context_raw'):
                    rows.append({'data_id':c['data_id'],'task_id':tid,'rho':rho,'method':name,**metrics(truth,pred[name],'classification',1.)})
            count+=1
        print('CONTEXT_ADAPTATION_AUDIT_CASE',tid,flush=True)
    assert count==140 and len(replays)==3
    frame=pd.DataFrame(rows);frame.to_csv(run.OUT/'condition_metrics.csv',index=False)
    table=frame.groupby(['data_id','method'])[['accuracy','auc']].mean().groupby('method').mean().reset_index();table['datasets']=35
    table.to_csv(run.OUT/'family_averages.csv',index=False)
    get=lambda name:float(table.loc[table.method==name,'accuracy'].iloc[0])
    gain=get('Shared_probe')-get('Base')
    atomic_json(run.OUT/'numerical_acceptance.json',{'pilot_only':False,'active_endpoint':['shift','classification','interaction_sampling'],
        'conditions':140,'datasets':35,'Base':get('Base'),'Shared_supervised_probe':get('Shared_probe'),'gain':gain,
        'gain_vs_Native_same_supervision':get('Shared_probe')-get('Native_probe'),
        'required_gain_strictly_greater_than':.02,'active_endpoint_pass':gain>.02,'all_twelve_pass':False,
        'extra_source_and64_test_subset_supervision_explicit':True})
    verify_registry(RUN/'results/frozen_endpoints_v391/registry.json')
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(run.OUT/'independent_audit.json',{'status':'PASS','pilot_only':False,'conditions':140,'contexts_reconstructed':context_count,
        'all_target_context_labels_crossfit48_or_final64':True,'evaluation_and_validation_own_labels_absent_from_contexts':True,
        'all_query_outputs_covered_exactly_once':True,'source_pool_disjoint_from_entire_original_query':True,
        'all_contexts_at_most256_rows':True,'Base_bitwise_original':True,'original_fusion_and_calibration_gates_reconstructed':True,
        'neural_prediction_spot_replays':replays,'not_all_neural_predictions_recomputed':True,'Markdown_bytes':0,'script_sha256':source.sha(__file__)})
    print(table.to_string(index=False),flush=True);print('GAIN',gain,flush=True)


if __name__=='__main__':
    main()
