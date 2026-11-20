"""Audit original-registry fine-tuned Shared and report all six mechanisms."""
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import shared_finetuned_inference_v309 as run
from prepare_data import ROOT,RUN,atomic_json,protocol
from run_tabicl import metrics


def main():
    out=run.OUT;assert json.loads((out/'execution.json').read_text())['status']=='COMPLETE'
    plan=json.loads((out/'plan.json').read_text())
    for module,key in [(run,'script_sha256'),(run.source.novel,'generator_sha256'),
                       (run.source.constructors,'registry_sha256'),(run.source.fusion,'fusion_sha256')]:
        assert run.source.sha(module.__file__)==plan[key]
    assert run.source.sha(run.training.OUT/'independent_training_audit.json')==plan['training_audit_sha256']
    assert run.source.sha(run.training.OUT/'case_manifest.json')==plan['training_manifest_sha256']
    cases=json.loads((run.training.OUT/'case_manifest.json').read_text())
    shifts={(c['task'],c['task_id']):c for c in json.loads((RUN/'results/shift_target_adaptation_manifest_v285/case_manifest.json').read_text())}
    rows=[];count=0;contexts_checked=0
    for c in cases:
        track,task,tid=c['track'],c['task'],c['task_id']
        assert run.source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:X0,y0,Q0,numeric,qids=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','target_indices'))
        if track=='shift':
            s=shifts[(task,tid)];assert run.source.sha(s['adaptation_input_file'])==s['adaptation_input_sha256']
            with np.load(s['adaptation_input_file']) as z:apos,epos,ay=(z[k] for k in ('target_adaptation_positions','evaluation_positions','target_adaptation_y'))
            truthfile,truthsha=s['evaluation_truth_file'],s['evaluation_truth_sha256']
            assert set(apos).isdisjoint(epos)
        else:
            apos,epos,ay=np.array([],dtype=int),np.arange(len(Q0)),y0[:0]
            truthfile,truthsha=c['truth_file'],c['truth_sha256']
        assert run.source.sha(truthfile)==truthsha
        with np.load(truthfile) as z:truth=z['y_target'];np.testing.assert_array_equal(z['target_indices'],qids[epos])
        checkpoint_dir=run.training.OUT/'checkpoints'/track/task/str(tid)
        checkpoint=json.loads((checkpoint_dir/'result.json').read_text())
        training=json.loads((checkpoint_dir/'input_audit.json').read_text())
        assert run.source.sha(checkpoint['checkpoint_file'])==checkpoint['checkpoint_sha256']
        pfile=run.source.OUT/'policies'/task/f'{c["fold"]}.json';policy=json.loads(pfile.read_text())
        assert c['data_id'] not in policy['training_data_ids']
        for family,rho in [('clean',0)]+[(f,r) for f in c['families'] for r in run.source.novel.STRENGTHS]:
            path=out/'predictions'/track/task/str(tid)/f'{family}_{rho}.npz'
            meta=json.loads(path.with_suffix('.json').read_text());assert run.source.sha(path)==meta['output_sha256']
            assert meta['checkpoint_sha256']==checkpoint['checkpoint_sha256']
            assert meta['checkpoint_weights_changed']==checkpoint['selected_weights_changed']
            assert meta['original_input_sha256']==c['input_sha256'] and meta['policy_sha256']==run.source.sha(pfile)
            assert meta['target_adaptation_positions']==apos.tolist() and meta['evaluation_positions_in_original_query']==epos.tolist()
            assert meta['target_labels_used_in_context']==len(ay) and not meta['final_eval_labels_opened']
            assert meta['source_supervised_finetuning'] and meta['actual_TFM_calls']<=7
            if rho:X,y,Qfull,_=run.source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:X,y,Qfull=X0,y0,Q0
            Q,AX=Qfull[epos],Qfull[apos]
            coreX,corey,cm=run.reserve_target('canonical',X,y,Q,AX,ay,task)
            assert len(corey)<=256
            assert protocol.array_sha256(coreX)==meta['core_context']['X_sha256']
            assert protocol.array_sha256(corey)==meta['core_context']['y_sha256']
            if len(ay):np.testing.assert_array_equal(corey[-len(ay):],ay)
            old=run.source.OUT/'predictions'/track/task/str(tid)/path.name
            assert run.source.sha(old)==meta['original_prediction_sha256']
            with np.load(path) as z,np.load(old) as original:
                np.testing.assert_array_equal(z['Base'],original['Base'][epos])
                views=run.source.constructors.shared_contexts(X,y,Q,len(y),task,z['Base_finetuned_same_target_labels'])
                assert len(views)==len(meta['contexts'])==z['members'].shape[0]==5
                columns=None
                if task=='classification':
                    order=run.source.constructors.specific.importance(X,y.astype(float))
                    columns=np.asarray([j for j in range(X.shape[1]) if j!=int(order[0])])
                    if not len(columns):columns=np.asarray([0])
                for (name,A,b,T),expected in zip(views,meta['contexts']):
                    V,labels,record=run.reserve_target(name,A,b,T,AX,ay,task,columns)
                    assert name==expected['name'] and len(labels)<=256
                    assert record['retained_candidate_positions']==expected['retained_candidate_positions']
                    assert protocol.array_sha256(V)==expected['X_sha256']
                    assert protocol.array_sha256(labels)==expected['y_sha256']
                    assert protocol.array_sha256(T)==expected['prediction_X_sha256']
                    if len(ay):np.testing.assert_array_equal(labels[-len(ay):],ay)
                    contexts_checked+=1
                mu,sigma=float(corey.mean()),max(float(corey.std()),1e-12)
                np.testing.assert_allclose([z['source_mean'],z['source_scale']],[mu,sigma],atol=1e-12,rtol=0)
                replay=run.source.fusion.predict({'task':task,'base':z['Base_finetuned_same_target_labels'],
                    'bank':z['members'],'mu':mu,'sigma':sigma},policy['Shared'])
                np.testing.assert_allclose(replay,z['Shared_finetuned'],atol=1e-12,rtol=0)
                mean=np.concatenate([z['Base_finetuned_same_target_labels'][None],z['members']],axis=0).mean(0)
                np.testing.assert_allclose(mean,z['Mean6_finetuned'],atol=1e-12,rtol=0)
                for method in ('Base','Base_finetuned','Base_finetuned_same_target_labels','Shared_finetuned','Mean6_finetuned'):
                    assert np.isfinite(z[method]).all() and len(z[method])==len(truth)
                    rows.append({'track':track,'task':task,'data_id':c['data_id'],'family':family,'rho':rho,'method':method,
                        'extra_source_labels':0 if method=='Base' else training['extra_labeled_source_references'],
                        'target_adaptation_labels':len(ay) if method in ('Base_finetuned_same_target_labels','Shared_finetuned','Mean6_finetuned') else 0,
                        'checkpoint_weights_changed':False if method=='Base' else checkpoint['selected_weights_changed'],
                        **metrics(truth,z[method],task,max(float(np.std(truth)),1e-12))})
            count+=1
        print('FINETUNED_SHARED_AUDIT_CASE',track,task,tid,flush=True)
    assert count==2175 and contexts_checked==5*2175
    df=pd.DataFrame(rows);df.to_csv(out/'condition_metrics.csv',index=False)
    keys=['track','task','family','method'];cols=['accuracy','auc','r2','nrmse','extra_source_labels','target_adaptation_labels','checkpoint_weights_changed']
    ds=df.groupby(keys+['data_id'])[cols].mean().reset_index()
    table=ds.groupby(keys)[cols].mean().reset_index();table=table.merge(ds.groupby(keys).size().rename('datasets').reset_index(),on=keys)
    table.to_csv(out/'family_averages.csv',index=False)
    selected=table[(table.family!='clean')&table.method.isin(['Base','Shared_finetuned'])]
    selected.to_csv(out/'base_shared_all_metrics.csv',index=False)
    gains=[]
    for (track,task,family),g in table[table.family!='clean'].groupby(['track','task','family']):
        metric='accuracy' if task=='classification' else 'r2'
        base=float(g.loc[g.method=='Base',metric].iloc[0]);shared=float(g.loc[g.method=='Shared_finetuned',metric].iloc[0])
        control=float(g.loc[g.method=='Base_finetuned_same_target_labels',metric].iloc[0])
        threshold=.05 if track=='perturbation' else .02
        gains.append({'track':track,'task':task,'family':family,'Base':base,'Shared':shared,'gain':shared-base,
            'gain_vs_finetuned_same_target_labels_Native1':shared-control,'required_gain_strictly_greater_than':threshold,
            'passed':shared-base>threshold})
    assert len(gains)==12
    atomic_json(out/'numerical_acceptance.json',{'family_results':gains,'all_twelve_pass':all(g['passed'] for g in gains),
        'passed_count':sum(g['passed'] for g in gains),'source_supervised_parameter_finetuning':True,
        'shift_target_adaptation_rows_excluded_from_final_eval':True,'single_seed':0,'context_cap':256})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(out/'independent_audit.json',{'status':'PASS','conditions':2175,'contexts_reconstructed':contexts_checked,
        'Base_bitwise_unchanged_on_same_held_out_eval_rows':True,'five_original_Shared_roles_reconstructed':True,
        'original_gate_fusion_replayed':True,'all_contexts_at_most256_rows':True,
        'source_supervised_training_and_target_adaptation_explicit':True,
        'new_reference_pool_not_used_for_direct_query_local_sampling':True,
        'training_checkpoint_audit_completed':True,'report_script_sha256':run.source.sha(__file__),'Markdown_bytes':0})
    print(table.to_string(index=False),flush=True);print(json.dumps(gains,indent=2),flush=True)


if __name__=='__main__':main()
