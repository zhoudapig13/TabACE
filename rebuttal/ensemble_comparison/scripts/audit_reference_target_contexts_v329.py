"""Audit full context provenance and cross-fit selection before reporting truth."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import reference_target_contexts_v323 as run
from prepare_data import ROOT,atomic_json,protocol
from run_tabicl import metrics


def main(pilot=False,partial=False):
    source=run.source;out=run.OUT;plan=json.loads((out/'plan.json').read_text())
    for module,key in [(run,'script_sha256'),(run.geometry,'geometry_sha256'),(run.selector,'selector_sha256'),
                       (source.novel,'generator_sha256'),(source.constructors,'registry_sha256'),(source.fusion,'fusion_sha256')]:
        assert source.sha(module.__file__)==plan[key]
    assert source.sha(run.oldbank.split.OUT/'case_manifest.json')==plan['manifest_sha256']
    assert source.sha(run.oldbank.training.OUT/'independent_training_audit.json')==plan['source_training_audit_sha256']
    allcases=json.loads((run.oldbank.split.OUT/'case_manifest.json').read_text())
    cases=[allcases[i] for i in run.PILOT_POSITIONS] if pilot else allcases
    rows=[];count=0;contexts=0;crossfit_predictions=0
    for c in cases:
        track,task,tid=c['track'],c['task'],c['task_id']
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            X0,y0,Q0,numeric,sids,qids=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        assert source.sha(c['reference_file'])==c['reference_sha256']
        with np.load(c['reference_file']) as z:RX,Ry,rids=(z[k] for k in ('X_train_reference','y_train_reference','train_reference_indices'))
        assert set(rids).isdisjoint(qids) and set(rids).isdisjoint(sids)
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        assert set(apos).isdisjoint(epos)
        folds=np.arange(len(apos))%4
        checkpointdir=run.oldbank.training.OUT/'checkpoints'/track/task/str(tid)
        checkpoint=json.loads((checkpointdir/'result.json').read_text())
        assert source.sha(checkpoint['checkpoint_file'])==checkpoint['checkpoint_sha256']
        pfile=source.OUT/'policies'/task/f'{c["fold"]}.json';gate=json.loads(pfile.read_text())
        assert c['data_id'] not in gate['training_data_ids']
        for family,rho in [('clean',0)]+[(f,r) for f in c['families'] for r in source.novel.STRENGTHS]:
            path=out/'predictions'/track/task/str(tid)/f'{family}_{rho}.npz'
            if not path.with_suffix('.json').exists():
                assert partial,(str(path),'missing condition')
                continue
            meta=json.loads(path.with_suffix('.json').read_text());assert source.sha(path)==meta['output_sha256']
            assert meta['reference_sha256']==c['reference_sha256'] and meta['adaptation_sha256']==c['adaptation_sha256']
            assert meta['input_sha256']==c['input_sha256'] and meta['source_training_checkpoint_sha256']==checkpoint['checkpoint_sha256']
            assert meta['policy_sha256']==source.sha(pfile) and meta['actual_GPU'] in range(6)
            assert meta['target_adaptation_positions']==apos.tolist() and meta['evaluation_positions']==epos.tolist()
            assert meta['adaptation_folds']==folds.tolist() and not meta['final_eval_labels_opened'] and not meta['final_eval_scores_used_for_selection']
            if rho:X,y,Q,genmeta=source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:X,y,Q,genmeta=X0,y0,Q0,{}
            for value,key in [(X,'observed_source_X_sha256'),(y,'observed_source_y_sha256'),(Q,'observed_query_X_sha256')]:
                assert protocol.array_sha256(value)==meta[key]
            poolX,pooly=np.r_[X,RX],np.r_[y,Ry]
            observed_ids=sids[np.array(genmeta['selected_source_positions'])] if genmeta.get('source_rows_resampled') else sids
            poolids=np.r_[observed_ids,rids]
            assert protocol.array_sha256(poolids)==meta['pool_row_ids_sha256'] and set(poolids).isdisjoint(qids)
            groups=run.geometry.construct(poolX,pooly,Q,task)
            original=source.OUT/'predictions'/track/task/str(tid)/path.name
            oldpath=run.oldbank.OUT/'predictions'/track/task/str(tid)/path.name
            assert source.sha(original)==meta['original_prediction_sha256'] and source.sha(oldpath)==meta['previous_bank_sha256']
            with np.load(path) as pred,np.load(original) as original_np,np.load(oldpath) as old:
                np.testing.assert_array_equal(pred['Base'],original_np['Base'][epos])
                for name in ['Native','Shared']:np.testing.assert_array_equal(pred[f'{name}_finetuned'],old[f'{name}_finetuned'][epos])
                coverage={'reference':np.zeros(len(Q),dtype=int),'crossfit':np.zeros(len(apos),dtype=int),'final':np.zeros(len(epos),dtype=int)}
                records=meta['predicted_context_sets']
                expected_sets=[]
                for group in groups:
                    qi,rr=group['query_positions'],group['pool_positions']
                    CX,Cy=poolX[rr],pooly[rr]
                    views=source.constructors.shared_contexts(CX,Cy,Q[qi],len(Cy),task,None)
                    columns=None
                    if task=='classification':
                        order=source.constructors.specific.importance(CX,Cy.astype(float))
                        columns=np.array([j for j in range(CX.shape[1]) if j!=int(order[0])])
                        if not len(columns):columns=np.array([0])
                    def check_set(mode,fold,train_positions,train_labels,wanted,output_positions):
                        nonlocal contexts,crossfit_predictions
                        if not len(wanted):return
                        matches=[r for r in records if r['mode']==mode and r['fold']==fold and r['group']==group['group']]
                        assert len(matches)==1
                        record=matches[0];expected_sets.append((mode,fold,group['group']))
                        for value,key in [(rr,'pool_positions'),(wanted,'query_positions'),(output_positions,'output_positions'),(train_positions,'target_training_positions')]:
                            assert value.tolist()==record[key]
                        assert set(train_positions).isdisjoint(wanted) and set(qids[train_positions]).isdisjoint(qids[wanted])
                        local={int(v):i for i,v in enumerate(qi)};take=np.array([local[int(v)] for v in wanted],dtype=int)
                        mu=sigma=None
                        assert len(record['contexts'])==6
                        for (name,A,b,Tall),saved in zip([('canonical',CX,Cy,Q[qi])]+views,record['contexts']):
                            T=Tall[take]
                            V,labels,context=run.reserve_target(name,A,b,T,Q[train_positions],train_labels,task,columns)
                            assert len(labels)<=256 and name==saved['name']
                            assert context['retained_candidate_positions']==saved['retained_candidate_positions']
                            assert context['source_rows']==saved['source_rows'] and context['target_rows']==saved['target_rows']==len(train_labels)
                            for v,k in [(V,'X_sha256'),(labels,'y_sha256'),(T,'query_X_sha256')]:assert protocol.array_sha256(v)==saved[k]
                            if len(train_labels):np.testing.assert_array_equal(labels[-len(train_labels):],train_labels)
                            if name=='canonical':mu,sigma=float(labels.mean()),max(float(labels.std()),1e-12)
                            contexts+=1
                        np.testing.assert_allclose([mu,sigma],[record['source_mean'],record['source_scale']],atol=1e-12,rtol=0)
                        if mode=='reference':native,members,shared=pred['reference_native_full'],pred['reference_members_full'],pred['reference_shared_full']
                        elif mode=='final':native,members,shared=pred['Native_target_context'],pred['final_members'],pred['Shared_target_context']
                        else:
                            native,members,shared=pred['cv_native'],pred['cv_members'],pred['cv_shared']
                            crossfit_predictions+=len(wanted)
                        replay=source.fusion.predict({'task':task,'base':native[output_positions],
                            'bank':members[:,output_positions],'mu':mu,'sigma':sigma},gate['Shared'])
                        np.testing.assert_allclose(replay,shared[output_positions],atol=1e-12,rtol=0)
                        coverage[mode][output_positions]+=1
                    check_set('reference',None,np.array([],dtype=int),ay[:0],qi,qi)
                    for fold in range(4):
                        val=np.flatnonzero((folds==fold)&np.isin(apos,qi));train=np.flatnonzero(folds!=fold)
                        check_set('crossfit',fold,apos[train],ay[train],apos[val],val)
                    final=np.flatnonzero(np.isin(epos,qi));check_set('final',None,apos,ay,epos[final],final)
                assert len(expected_sets)==len(records)
                for value in coverage.values():np.testing.assert_array_equal(value,np.ones_like(value))
                for name in ['Native','Shared']:
                    policy=run.selector.select(ay,np.stack([original_np['Base'][apos],old[f'{name}_finetuned'][apos],pred[f'cv_{name.lower()}']]),task,max(float(y.std()),1e-12))
                    saved=meta[f'{name}_policy'];assert policy['weights']==saved['weights']
                    np.testing.assert_allclose(policy['score'],saved['score'],atol=1e-12,rtol=0)
                    output=run.selector.apply(policy,np.stack([original_np['Base'][epos],old[f'{name}_finetuned'][epos],pred[f'{name}_target_context']]),task)
                    np.testing.assert_allclose(output,pred[f'{name}_adapted'],atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(pred[f'{name}_reference_only'],pred[f'reference_{name.lower()}_full'][epos])
                assert source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
                with np.load(c['heldout_truth_file']) as truth_np:
                    truth=truth_np['y_target'];np.testing.assert_array_equal(truth_np['target_indices'],qids[epos])
                for name in ['Base','Native_finetuned','Shared_finetuned','Native_reference_only','Shared_reference_only','Native_target_context','Shared_target_context','Native_adapted','Shared_adapted']:
                    assert np.isfinite(pred[name]).all() and len(pred[name])==len(truth)
                    rows.append({'track':track,'task':task,'task_id':tid,'data_id':c['data_id'],'family':family,'rho':rho,'method':name,
                        'extra_clean_source_labels':0 if name=='Base' else len(Ry),
                        'target_adaptation_labels_available':len(ay) if name.endswith('_adapted') or name.endswith('_target_context') else 0,
                        **metrics(truth,pred[name],task,max(float(truth.std()),1e-12))})
            count+=1
        print('REFERENCE_TARGET_AUDIT_CASE',track,task,tid,flush=True)
    expected=plan['pilot_conditions'] if pilot else plan['conditions']
    if not partial:assert count==expected
    assert count>0
    prefix=('pilot_' if pilot else '')+('partial_' if count<expected else '')
    df=pd.DataFrame(rows);df.to_csv(out/f'{prefix}condition_metrics.csv',index=False)
    keys=['track','task','family','method'];columns=['accuracy','auc','r2','nrmse','extra_clean_source_labels','target_adaptation_labels_available']
    valid=df[df.rho>0].groupby(keys+['data_id']).rho.nunique().reset_index(name='strengths')
    valid=valid[valid.strengths==4][keys+['data_id']]
    df=pd.concat([df[df.rho==0],df[df.rho>0].merge(valid,on=keys+['data_id'])],ignore_index=True)
    ds=df.groupby(keys+['data_id'])[columns].mean().reset_index()
    table=ds.groupby(keys)[columns].mean().reset_index()
    table=table.merge(ds.groupby(keys).size().rename('datasets').reset_index(),on=keys)
    table.to_csv(out/f'{prefix}family_averages.csv',index=False)
    gains=[]
    for (track,task,family),g in table[table.family!='clean'].groupby(['track','task','family']):
        metric='accuracy' if task=='classification' else 'r2';value=lambda method:float(g.loc[g.method==method,metric].iloc[0])
        baseline=value('Base');adapted=value('Shared_adapted');threshold=.05 if track=='perturbation' else .02
        gains.append({'track':track,'task':task,'family':family,'datasets':int(g.datasets.iloc[0]),'Base':baseline,'Shared_adapted':adapted,
            'gain':adapted-baseline,'gain_vs_same_budget_Native':adapted-value('Native_adapted'),
            'reference_only_gain':value('Shared_reference_only')-baseline,'raw_target_context_gain':value('Shared_target_context')-baseline,
            'required_gain_strictly_greater_than':threshold,'passed':adapted-baseline>threshold})
    atomic_json(out/f'{prefix}numerical_acceptance.json',{'pilot_only':pilot,'conditions':count,'family_results':gains,
        'passed_count':sum(g['passed'] for g in gains),'all_twelve_pass':not pilot and count==2175 and len(gains)==12 and all(g['passed'] for g in gains),
        'extra_clean_source_and_target_subset_supervision':True,'not_original_zero_shot_Shared':True,'seed':0})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(out/f'{prefix}independent_audit.json',{'status':'PASS','complete':count==expected,'pilot_only':pilot,'conditions':count,
        'contexts_reconstructed':contexts,'crossfit_validation_predictions_covered':crossfit_predictions,
        'adaptation_and_evaluation_row_ids_disjoint':True,'crossfit_validation_labels_absent_from_their_contexts':True,
        'reference_source_pool_ids_disjoint_from_entire_original_query':True,'Base_bitwise_unchanged_on_fixed_evaluation_rows':True,
        'convex_deployment_policies_reselected_on_crossfit_predictions':True,'context_cap':256,'allowed_GPUs':[0,1,2,3,4,5],
        'neural_predictions_all_recomputed_in_this_CPU_audit':False,'Markdown_bytes':0,'audit_script_sha256':source.sha(__file__)})
    print(table.to_string(index=False),flush=True);print(json.dumps(gains,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.pilot,a.partial)
