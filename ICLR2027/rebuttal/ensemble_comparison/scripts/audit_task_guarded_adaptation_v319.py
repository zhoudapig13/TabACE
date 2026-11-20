"""Reconstruct contexts, reselect adaptation-only policies, then open held-out truth."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import task_guarded_adaptation_v318 as run
from prepare_data import ROOT,atomic_json,protocol
from run_tabicl import metrics


def main(partial=False):
    source=run.source;out=run.OUT
    plan=json.loads((out/'plan.json').read_text());bankplan=json.loads((run.bank.OUT/'plan.json').read_text())
    assert source.sha(run.__file__)==plan['script_sha256'] and source.sha(run.kernel.__file__)==plan['kernel_sha256']
    assert source.sha(run.bank.__file__)==bankplan['script_sha256']
    for module,key in [(source.novel,'generator_sha256'),(source.constructors,'registry_sha256'),(source.fusion,'fusion_sha256')]:
        assert source.sha(module.__file__)==bankplan[key]
    cases=json.loads((run.bank.split.OUT/'case_manifest.json').read_text())
    rows=[];count=0;replayed=0
    for c in cases:
        track,task,tid=c['track'],c['task'],c['task_id']
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as labels:
            apos,epos,ay,aids,eids=(labels[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y','adaptation_indices','evaluation_indices'))
        assert set(apos).isdisjoint(epos) and set(aids).isdisjoint(eids)
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            X0,y0,Q0,numeric,sids,qids=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        np.testing.assert_array_equal(aids,qids[apos]);np.testing.assert_array_equal(eids,qids[epos])
        assert set(sids).isdisjoint(qids)
        checkpointdir=run.bank.training.OUT/'checkpoints'/track/task/str(tid)
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
            assert meta['adaptation_sha256']==c['adaptation_sha256']
            assert meta['evaluation_positions']==epos.tolist() and meta['adaptation_positions']==apos.tolist()
            assert not meta['final_eval_labels_opened'] and not meta['final_eval_scores_used_for_selection']
            bp=run.bank.OUT/'predictions'/track/task/str(tid)/path.name
            bm=json.loads(bp.with_suffix('.json').read_text());assert source.sha(bp)==bm['output_sha256']==meta['bank_sha256']
            assert bm['checkpoint_sha256']==checkpoint['checkpoint_sha256'] and bm['policy_sha256']==source.sha(pfile)
            assert bm['actual_GPU'] in range(6) and bm['target_labels_used_in_context']==0 and not bm['target_labels_opened']
            if rho:X,y,Q,_=source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:X,y,Q=X0,y0,Q0
            assert len(y)<=256
            assert protocol.array_sha256(X)==bm['observed_source_X_sha256']
            assert protocol.array_sha256(y)==bm['observed_source_y_sha256']
            assert protocol.array_sha256(Q)==bm['observed_query_X_sha256']
            with np.load(bp) as z,np.load(path) as pred:
                assert len(z['latent'])==len(Q) and np.isfinite(z['latent']).all()
                views=source.constructors.shared_contexts(X,y,Q,len(y),task,z['Native_finetuned'])
                assert len(views)==len(bm['contexts'])==len(z['members'])==5
                for (name,A,b,T),record in zip(views,bm['contexts']):
                    assert name==record['name'] and len(b)==record['rows'] and len(b)<=256
                    assert protocol.array_sha256(A)==record['X_sha256']
                    assert protocol.array_sha256(b)==record['y_sha256']
                    assert protocol.array_sha256(T)==record['query_sha256']
                shared=source.fusion.predict({'task':task,'base':z['Native_finetuned'],'bank':z['members'],
                    'mu':float(y.mean()),'sigma':max(float(y.std()),1e-12)},gate['Shared'])
                np.testing.assert_allclose(shared,z['Shared_finetuned'],atol=1e-12,rtol=0)
                for method,is_shared in [('Native',False),('Shared',True)]:
                    name=f'{method}_finetuned'
                    np.testing.assert_array_equal(z[name][epos],pred[name])
                    output,policy=run.select_and_predict(run.features(z,task,is_shared),z[name],ay,apos,epos,task)
                    stored=meta[f'{method}_policy']
                    assert policy['kind']==stored['kind'] and policy['feature']==stored['feature']
                    np.testing.assert_allclose(policy['score'],stored['score'],atol=1e-12,rtol=0)
                    if policy['kind']=='kernel':
                        assert policy['bandwidth_multiplier']==stored['bandwidth_multiplier'] and policy['regularization']==stored['regularization']
                        assert policy['strength']==stored['strength']
                        np.testing.assert_allclose(policy['alpha'],stored['alpha'],atol=1e-12,rtol=0)
                    assert policy['source_label_scale']==stored['source_label_scale']==float(z['source_scale'])
                    np.testing.assert_allclose(output,pred[f'{method}_adapted'],atol=1e-12,rtol=0)
                    if task=='regression':
                        assert np.max(np.abs(output-z[name][epos]))<=.25*float(z['source_scale'])+1e-9
                    replayed+=1
                original=source.OUT/'predictions'/track/task/str(tid)/path.name
                with np.load(original) as old:np.testing.assert_array_equal(old['Base'][epos],pred['Base'])
                # Final evaluation labels are opened only after both policies were replayed.
                assert source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
                with np.load(c['heldout_truth_file']) as truthfile:
                    truth=truthfile['y_target'];np.testing.assert_array_equal(truthfile['target_indices'],eids)
                for method in ['Base','Native_finetuned','Shared_finetuned','Native_adapted','Shared_adapted']:
                    assert len(pred[method])==len(truth) and np.isfinite(pred[method]).all()
                    rows.append({'track':track,'task':task,'task_id':tid,'data_id':c['data_id'],'family':family,'rho':rho,'method':method,
                        'target_adaptation_labels':len(ay) if method.endswith('_adapted') else 0,
                        **metrics(truth,pred[method],task,max(float(np.std(truth)),1e-12))})
            count+=1
    assert count>0
    if not partial:assert count==2175 and replayed==4350
    prefix='partial_' if count<2175 else ''
    df=pd.DataFrame(rows);df.to_csv(out/f'{prefix}condition_metrics.csv',index=False)
    keys=['track','task','family','method'];columns=['accuracy','auc','r2','nrmse','target_adaptation_labels']
    # Partial summaries include only datasets with all four strengths completed.
    valid=df[df.rho>0].groupby(keys+['data_id']).rho.nunique().reset_index(name='strengths')
    valid=valid[valid.strengths==4][keys+['data_id']]
    selected=pd.concat([df[df.rho==0],df[df.rho>0].merge(valid,on=keys+['data_id'])],ignore_index=True)
    ds=selected.groupby(keys+['data_id'])[columns].mean().reset_index()
    table=ds.groupby(keys)[columns].mean().reset_index()
    table=table.merge(ds.groupby(keys).size().rename('datasets').reset_index(),on=keys)
    table.to_csv(out/f'{prefix}family_averages.csv',index=False)
    gains=[]
    for (track,task,family),g in table[table.family!='clean'].groupby(['track','task','family']):
        metric='accuracy' if task=='classification' else 'r2'
        base=float(g.loc[g.method=='Base',metric].iloc[0]);shared=float(g.loc[g.method=='Shared_adapted',metric].iloc[0])
        control=float(g.loc[g.method=='Native_adapted',metric].iloc[0]);threshold=.05 if track=='perturbation' else .02
        gains.append({'track':track,'task':task,'family':family,'datasets':int(g.datasets.iloc[0]),'Base':base,'Shared_adapted':shared,
            'gain':shared-base,'gain_vs_same_label_budget_Native':shared-control,
            'required_gain_strictly_greater_than':threshold,'passed':shared-base>threshold})
    atomic_json(out/f'{prefix}numerical_acceptance.json',{'conditions':count,'family_results':gains,
        'passed_count':sum(g['passed'] for g in gains),'all_twelve_pass':len(gains)==12 and all(g['passed'] for g in gains),
        'per_task_target_supervised_adaptation':True,'not_original_Shared':True,'single_seed':0})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(out/f'{prefix}independent_audit.json',{'status':'PASS','complete':count==2175,'conditions':count,
        'contexts_reconstructed':count*5,'adaptation_policies_reselected':replayed,
        'Base_bitwise_preserved_on_same_evaluation_rows':True,'no_adaptation_row_in_evaluation':True,
        'no_adaptation_labels_in_TFM_contexts':True,'adapter_model_selection_uses_adaptation_labels_only':True,
        'maximum_GPU_index_used':5,'Markdown_bytes':0,'audit_script_sha256':source.sha(__file__)})
    print(table.to_string(index=False),flush=True);print(json.dumps(gains,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
