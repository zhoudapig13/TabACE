"""Reconstruct every expanded-pool context, replay fusion and score fixed forecasts."""
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import shared_expanded_pool_v283 as run
from prepare_data import ROOT,atomic_json,protocol
from run_tabicl import metrics


def main():
    out=run.OUT
    assert json.loads((out/'execution.json').read_text())['status']=='COMPLETE'
    plan=json.loads((out/'plan.json').read_text())
    for module,key in [(run,'script_sha256'),(run.constructors,'constructors_sha256'),(run.posterior,'posterior_sha256')]:
        assert run.source.sha(module.__file__)==plan[key]
    assert run.source.sha(run.references.OUT/'reference_manifest.json')==plan['reference_manifest_sha256']
    assert run.source.sha(run.constructors.parent.__file__)==plan['underlying_constructor_sha256']
    refs={(r['task'],r['task_id']):r for r in json.loads((run.references.OUT/'reference_manifest.json').read_text())}
    records=[];ref_records=[];count=0;contexts_replayed=0;all_calls=0
    for c in json.loads((run.source.OUT/'case_manifest.json').read_text()):
        if c['track']!='perturbation':continue
        task,tid=c['task'],c['task_id']
        assert run.source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            X0,y0,Q0,numeric,sid,qid=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        r=refs[(task,tid)];assert run.source.sha(r['reference_file'])==r['reference_sha256']
        with np.load(r['reference_file']) as z:
            TX,Ty,tids,C,Cy,cids=(z[k] for k in ('X_train_reference','y_train_reference','train_reference_indices','X_calibration','y_calibration','calibration_indices'))
        assert set(map(int,tids)).isdisjoint(map(int,np.r_[sid,qid,cids]))
        assert set(map(int,cids)).isdisjoint(map(int,np.r_[sid,qid]))
        policy_path=run.source.OUT/'policies'/task/f'{c["fold"]}.json'
        policy=json.loads(policy_path.read_text());assert c['data_id'] not in policy['training_data_ids']
        assert run.source.sha(c['truth_file'])==c['truth_sha256']
        with np.load(c['truth_file']) as z:truth=z['y_target']
        for family,rho in [('clean',0)]+[(f,s) for f in c['families'] for s in run.source.novel.STRENGTHS]:
            path=out/'predictions'/task/str(tid)/f'{family}_{rho}.npz'
            meta=json.loads(path.with_suffix('.json').read_text());assert run.source.sha(path)==meta['output_sha256']
            assert meta['original_input_sha256']==c['input_sha256'] and meta['reference_sha256']==r['reference_sha256']
            assert meta['policy_sha256']==run.source.sha(policy_path)
            assert meta['additional_training_labels']==len(Ty)
            assert not meta['reserved_calibration_labels_used'] and not meta['query_labels_opened']
            if rho:X,y,Q,_=run.source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:X,y,Q=X0,y0,Q0
            np.testing.assert_array_equal(y,y0);np.testing.assert_array_equal(Q,Q0)
            poolX,pooly=np.concatenate([X,TX]),np.concatenate([y,Ty])
            assert protocol.array_sha256(poolX)==meta['pool_X_sha256']
            assert protocol.array_sha256(pooly)==meta['pool_y_sha256']
            geometry=run.constructors.geometry(poolX,pooly,Q,C,task)
            seen=[];cseen=[]
            for group_meta in meta['groups']:
                group=group_meta['group']
                qi,ci,canonical,T,views=run.constructors.contexts(poolX,pooly,Q,C,y,task,geometry,group)
                assert qi.tolist()==group_meta['query_positions'] and ci.tolist()==group_meta['calibration_positions']
                assert canonical.tolist()==group_meta['canonical_pool_rows'] and len(canonical)==256
                if task=='classification':
                    np.testing.assert_array_equal(np.bincount(pooly[canonical]),np.bincount(y))
                assert len(views)==5
                for actual,expected in zip(views,group_meta['contexts']):
                    name,A,b,V=actual
                    assert name==expected['name'] and len(b)==expected['rows'] and len(b)<=256
                    assert protocol.array_sha256(A)==expected['X_sha256']
                    assert protocol.array_sha256(b)==expected['y_sha256']
                    assert protocol.array_sha256(V)==expected['prediction_X_sha256']
                    contexts_replayed+=1
                seen.extend(qi.tolist());cseen.extend(ci.tolist())
            assert sorted(seen)==list(range(len(Q))) and sorted(cseen)==list(range(len(C)))
            assert meta['actual_TFM_calls']==6*len(meta['groups'])<=24
            all_calls+=meta['actual_TFM_calls']
            old=run.source.OUT/'predictions/perturbation'/task/str(tid)/path.name
            assert run.source.sha(old)==meta['original_prediction_sha256']
            with np.load(path) as z,np.load(old) as previous:
                np.testing.assert_array_equal(z['Base'],previous['Base'])
                mu,sigma=float(pooly.mean()),max(float(pooly.std()),1e-12)
                np.testing.assert_allclose([z['source_mean'],z['source_scale']],[mu,sigma],rtol=0,atol=1e-12)
                raw=run.source.fusion.predict({'task':task,'base':z['Local_core'],'bank':z['members'],
                    'mu':mu,'sigma':sigma},policy['Shared'])
                rraw=run.source.fusion.predict({'task':task,'base':z['reference_core'],'bank':z['reference_members'],
                    'mu':mu,'sigma':sigma},policy['Shared'])
                np.testing.assert_allclose(raw,z['Shared_raw'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(rraw,z['reference_Shared'],atol=1e-12,rtol=0)
                if task=='classification':
                    adapted,_=run.posterior.align_probability_marginal(raw,y)
                    control,_=run.posterior.align_probability_marginal(z['Local_core'],y)
                else:adapted,control=raw,z['Local_core']
                np.testing.assert_allclose(adapted,z['Shared_adapted'],atol=1e-10,rtol=0)
                np.testing.assert_allclose(control,z['Base_same_pool'],atol=1e-10,rtol=0)
                for name in ('Base','Base_same_pool','Local_core','Shared_raw','Shared_adapted'):
                    assert np.isfinite(z[name]).all()
                    records.append({'task':task,'data_id':c['data_id'],'family':family,'rho':rho,
                        'method':name,'extra_training_labels':0 if name=='Base' else len(Ty),
                        **metrics(truth,z[name],task,max(float(np.std(truth)),1e-12))})
                for name in ('reference_core','reference_Shared'):
                    ref_records.append({'task':task,'data_id':c['data_id'],'family':family,'rho':rho,'method':name,
                        **metrics(Cy,z[name],task,max(float(np.std(Cy)),1e-12))})
            count+=1
        print('EXPANDED_POOL_AUDIT_CASE',task,tid,flush=True)
    assert count==1327
    df=pd.DataFrame(records);df.to_csv(out/'condition_metrics.csv',index=False)
    pd.DataFrame(ref_records).to_csv(out/'reserved_reference_metrics.csv',index=False)
    keys,cols=['task','family','method'],['accuracy','auc','r2','nrmse','extra_training_labels']
    ds=df.groupby(keys+['data_id'])[cols].mean().reset_index()
    table=ds.groupby(keys)[cols].mean().reset_index()
    table=table.merge(ds.groupby(keys).size().rename('datasets').reset_index(),on=keys)
    table.to_csv(out/'family_averages.csv',index=False)
    gains=[]
    for (task,family),g in table[table.family!='clean'].groupby(['task','family']):
        metric='accuracy' if task=='classification' else 'r2'
        base=float(g.loc[g.method=='Base',metric].iloc[0]);shared=float(g.loc[g.method=='Shared_adapted',metric].iloc[0])
        control=float(g.loc[g.method=='Base_same_pool',metric].iloc[0])
        gains.append({'task':task,'family':family,'metric':metric,'Base':base,'Shared':shared,
            'gain_vs_original_Base':shared-base,'gain_vs_same_pool_control':shared-control,
            'threshold':.05,'passed':shared-base>.05})
    atomic_json(out/'numerical_acceptance.json',{'family_results':gains,'all_six_pass':all(g['passed'] for g in gains)})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(out/'independent_audit.json',{'status':'PASS','conditions':count,
        'five_rule_contexts_independently_reconstructed':contexts_replayed,'TFM_calls':all_calls,
        'Base_bitwise_unchanged':True,'reference_training_calibration_and_final_query_rows_disjoint':True,
        'reserved64_calibration_labels_not_used_by_method':True,'all_TFM_contexts_at_most256_rows':True,
        'classification_canonical_source_label_counts_preserved':True,
        'five_Shared_roles_and_original_gate_replayed':True,
        'larger_supervised_pool_and_query_local_compute_explicit':True,
        'report_script_sha256':run.source.sha(__file__),'Markdown_bytes':0})
    print(table.to_string(index=False),flush=True);print(json.dumps(gains,indent=2),flush=True)


if __name__=='__main__':main()
