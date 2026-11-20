"""Reconstruct supervised target contexts and score only disjoint held-out rows."""
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import shared_shift_target_adaptation_v292 as run
from prepare_data import ROOT,RUN,atomic_json,protocol
from run_tabicl import metrics


def main():
    out=run.OUT
    assert json.loads((out/'execution.json').read_text())['status']=='COMPLETE'
    plan=json.loads((out/'plan.json').read_text())
    for module,key in [(run,'script_sha256'),(run.constructor,'target_mix_sha256'),
                       (run.constructor.parent,'cluster_geometry_sha256'),
                       (run.constructor.parent.parent,'underlying_constructor_sha256')]:
        assert run.source.sha(module.__file__)==plan[key]
    assert run.source.sha(run.references.OUT/'case_manifest.json')==plan['case_manifest_sha256']
    rows=[];count=0;reconstructed=0
    for c in json.loads((run.references.OUT/'case_manifest.json').read_text()):
        task,tid=c['task'],c['task_id']
        assert run.source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            X0,y0,Q0,numeric,sid,qid=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        assert run.source.sha(c['adaptation_input_file'])==c['adaptation_input_sha256']
        with np.load(c['adaptation_input_file']) as z:
            TX,Ty,tids,C,cids,apos,ay,epos,aids,eids=(z[k] for k in
                ('X_train_reference','y_train_reference','train_reference_indices','X_calibration','calibration_indices',
                 'target_adaptation_positions','target_adaptation_y','evaluation_positions','target_adaptation_indices','evaluation_indices'))
        assert set(map(int,aids)).isdisjoint(map(int,eids))
        assert set(map(int,tids)).isdisjoint(map(int,np.r_[sid,qid,cids]))
        np.testing.assert_array_equal(qid[apos],aids);np.testing.assert_array_equal(qid[epos],eids)
        n=min(64,max(16,len(Q0)//4));order=np.random.default_rng(0).permutation(len(Q0))
        np.testing.assert_array_equal(apos,order[:n]);np.testing.assert_array_equal(epos,order[n:])
        assert run.source.sha(c['evaluation_truth_file'])==c['evaluation_truth_sha256']
        with np.load(c['evaluation_truth_file']) as z:truth=z['y_target'];np.testing.assert_array_equal(z['target_indices'],eids)
        assert run.source.sha(c['truth_file'])==c['truth_sha256']
        with np.load(c['truth_file']) as z:
            np.testing.assert_array_equal(z['y_target'][apos],ay)
            np.testing.assert_array_equal(z['y_target'][epos],truth)
        pfile=run.source.OUT/'policies'/task/f'{c["fold"]}.json'
        policy=json.loads(pfile.read_text());assert c['data_id'] not in policy['training_data_ids']
        for family,rho in [('clean',0)]+[(f,s) for f in c['families'] for s in run.source.novel.STRENGTHS]:
            path=out/'predictions'/task/str(tid)/f'{family}_{rho}.npz'
            m=json.loads(path.with_suffix('.json').read_text());assert run.source.sha(path)==m['output_sha256']
            assert m['original_input_sha256']==c['input_sha256'] and m['adaptation_input_sha256']==c['adaptation_input_sha256']
            assert m['policy_sha256']==run.source.sha(pfile)
            assert not m['source_calibration_labels_used'] and not m['final_eval_labels_opened']
            assert m['external_source_labels']==len(Ty) and m['original_test_labels_used_for_adaptation']==len(ay)
            assert m['held_out_evaluation_rows']==len(epos)
            if rho:X,y,Qfull,_=run.source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:X,y,Qfull=X0,y0,Q0
            Q,AX=Qfull[epos],Qfull[apos]
            assert protocol.array_sha256(AX)==m['target_adaptation_X_sha256']
            assert protocol.array_sha256(ay)==m['target_adaptation_y_sha256']
            poolX,pooly=np.concatenate([X,TX]),np.concatenate([y,Ty])
            assert protocol.array_sha256(poolX)==m['pool_X_sha256'] and protocol.array_sha256(pooly)==m['pool_y_sha256']
            quota=np.r_[y,ay,np.unique(pooly)] if task=='classification' else y
            geometry=run.constructor.parent.geometry(poolX,pooly,Q,C,task)
            seen=[];cseen=[]
            for gm in m['groups']:
                qi,ci,canonical,T,views=run.constructor.parent.contexts(poolX,pooly,Q,C,quota,task,geometry,gm['group'])
                assert qi.tolist()==gm['evaluation_positions'] and ci.tolist()==gm['calibration_positions']
                assert canonical.tolist()==gm['canonical_pool_rows']
                A,b,cm=run.constructor.mix('canonical',poolX[canonical],pooly[canonical],T,AX,ay,task,quota)
                assert protocol.array_sha256(A)==gm['canonical_context']['X_sha256']
                assert protocol.array_sha256(b)==gm['canonical_context']['y_sha256']
                assert len(b)<=256;np.testing.assert_array_equal(b[-len(ay):],ay)
                for (name,V,labels,S),expected in zip(views,gm['contexts']):
                    A,b,record=run.constructor.mix(name,V,labels,S,AX,ay,task,quota,geometry.get('columns'))
                    assert name==expected['name'] and len(b)<=256
                    assert record['retained_source_positions_in_candidate']==expected['retained_source_positions_in_candidate']
                    assert record['labeled_target_adaptation_rows']==len(ay)
                    assert protocol.array_sha256(A)==expected['X_sha256']
                    assert protocol.array_sha256(b)==expected['y_sha256']
                    assert protocol.array_sha256(S)==expected['prediction_X_sha256']
                    np.testing.assert_array_equal(b[-len(ay):],ay)
                    reconstructed+=1
                seen.extend(qi.tolist());cseen.extend(ci.tolist())
            assert sorted(seen)==list(range(len(Q))) and sorted(cseen)==list(range(len(C)))
            assert m['actual_TFM_calls']==1+6*len(m['groups'])<=25
            original=run.source.OUT/'predictions/shift'/task/str(tid)/path.name
            assert run.source.sha(original)==m['original_prediction_sha256']
            with np.load(path) as z,np.load(original) as old:
                np.testing.assert_array_equal(z['Base'],old['Base'][epos])
                mu,sigma=float(np.r_[pooly,ay].mean()),max(float(np.r_[pooly,ay].std()),1e-12)
                np.testing.assert_allclose([z['source_mean'],z['source_scale']],[mu,sigma],atol=1e-12,rtol=0)
                p=run.source.fusion.predict({'task':task,'base':z['Base_same_pool_and_target_labels'],
                    'bank':z['members'],'mu':mu,'sigma':sigma},policy['Shared'])
                np.testing.assert_allclose(p,z['Shared_adapted'],atol=1e-12,rtol=0)
                for method in ('Base','Base_same_pool_and_target_labels','Target_only','Shared_adapted'):
                    assert np.isfinite(z[method]).all() and len(z[method])==len(truth)
                    rows.append({'track':'shift','task':task,'data_id':c['data_id'],'family':family,'rho':rho,
                        'method':method,'extra_source_labels':len(Ty) if method not in ('Base','Target_only') else 0,
                        'target_adaptation_labels':0 if method=='Base' else len(ay),'evaluation_rows':len(truth),
                        **metrics(truth,z[method],task,max(float(np.std(truth)),1e-12))})
            count+=1
        print('SHIFT_TARGET_AUDIT_CASE',task,tid,flush=True)
    assert count==848
    df=pd.DataFrame(rows);df.to_csv(out/'condition_metrics.csv',index=False)
    keys=['track','task','family','method'];cols=['accuracy','auc','r2','nrmse','extra_source_labels','target_adaptation_labels','evaluation_rows']
    ds=df.groupby(keys+['data_id'])[cols].mean().reset_index()
    table=ds.groupby(keys)[cols].mean().reset_index();table=table.merge(ds.groupby(keys).size().rename('datasets').reset_index(),on=keys)
    table.to_csv(out/'family_averages.csv',index=False)
    gains=[]
    for (task,family),g in table[table.family!='clean'].groupby(['task','family']):
        metric='accuracy' if task=='classification' else 'r2'
        base=float(g.loc[g.method=='Base',metric].iloc[0]);shared=float(g.loc[g.method=='Shared_adapted',metric].iloc[0])
        control=float(g.loc[g.method=='Base_same_pool_and_target_labels',metric].iloc[0])
        gains.append({'track':'shift','task':task,'family':family,'Base':base,'Shared':shared,
            'gain':shared-base,'gain_vs_same_supervision_control':shared-control,'threshold':.02,'passed':shared-base>.02})
    atomic_json(out/'numerical_acceptance.json',{'family_results':gains,'all_six_shift_pass':all(g['passed'] for g in gains)})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(out/'independent_audit.json',{'status':'PASS','conditions':848,
        'context_rules_independently_reconstructed':reconstructed,
        'all_contexts_at_most256_rows_and_contain_all_adaptation_labels':True,
        'evaluation_adaptation_and_source_reference_row_ids_disjoint':True,
        'Base_and_Shared_identical_held_out_eval_rows':True,
        'Base_bitwise_matches_original_forecast_subset':True,'final_eval_labels_not_used_in_method':True,
        'original_test_subset_adaptation_explicit':True,'report_script_sha256':run.source.sha(__file__),'Markdown_bytes':0})
    perturb=RUN/'results/shared_expanded_pool_v291'
    if not (perturb/'independent_audit.json').exists():
        atomic_json(out/'combined_report_pending.json',{'status':'WAITING_FOR_PERTURBATION_AUDIT',
            'shift_audit_complete':True,'perturbation_parent':str(perturb)})
        print(table.to_string(index=False),flush=True)
        print(json.dumps(gains,indent=2),flush=True)
        return
    assert json.loads((perturb/'independent_audit.json').read_text())['status']=='PASS'
    ptable=pd.read_csv(perturb/'family_averages.csv');ptable['track']='perturbation'
    combined=pd.concat([ptable,table],ignore_index=True)
    combined=combined[(combined.family!='clean')&combined.method.isin(['Base','Shared_adapted'])]
    combined.to_csv(out/'combined_base_shared_results.csv',index=False)
    pg=json.loads((perturb/'numerical_acceptance.json').read_text())['family_results']
    allresults=[]
    for g in pg:
        allresults.append({'track':'perturbation','task':g['task'],'family':g['family'],
            'Base':g['Base'],'Shared':g['Shared'],'gain':g['gain_vs_original_Base'],
            'threshold':.05,'passed':g['gain_vs_original_Base']>.05})
    allresults.extend(gains);assert len(allresults)==12
    atomic_json(out/'combined_numerical_acceptance.json',{'family_results':allresults,'all_twelve_pass':all(g['passed'] for g in allresults),
        'perturbation_parent':str(perturb),'shift_parent':str(out),
        'strict_thresholds':True,'target_adaptation_excluded_from_shift_evaluation':True})
    print(table.to_string(index=False),flush=True);print(json.dumps(allresults,indent=2),flush=True)


if __name__=='__main__':main()
