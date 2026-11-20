"""Independently refit reference fusion and audit all held-out row memberships."""
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import shared_reference_calibration_v272 as run
from prepare_data import ROOT, atomic_json, protocol
from run_tabicl import metrics


def main():
    out=run.OUT
    assert json.loads((out/'execution.json').read_text())['status']=='COMPLETE'
    plan=json.loads((out/'plan.json').read_text())
    assert run.source.sha(run.__file__)==plan['script_sha256']
    assert run.source.sha(run.fusion.__file__)==plan['fusion_sha256']
    assert run.source.sha(run.source.constructors.__file__)==plan['constructor_sha256']
    assert run.source.sha(run.source.fusion.__file__)==plan['original_fusion_sha256']
    refs={(r['task'],r['task_id']):r for r in json.loads((run.reference.OUT/'reference_manifest.json').read_text())}
    records,weights=[],[]
    conditions=0
    for c in json.loads((run.source.OUT/'case_manifest.json').read_text()):
        if c['track']!='perturbation':
            continue
        task,tid=c['task'],c['task_id']
        assert run.source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            X0,y0,Q0,numeric,sid,qid=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        r=refs[(task,tid)]
        assert run.source.sha(r['reference_file'])==r['reference_sha256']
        with np.load(r['reference_file']) as z:
            RX,Ry,rid=(z[k] for k in ('X_reference','y_reference','reference_indices'))
        assert set(map(int,rid)).isdisjoint(map(int,np.r_[sid,qid]))
        cleanfile=out/'reference_forecasts'/task/f'{tid}.npz'
        cm=json.loads(cleanfile.with_suffix('.json').read_text())
        assert run.source.sha(cleanfile)==cm['output_sha256']
        with np.load(cleanfile) as z:
            clean_ref_oof,clean_ref_query=z['reference_OOF'],z['query']
        folds=list(KFold(4,shuffle=True,random_state=0).split(RX))
        assert cm['folds']==[{'train':tr.tolist(),'validation':va.tolist()} for tr,va in folds]
        assert not cm['query_labels_opened']
        assert run.source.sha(c['truth_file'])==c['truth_sha256']
        with np.load(c['truth_file']) as z:
            truth=z['y_target']
        priority=run.source_priority(y0,task)
        for family,rho in [('clean',0)]+[(f,s) for f in c['families'] for s in run.source.novel.STRENGTHS]:
            path=out/'predictions'/task/str(tid)/f'{family}_{rho}.npz'
            m=json.loads(path.with_suffix('.json').read_text())
            assert run.source.sha(path)==m['output_sha256']
            assert m['original_input_sha256']==c['input_sha256']
            assert m['reference_sha256']==r['reference_sha256']
            assert m['clean_reference_forecast_sha256']==cm['output_sha256']
            assert m['local_fitting_labels_sha256']==protocol.array_sha256(Ry)
            assert not m['final_query_labels_opened']
            assert m['extra_reference_labels']==64 and m['new_TFM_calls']==10
            if rho:
                X,y,Q,_=run.source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
                np.testing.assert_array_equal(Q,Q0)
            else:
                X,y=X0,y0
            seen=[]
            for meta,(tr,va) in zip(m['reference_folds'],folds):
                assert meta['reference_train']==tr.tolist() and meta['reference_validation']==va.tolist()
                assert set(tr).isdisjoint(va)
                source_positions=priority[:256-len(tr)]
                np.testing.assert_array_equal(source_positions,meta['source_positions'])
                A=np.concatenate([X[source_positions],RX[tr]])
                b=np.concatenate([y[source_positions],Ry[tr]])
                assert len(b)==256
                assert protocol.array_sha256(A)==meta['context_X_sha256']
                assert protocol.array_sha256(b)==meta['context_y_sha256']
                seen.extend(va.tolist())
            assert sorted(seen)==list(range(64))
            old=run.source.OUT/'predictions/perturbation'/task/str(tid)/path.name
            parent=run.parent.OUT/'predictions'/task/str(tid)/path.name
            assert run.source.sha(old)==m['original_prediction_sha256']
            assert run.source.sha(parent)==m['mixed_parent_sha256']
            with np.load(path) as z,np.load(old) as original,np.load(parent) as mixed:
                np.testing.assert_array_equal(z['Base'],original['Base'])
                np.testing.assert_array_equal(z['OriginalShared'],original['Shared'])
                np.testing.assert_array_equal(z['Base_plus_same_reference'],mixed['Base_plus_same_reference'])
                P=z['reference_expert_predictions']
                np.testing.assert_array_equal(P[3],clean_ref_oof)
                expected=np.stack([original['Base'],original['Shared'],mixed['Base_plus_same_reference'],clean_ref_query])
                np.testing.assert_array_equal(z['query_expert_predictions'],expected)
                # Model reconstruction uses reference labels only, never truth.
                local=run.fusion.fit(P,Ry,task,max(float(np.std(Ry)),1e-12))
                np.testing.assert_allclose(local['weights'],m['local_model']['weights'],rtol=0,atol=1e-10)
                replay=run.fusion.predict(expected,local)
                np.testing.assert_allclose(replay,z['Shared_reference_calibrated'],rtol=0,atol=1e-10)
                predictions={'Base':z['Base'],'OriginalShared':z['OriginalShared'],
                    'Base_plus_same_reference':z['Base_plus_same_reference'],
                    'Reference_only':clean_ref_query,'Shared_reference_calibrated':replay}
                for method,pred in predictions.items():
                    assert np.isfinite(pred).all()
                    scores=metrics(truth,pred,task,max(float(np.std(truth)),1e-12))
                    records.append({'task':task,'data_id':c['data_id'],'family':family,'rho':rho,'method':method,**scores})
                weights.append({'task':task,'data_id':c['data_id'],'family':family,'rho':rho,
                    **dict(zip(plan['experts'],local['weights']))})
            conditions+=1
    assert conditions==1327
    df=pd.DataFrame(records)
    df.to_csv(out/'condition_metrics.csv',index=False)
    pd.DataFrame(weights).to_csv(out/'local_weights.csv',index=False)
    keys,cols=['task','family','method'],['accuracy','auc','r2','nrmse']
    ds=df.groupby(keys+['data_id'])[cols].mean().reset_index()
    table=ds.groupby(keys)[cols].mean().reset_index()
    table=table.merge(ds.groupby(keys).size().rename('datasets').reset_index(),on=keys)
    table.to_csv(out/'family_averages.csv',index=False)
    gains=[]
    for (task,family),g in table[table.family!='clean'].groupby(['task','family']):
        metric='accuracy' if task=='classification' else 'r2'
        base=float(g.loc[g.method=='Base',metric].iloc[0])
        shared=float(g.loc[g.method=='Shared_reference_calibrated',metric].iloc[0])
        control=float(g.loc[g.method=='Base_plus_same_reference',metric].iloc[0])
        gains.append({'task':task,'family':family,'metric':metric,'Base':base,'Shared_reference_calibrated':shared,
            'gain_vs_Base':shared-base,'gain_vs_same_reference_control':shared-control,
            'required_gain':.05,'passed':shared-base>=.05})
    assert len(gains)==6
    atomic_json(out/'numerical_acceptance.json',{'all_six_pass':all(r['passed'] for r in gains),'family_results':gains})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(out/'independent_audit.json',{'status':'PASS','conditions':conditions,
        'Base_and_original_Shared_bitwise_unchanged':True,'reference_training_validation_memberships_disjoint':True,
        'reference_crossfit_memberships_replayed':conditions*4,'local_models_independently_refitted':conditions,
        'prediction_fusion_replayed':conditions,'extra_reference_supervision_and_matched_control_explicit':True,
        'report_script_sha256':run.source.sha(__file__),'Markdown_bytes':0})
    print(table.to_string(index=False),flush=True)
    print(json.dumps(gains,indent=2),flush=True)


if __name__=='__main__':
    main()
