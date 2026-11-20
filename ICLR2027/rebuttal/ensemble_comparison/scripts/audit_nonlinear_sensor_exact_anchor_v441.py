"""Independently enumerate the new anchor gate; reuse fully audited inner fits."""
import json
import numpy as np
import pandas as pd
import nonlinear_sensor_exact_base_anchored_probe_v440 as run
from prepare_data import RUN,ROOT,atomic_json
from run_tabicl import normalize,metrics
from serial_endpoint_state_v371 import verify_registry


def main():
    plan=json.loads((run.OUT/'plan.json').read_text())
    source=run.source
    assert source.sha(run.__file__)==plan['script_sha256'] and source.sha(run.gate.__file__)==plan['gate_sha256']
    assert source.sha(run.inner.OUT/'independent_audit.json')==plan['inner_full_audit_sha256']
    assert source.sha(run.inner.OUT/'plan.json')==plan['inner_plan_sha256']
    assert source.sha(run.split64.OUT/'plan.json')==plan['split_plan_sha256']
    old_cases={c['task_id']:c for c in run.features.schedule.endpoint.endpoint_cases()}
    rows=[];count=0
    for c in run.split64.endpoint_cases():
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:
            a,e,ay,aids,eids=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y','adaptation_indices','evaluation_indices'))
        assert len(a)==len(e)==64 and set(a).isdisjoint(e) and set(aids).isdisjoint(eids)
        with np.load(old_cases[c['task_id']]['adaptation_file']) as z:
            old_a,old_e=z['adaptation_positions'],z['evaluation_positions']
        np.testing.assert_array_equal(a,np.r_[old_a,old_e[:32]])
        np.testing.assert_array_equal(e,old_e[32:])
        for rho in source.novel.STRENGTHS:
            path=run.OUT/'predictions'/str(c['task_id'])/f'nonlinear_sensor_{rho}.npz'
            meta=json.loads(path.with_suffix('.json').read_text())
            assert source.sha(path)==meta['output_sha256'] and not meta['evaluation_labels_opened']
            assert meta['adaptation_sha256']==c['adaptation_sha256']
            inner=run.inner.OUT/'predictions'/str(c['task_id'])/path.name
            im=json.loads(inner.with_suffix('.json').read_text())
            assert source.sha(inner)==im['output_sha256']==meta['inner_prediction_sha256']
            assert source.sha(inner.with_suffix('.json'))==meta['inner_metadata_sha256']
            f=run.features.OUT/'features'/str(c['task_id'])/path.name
            fm=json.loads(f.with_suffix('.json').read_text())
            assert source.sha(f)==fm['output_sha256']==meta['features_sha256']
            offset=fm['reference_rows']+fm['calibration_rows']
            original=source.OUT/'predictions'/'shift'/'classification'/str(c['task_id'])/path.name
            assert source.sha(original)==meta['original_prediction_sha256']
            with np.load(f) as fa,np.load(inner) as ia,np.load(original) as oa,np.load(path) as actual:
                base_cv=normalize(oa['Base'][a])
                base=oa['Base'][e]
                np.testing.assert_array_equal(actual['Base'],base)
                np.testing.assert_allclose(actual['Original_Base_cv'],base_cv,atol=1e-12,rtol=0)
                for name in ['Native','Shared']:
                    policy=im[name+'_policy']
                    raw=fa[name+'_raw'][offset+a]
                    cv=raw if policy['kind']=='raw' else normalize((1-policy['alpha'])*raw+policy['alpha']*ia[f'{name}_cv_target_{policy["C"]}'])
                    base_acc=float(np.mean(base_cv.argmax(1)==ay))
                    candidates=[]
                    trial_betas=[v['beta'] for v in meta[name+'_outer_gate']['trials']]
                    assert 0. in trial_betas and 1. in trial_betas
                    for beta in trial_betas:
                        assert 0.<=beta<=1.
                        p=normalize((1-beta)*base_cv+beta*cv)
                        acc=float(np.mean(p.argmax(1)==ay))
                        nll=-float(np.log(np.maximum(p[np.arange(len(ay)),ay.astype(int)],1e-12)).mean())
                        if beta==0. or acc>base_acc:
                            candidates.append((acc,-nll,-beta,beta))
                    selected=max(candidates)[3]
                    chosen=normalize((1-selected)*base_cv+selected*cv)
                    selected_acc=float(np.mean(chosen.argmax(1)==ay))
                    selected_nll=-float(np.log(np.maximum(chosen[np.arange(len(ay)),ay.astype(int)],1e-12)).mean())
                    delta=normalize(cv)-base_cv
                    cuts=[0.,1.]
                    for aclass in range(base_cv.shape[1]):
                        for bclass in range(aclass):
                            den=delta[:,aclass]-delta[:,bclass]
                            ok=np.abs(den)>1e-14
                            v=(base_cv[:,bclass]-base_cv[:,aclass])[ok]/den[ok]
                            cuts.extend(v[(v>0.)&(v<1.)].tolist())
                    cuts=sorted(set(cuts))
                    from scipy.optimize import minimize_scalar
                    for left,right in zip(cuts[:-1],cuts[1:]):
                        mid=(left+right)/2.
                        accuracy=float(np.mean((base_cv+mid*delta).argmax(1)==ay))
                        assert accuracy<=selected_acc
                        if accuracy==selected_acc and selected_acc>base_acc:
                            margin=min(1e-9,(right-left)/4.)
                            objective=lambda b:-float(np.log(np.maximum((base_cv+b*delta)[np.arange(len(ay)),ay.astype(int)],1e-12)).mean())
                            lo,hi=left+margin,right-margin
                            if hi>lo:
                                opt=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-10})
                                bound=min(objective(lo),objective(hi),float(opt.fun))
                                assert selected_nll<=bound+1e-7
                    assert selected==meta[name+'_outer_gate']['beta']
                    expected=normalize((1-selected)*base+selected*ia[name+'_probe'])
                    np.testing.assert_allclose(actual[name+'_probe'],expected,atol=1e-12,rtol=0)
                    np.testing.assert_allclose(actual[name+'_cv_probe'],cv,atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(actual[name+'_inner_probe'],ia[name+'_probe'])
                # Evaluation labels are opened only after independently
                # reconstructing every training-label-based gate and output.
                assert source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
                with np.load(c['heldout_truth_file']) as z:
                    y=z['y_target'];np.testing.assert_array_equal(z['target_indices'],eids)
                for method in ['Base','Native_probe','Shared_probe']:
                    rows.append({'data_id':c['data_id'],'task_id':c['task_id'],'rho':rho,'method':method,
                                 **metrics(y,actual[method],'classification',1.)})
            count+=1
    assert count==128
    frame=pd.DataFrame(rows)
    frame.to_csv(run.OUT/'condition_metrics.csv',index=False)
    table=frame.groupby(['data_id','method'])[['accuracy','auc']].mean().groupby('method').mean().reset_index()
    table['datasets']=32
    table.to_csv(run.OUT/'family_averages.csv',index=False)
    get=lambda name:float(table.loc[table.method==name,'accuracy'].iloc[0])
    gain=get('Shared_probe')-get('Base')
    atomic_json(run.OUT/'numerical_acceptance.json',{'pilot_only':False,
        'active_endpoint':['shift','classification','nonlinear_sensor'],'conditions':128,'datasets':32,
        'Base':get('Base'),'Shared_supervised_probe':get('Shared_probe'),'gain':gain,
        'gain_vs_Native_same_supervision':get('Shared_probe')-get('Native_probe'),
        'required_gain_strictly_greater_than':.02,'active_endpoint_pass':gain>.02,'all_twelve_pass':False,
        'extra_source_and64_test_subset_supervision_explicit':True})
    verify_registry(RUN/'results/frozen_endpoints_v381/registry.json')
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(run.OUT/'independent_audit.json',{'status':'PASS','pilot_only':False,'conditions':128,
        'target_split_nested_prefix_independently_checked':True,'all_class_decision_intervals_and_NLL_optima_independently_checked_from_adaptation_labels_only':True,
        'Original_Base_predictions_bitwise_unchanged':True,'source_and64_target_label_inner_models_fully_audited_upstream':True,
        'inner_full_audit_sha256':plan['inner_full_audit_sha256'],
        'all_adaptation_rows_excluded_from_evaluation':True,'Markdown_bytes':0,'script_sha256':source.sha(__file__)})
    print(table.to_string(index=False),flush=True)
    print('GAIN',gain,flush=True)


if __name__=='__main__':
    main()
