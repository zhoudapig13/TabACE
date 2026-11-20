"""Add an original-backbone identity anchor; no new labels or model fits."""
import json
import numpy as np
import nonlinear_sensor_nested64_isolated_probe_v432 as inner
import base_anchored_gate_v436 as gate
from prepare_data import RUN, ROOT, atomic_json
from run_tabicl import normalize

source,features,split64=inner.source,inner.features,inner.split64
OUT=RUN/'results/nonlinear_sensor_base_anchored_probe_v437'


def prepare():
    audit=inner.OUT/'independent_audit.json'
    report=json.loads(audit.read_text())
    assert report['status']=='PASS' and report['conditions']==128
    plan={'seed':0,'active_endpoint':['shift','classification','nonlinear_sensor'],
          'datasets':32,'conditions':128,'target_adaptation_rows':64,'evaluation_rows':64,
          'change':'original pretrained Base posterior anchor with calibration-only blend selection',
          'same_inner_models_features_labels_and_evaluation_subset':True,
          'beta_grid':list(gate.BETAS),'selection':'strict target-calibration accuracy improvement; NLL then smaller beta tie-break',
          'inner_full_audit_sha256':source.sha(audit),'inner_plan_sha256':source.sha(inner.OUT/'plan.json'),
          'split_plan_sha256':source.sha(split64.OUT/'plan.json'),
          'original_Base_predictions_reused':True,'extra_source_and64_original_test_subset_labels':True,
          'not_original_zero_shot_Shared':True,'previously_scored_development_cohort':True,
          'final_eval_labels_used_for_fit_or_selection':False,
          'script_sha256':source.sha(__file__),'gate_sha256':source.sha(gate.__file__),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text())==plan
    else:
        atomic_json(p,plan)


def crossfit_prediction(name, feature_arrays, inner_arrays, meta, offset, apos):
    policy=meta[name+'_policy']
    raw=feature_arrays[name+'_raw'][offset+apos]
    if policy['kind']=='raw':
        return raw.copy()
    cv=inner_arrays[f'{name}_cv_target_{policy["C"]}']
    return normalize((1-policy['alpha'])*raw+policy['alpha']*cv)


def main():
    prepare();count=0
    for c in split64.endpoint_cases():
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:
            apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        assert len(apos)==len(epos)==64 and set(apos).isdisjoint(epos)
        for rho in source.novel.STRENGTHS:
            path=inner.OUT/'predictions'/str(c['task_id'])/f'nonlinear_sensor_{rho}.npz'
            meta=json.loads(path.with_suffix('.json').read_text())
            assert source.sha(path)==meta['output_sha256']
            feature=features.OUT/'features'/str(c['task_id'])/path.name
            fm=json.loads(feature.with_suffix('.json').read_text())
            assert source.sha(feature)==fm['output_sha256']==meta['features_sha256']
            offset=fm['reference_rows']+fm['calibration_rows']
            original=source.OUT/'predictions'/'shift'/'classification'/str(c['task_id'])/path.name
            assert source.sha(original)==meta['original_prediction_sha256']
            arrays={};policies={}
            with np.load(original) as original_arrays,np.load(feature) as fa,np.load(path) as ia:
                base_cv=original_arrays['Base'][apos]
                arrays['Base']=original_arrays['Base'][epos]
                arrays['Original_Base_cv']=base_cv
                for name in ['Native','Shared']:
                    cv=crossfit_prediction(name,fa,ia,meta,offset,apos)
                    policy=gate.select(ay,base_cv,cv)
                    arrays[name+'_probe']=gate.apply(policy,arrays['Base'],ia[name+'_probe'])
                    arrays[name+'_inner_probe']=ia[name+'_probe']
                    arrays[name+'_cv_probe']=cv
                    policies[name+'_outer_gate']=policy
            dest=OUT/'predictions'/str(c['task_id'])/path.name
            source.npwrite(dest,**arrays)
            atomic_json(dest.with_suffix('.json'),dict(meta,output_sha256=source.sha(dest),
                inner_prediction_sha256=source.sha(path),inner_metadata_sha256=source.sha(path.with_suffix('.json')),
                outer_gate_uses_only64_adaptation_labels=True,**policies))
            count+=1
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','conditions':count,'expected':128})
    assert count==128 and (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0


if __name__=='__main__':
    main()
