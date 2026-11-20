"""Fit source-reference supervised readouts only for the first active endpoint."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import heavy_probe_features_v340 as features
import domain_validated_probe_adapter_v365 as probe
from prepare_data import RUN,ROOT,atomic_json,protocol

source=features.source
OUT=RUN/'results/heavy_domain_validated_probe_v366'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'seed':0,'active_endpoint':list(features.schedule.endpoint.ENDPOINT),'datasets':64,'conditions':256,
        'row_name':'Shared plus supervised class-prior-weighted linear readout, explicitly adapted',
        'target_labels':'same32 v311 adaptation rows; four-fold training24 and validation8; final fit all32',
        'reference_labels':'existing clean independent training references, plus reserved64 validation labels',
        'encoder':'frozen source-only v307 TFM; no target labels in feature extraction or TFM contexts',
        'head':'unchanged source-prior weighted logistic readout; only the validation selection objective changes',
        'class_prior_source':'empirical observable source-label proportions; no target/eval labels in prior','target_prior_smoothing_mass':0.,'class_weight_ratio_clip':[.05,20.],'readout_max_iterations':1000,'reason_for_solver_budget_increase':'selected pilot readouts reached former200-iteration cap','penalties':list(probe.PENALTIES),'blend_alphas':[0.,.25,.5,.75,1.],
        'selection':'equal-domain accuracy with source-prior weighted source validation, equal-domain NLL tie-break; target OOF accuracy third',
        'target_training_weight':4.,'Native_and_Shared_same_labels':True,'source_validation_prior_smoothing_mass':8.,'cached_OOF_origin':'v358 identical features, labels, folds and training weights; independently checked by full refit audit',
        'heldout_eval_labels_used_for_fit_or_selection':False,'other_endpoints_not_run':True,
        'not_original_zero_shot_Shared':True,'previously_scored_development_cohort':True,
        'frozen_registry_sha256':features.schedule.verify_frozen(),'features_plan_sha256':source.sha(features.OUT/'plan.json'),
        'script_sha256':source.sha(__file__),'probe_sha256':source.sha(probe.__file__),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def main(pilot=False,require_complete=False):
    prepare();fp=json.loads((features.OUT/'plan.json').read_text())
    cases=features.schedule.endpoint.endpoint_cases()
    if pilot:cases=[c for c in cases if c['task_id'] in fp['pilot_task_ids']]
    complete=0
    for c in cases:
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as original_input:source_labels=original_input['y_source']
        source_prior=np.bincount(source_labels.astype(int))/len(source_labels)
        tid=c['task_id'];assert source.sha(c['reference_file'])==c['reference_sha256']
        with np.load(c['reference_file']) as z:Ry,Cy,rids,cids=(z[k] for k in ('y_train_reference','y_calibration','train_reference_indices','calibration_indices'))
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:apos,epos,ay,aids,eids=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y','adaptation_indices','evaluation_indices'))
        assert set(np.r_[rids,aids]).isdisjoint(eids) and set(np.r_[rids,aids]).isdisjoint(cids)
        for rho in source.novel.STRENGTHS:
            f=features.OUT/'features'/str(tid)/f'cell_heavy_tail_{rho}.npz'
            if not f.with_suffix('.json').exists():continue
            dest=OUT/'predictions'/str(tid)/f.name
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];complete+=1;continue
            fm=json.loads(f.with_suffix('.json').read_text());assert source.sha(f)==fm['output_sha256']
            assert fm['reference_rows']==len(Ry) and fm['calibration_rows']==len(Cy) and not fm['target_labels_opened']
            arrays={};policies={};oofs={}
            old=RUN/'results/heavy_source_prior_probe_v358/predictions'/str(tid)/f.name
            old_meta=json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old)==old_meta['output_sha256'] and old_meta['features_sha256']==fm['output_sha256']
            assert old_meta['adaptation_sha256']==c['adaptation_sha256'] and old_meta['reference_sha256']==c['reference_sha256']
            with np.load(f) as z,np.load(old) as cached:
                for name in ['Native','Shared']:
                    cache={str(C):{'target':cached[f'{name}_cv_target_{C}'],'source_validation':cached[f'{name}_cv_source_{C}']} for C in probe.PENALTIES}
                    result,policy,oof=probe.adapt(z[name+'_features'],z[name+'_raw'],Ry,Cy,ay,apos,epos,source_prior=source_prior,cached_oof=cache)
                    arrays[name+'_probe']=result;policies[name+'_policy']=policy
                    offset=len(Ry)+len(Cy)
                    arrays[name+'_raw']=z[name+'_raw'][offset+epos]
                    for C,value in oof.items():
                        oofs[f'{name}_cv_target_{C}']=value['target'];oofs[f'{name}_cv_source_{C}']=value['source_validation']
            original=source.OUT/'predictions'/'perturbation'/'classification'/str(tid)/f.name
            om=json.loads(original.with_suffix('.json').read_text());assert source.sha(original)==om['output_sha256']
            with np.load(original) as z:arrays['Base']=z['Base'][epos]
            source.npwrite(dest,**arrays,**oofs)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('task_id','data_id','task','track','fold')}|
                {'family':'cell_heavy_tail','rho':rho,'output_sha256':source.sha(dest),'features_sha256':fm['output_sha256'],
                 'reference_sha256':c['reference_sha256'],'adaptation_sha256':c['adaptation_sha256'],
                 'original_prediction_sha256':om['output_sha256'],'adaptation_positions':apos.tolist(),'evaluation_positions':epos.tolist(),
                 'training_row_ids_sha256':protocol.array_sha256(np.r_[rids,aids]),
                 'validation_source_row_ids_sha256':protocol.array_sha256(cids),'evaluation_row_ids_sha256':protocol.array_sha256(eids),
                 'evaluation_labels_opened':False,'cached_oof_sha256':old_meta['output_sha256'],**policies})
            complete+=1
        print('HEAVY_SUPERVISED_PROBE_CASE',tid,'conditions',complete,flush=True)
    prefix='pilot_' if pilot else '';expected=4*len(cases)
    atomic_json(OUT/f'{prefix}execution.json',{'status':'COMPLETE' if complete==expected else 'PARTIAL','conditions':complete,'expected':expected})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    features.schedule.verify_frozen()
    if require_complete:assert complete==expected


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--require-complete',action='store_true');a=p.parse_args();main(a.pilot,a.require_complete)
