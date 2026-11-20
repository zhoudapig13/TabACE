"""Fit source-reference supervised readouts only for the first active endpoint."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import interaction_sampling_probe_features_v394 as features
import domain_balanced_probe_adapter_v417 as probe
import interaction_sampling_nested_labels_v447 as split64
from prepare_data import RUN,ROOT,atomic_json,protocol

source=features.source
OUT=RUN/'results/interaction_sampling_nested64_isolated_probe_v448'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    split_plan=json.loads((split64.OUT/'plan.json').read_text())
    assert source.sha(split64.OUT/'case_manifest.json')==split_plan['manifest_sha256']
    assert source.sha(split64.__file__)==split_plan['script_sha256']
    assert split_plan['target_adaptation_rows']==64 and split_plan['evaluation_rows']==64
    plan={'seed':0,'active_endpoint':list(features.schedule.endpoint.ENDPOINT),'datasets':35,'conditions':140,
        'row_name':'Shared plus supervised class-prior-weighted linear readout, explicitly adapted',
        'target_labels':'nested64 target labels, four-fold training48 and validation16; final fit64, evaluation64; additional32 original-test labels explicitly permitted',
        'reference_labels':'existing clean independent training references, plus reserved64 validation labels',
        'encoder':'frozen source-only v307 TFM; no target labels in feature extraction or TFM contexts',
        'head':'source-prior weighted logistic readout; equalized source/target domain loss before class correction, with baseline total loss mass preserved',
        'class_prior_source':'empirical observable source-label proportions; no target/eval labels in prior','target_prior_smoothing_mass':0.,'class_weight_ratio_clip':[.05,20.],'readout_max_iterations':1000,'reason_for_solver_budget_increase':'selected pilot readouts reached former200-iteration cap','penalties':list(probe.PENALTIES),'blend_alphas':[0.,.25,.5,.75,1.],
        'selection':'target OOF accuracy; equal-domain source-prior-weighted NLL tie-break; source validation accuracy third',
        'split_preparation_outside_fitting_process':True,'target_training_weight':'max(4,source_reference_rows/target_training_rows)','target_split_plan_sha256':source.sha(split64.OUT/'plan.json'),'Native_and_Shared_same_labels':True,'total_loss_weight_normalization':'source_rows+4*target_rows','source_validation_prior_smoothing_mass':8.,
        'heldout_eval_labels_used_for_fit_or_selection':False,'other_endpoints_not_run':True,
        'not_original_zero_shot_Shared':True,'previously_scored_development_cohort':True,
        'frozen_registry_sha256':features.schedule.verify_frozen(),'features_plan_sha256':source.sha(features.OUT/'plan.json'),
        'script_sha256':source.sha(__file__),'probe_sha256':source.sha(probe.__file__),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def main(pilot=False,require_complete=False):
    prepare();fp=json.loads((features.OUT/'plan.json').read_text())
    cases=split64.endpoint_cases()
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
            f=features.OUT/'features'/str(tid)/f'interaction_sampling_{rho}.npz'
            if not f.with_suffix('.json').exists():continue
            dest=OUT/'predictions'/str(tid)/f.name
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];complete+=1;continue
            fm=json.loads(f.with_suffix('.json').read_text());assert source.sha(f)==fm['output_sha256']
            assert fm['reference_rows']==len(Ry) and fm['calibration_rows']==len(Cy) and not fm['target_labels_opened']
            arrays={};policies={};oofs={}
            with np.load(f) as z:
                for name in ['Native','Shared']:
                    result,policy,oof=probe.adapt(z[name+'_features'],z[name+'_raw'],Ry,Cy,ay,apos,epos,source_prior=source_prior)
                    arrays[name+'_probe']=result;policies[name+'_policy']=policy
                    offset=len(Ry)+len(Cy)
                    arrays[name+'_raw']=z[name+'_raw'][offset+epos]
                    for C,value in oof.items():
                        oofs[f'{name}_cv_target_{C}']=value['target'];oofs[f'{name}_cv_source_{C}']=value['source_validation']
            original=source.OUT/'predictions'/'shift'/'classification'/str(tid)/f.name
            om=json.loads(original.with_suffix('.json').read_text());assert source.sha(original)==om['output_sha256']
            with np.load(original) as z:arrays['Base']=z['Base'][epos]
            source.npwrite(dest,**arrays,**oofs)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('task_id','data_id','task','track','fold')}|
                {'family':'interaction_sampling','rho':rho,'output_sha256':source.sha(dest),'features_sha256':fm['output_sha256'],
                 'reference_sha256':c['reference_sha256'],'adaptation_sha256':c['adaptation_sha256'],
                 'original_prediction_sha256':om['output_sha256'],'adaptation_positions':apos.tolist(),'evaluation_positions':epos.tolist(),
                 'training_row_ids_sha256':protocol.array_sha256(np.r_[rids,aids]),
                 'validation_source_row_ids_sha256':protocol.array_sha256(cids),'evaluation_row_ids_sha256':protocol.array_sha256(eids),
                 'evaluation_labels_opened':False,**policies})
            complete+=1
        print('HEAVY_SUPERVISED_PROBE_CASE',tid,'conditions',complete,flush=True)
    prefix='pilot_' if pilot else '';expected=4*len(cases)
    atomic_json(OUT/f'{prefix}execution.json',{'status':'COMPLETE' if complete==expected else 'PARTIAL','conditions':complete,'expected':expected})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    features.schedule.verify_frozen()
    if require_complete:assert complete==expected


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--require-complete',action='store_true');a=p.parse_args();main(a.pilot,a.require_complete)
