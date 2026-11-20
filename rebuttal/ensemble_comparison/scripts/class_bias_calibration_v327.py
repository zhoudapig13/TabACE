"""Target-label class-intercept adaptation; all neural features stay frozen.

Regression results are copied unchanged from the reference/context experiment.
Class-bias LOO features never include a model fitted on target labels, preventing
indirect LOO leakage through a target-trained feature extractor.
"""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import reference_target_contexts_v323 as parent
import crossfit_class_bias_v326 as calibration
from prepare_data import RUN,ROOT,atomic_json

source=parent.source
OUT=RUN/'results/class_bias_calibration_v327'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'seed':0,'cases':175,'conditions':2175,'classification_only_change':True,
        'regression_results_unchanged_from':str(parent.OUT),
        'calibration_features':'original frozen Base, source-finetuned same method, reference-only same method',
        'calibration_feature_TFM_has_target_labels_in_context':False,
        'class_calibration':'one bounded additive logit intercept per class with L2 regularization',
        'penalties':[.01,.1,1.],'offset_cap':3.,'gradient_iterations_max':60,
        'selection':'exact leave-one-out label exclusion for intercepts; same v322 cross-fit convex selector',
        'final_selector_components':['original frozen Base','LOO-selected class-bias-calibrated source-only model','reference-plus-target-context model'],
        'target_labels':'existing v311 adaptation subset only; evaluation rows unchanged',
        'extra_clean_source_and_test_subset_labels_used':True,'not_original_zero_shot_Shared':True,
        'previously_scored_development_cohort':True,'heldout_evaluation_labels_used_for_selection':False,
        'parent_plan_sha256':source.sha(parent.OUT/'plan.json'),'script_sha256':source.sha(__file__),
        'calibration_sha256':source.sha(calibration.__file__),'selector_sha256':source.sha(parent.selector.__file__),
        'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def adapt_classification(c,z,old,original,ay,apos,epos):
    arrays={k:z[k] for k in z.files};policies={}
    for name in ['Native','Shared']:
        key=name.lower()
        sources={'original_Base':original['Base'],'finetuned':old[f'{name}_finetuned'],
            'reference_only':z[f'reference_{key}_full']}
        bias,cv=calibration.fit({k:v[apos] for k,v in sources.items()},ay)
        calibrated=calibration.apply(bias,{k:v[epos] for k,v in sources.items()})
        policy=parent.selector.select(ay,np.stack([original['Base'][apos],cv,z[f'cv_{key}']]),'classification',1.)
        arrays[f'{name}_calibrated']=calibrated;arrays[f'cv_calibrated_{key}']=cv
        arrays[f'{name}_adapted']=parent.selector.apply(policy,
            np.stack([original['Base'][epos],calibrated,z[f'{name}_target_context']]),'classification')
        policies[f'{name}_bias_policy']=bias;policies[f'{name}_policy']=policy
    return arrays,policies


def main(pilot=False,require_complete=False):
    prepare();cases=json.loads((parent.oldbank.split.OUT/'case_manifest.json').read_text())
    if pilot:cases=[cases[i] for i in parent.PILOT_POSITIONS]
    complete=0
    for c in cases:
        track,task,tid=c['track'],c['task'],c['task_id']
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as labels:apos,epos,ay=(labels[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        for family,rho in [('clean',0)]+[(f,r) for f in c['families'] for r in source.novel.STRENGTHS]:
            inp=parent.OUT/'predictions'/track/task/str(tid)/f'{family}_{rho}.npz'
            if not inp.with_suffix('.json').exists():continue
            dest=OUT/'predictions'/track/task/str(tid)/inp.name
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];complete+=1;continue
            pm=json.loads(inp.with_suffix('.json').read_text());assert source.sha(inp)==pm['output_sha256']
            with np.load(inp) as z:
                if task=='classification':
                    oldfile=parent.oldbank.OUT/'predictions'/track/task/str(tid)/inp.name
                    originalfile=source.OUT/'predictions'/track/task/str(tid)/inp.name
                    assert source.sha(oldfile)==pm['previous_bank_sha256'] and source.sha(originalfile)==pm['original_prediction_sha256']
                    with np.load(oldfile) as old,np.load(originalfile) as original:
                        arrays,policies=adapt_classification(c,z,old,original,ay,apos,epos)
                else:arrays,policies={k:z[k] for k in z.files},{}
            source.npwrite(dest,**arrays)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('track','task','task_id','data_id','fold')}|
                {'family':family,'rho':rho,'output_sha256':source.sha(dest),'parent_output_sha256':pm['output_sha256'],
                 'adaptation_sha256':c['adaptation_sha256'],'evaluation_positions':epos.tolist(),'adaptation_positions':apos.tolist(),
                 'class_bias_features_depend_on_target_labels':False,'evaluation_labels_opened':False,
                 'classification_only_change':task=='classification',**policies})
            complete+=1
        print('CLASS_BIAS_CASE',track,task,tid,'conditions',complete,flush=True)
    expected=sum(1+4*len(c['families']) for c in cases)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(OUT/f'{"pilot_" if pilot else ""}execution.json',{'status':'COMPLETE' if complete==expected else 'PARTIAL','conditions':complete,'expected':expected})
    if require_complete:assert complete==expected


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--require-complete',action='store_true');a=p.parse_args();main(a.pilot,a.require_complete)
