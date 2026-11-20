"""Dataset-specific calibration on reassigned target labels, never fresh truth.

All hyperparameters use the nested32/32 calibration split. The final policy is
refitted on cross-fitted64 calibration rows. Native1 with the same target64
labels is the primary comparator; the zero-target-label Base is also retained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

import target_supervised_banks_v98 as banks
import semisynthetic_ranking_calibration_v96 as rank
from semisynthetic_feature_adapter_v93 import predict as gated_predict
from prepare_data import RUN,atomic_json
from fit_combiners import fold_mapping
from run_tabicl import metrics

OUT=RUN/'results/target_supervised_adaptation_v99'
REGULARIZATIONS=(.001,.01,.1,1.)


def predict(c,policy):
    if policy['kind']=='member':return c['bank'][policy['slot']]
    return gated_predict(c,policy)


def load_cell(task_id,family,rho,stages,matched,data):
    parts=[]
    for stage in stages:
        path=banks.OUT/'banks'/str(task_id)/f'{family}_{rho}'/f'{stage}.npz'
        with np.load(path) as z:
            bank=z['members'].copy()
            if not matched:bank[3]=z['bootstrap']
            parts.append({'base':z['base_target'],'bank':bank,'positions':z['query_positions'],
                          'zero':z['base_zero_target'] if 'base_zero_target' in z.files else None})
    positions=np.concatenate([p['positions'] for p in parts]);order=np.argsort(positions)
    base=np.concatenate([p['base'] for p in parts])[order]
    bank=np.concatenate([p['bank'] for p in parts],axis=1)[:,order]
    positions=positions[order]
    if stages==['evaluation']:
        y=None
    else:
        assert np.all(positions<len(data['y_calibration']))
        y=data['y_calibration'][positions]
    return {'task':'classification','role':'validation','base':base,'bank':bank,'y':y,
        'positions':positions,'zero':parts[0]['zero'] if stages==['evaluation'] else None,
        'meta':{'task_id':task_id,'data_id':data['data_id'],'family':family,'rho':rho,'scale':1.}}


def cells_for(task_id,stage,matched,seen,data):
    stages={'adapt':['adapt_oof_0','adapt_oof_1'],'validation':['validation'],
            'refit':['all_oof_0','all_oof_1'],'evaluation':['evaluation']}[stage]
    out=[]
    for family,rho in banks.CONDITIONS:
        if stage=='evaluation' and family not in ('clean','shortcut'):continue
        if stage!='evaluation' and not seen and family=='shortcut':continue
        out.append(load_cell(task_id,family,rho,stages,matched,data))
    return out


def source_prior(meta,method):
    mapping,_=fold_mapping('classification','five');fold=mapping[int(meta['data_id'])]
    path=rank.OUT/'folds/classification'/str(fold)/f'{method}_policy.json'
    saved=json.loads(path.read_text());assert int(meta['data_id']) in saved['excluded_data_ids']
    policy=saved['policy']
    values=np.asarray(policy['parameters']) if policy['kind']!='base' else np.r_[np.zeros(5),np.ones(10),.5]
    return values,{'path':str(path),'sha256':banks.split.parent.original.sha(path),'outer_fold':fold}


def fit(cells,regularization,prior):
    tensors=rank.pair_data(cells)
    initial=prior.copy()
    def objective(values):
        p=torch.tensor(values,dtype=torch.float64,requires_grad=True)
        positive=rank.mapped_probability(p,tensors,'positive').clamp(1e-7,1-1e-7)
        negative=rank.mapped_probability(p,tensors,'negative').clamp(1e-7,1-1e-7)
        penalty=(p-torch.as_tensor(prior,dtype=torch.float64)).square().mean()
        loss=(torch.nn.functional.softplus(-(torch.logit(positive)-torch.logit(negative)))*tensors['weights']).sum()+regularization*penalty
        loss.backward();return float(loss.detach()),p.grad.detach().numpy()
    result=minimize(objective,initial,method='L-BFGS-B',jac=True,
        bounds=[(-8,8)]*5+[(-10,10)]*10+[(0,1)],options={'maxiter':220,'ftol':1e-9,'gtol':1e-6})
    assert np.isfinite(result.x).all()
    return {'kind':'agreement_gate_adapted','parameters':result.x.tolist(),'regularization':regularization,
        'gate_variant':'entropy_or_disagreement','target_supervised':True,'training_objective':float(result.fun),
        'optimizer_success':bool(result.success),'prior_parameters':prior.tolist(),'sampled_pairs':len(tensors['weights'])}


def scores(cells,policy):
    values=[];accuracies=[]
    for c in cells:
        metric=metrics(c['y'],predict(c,policy),'classification',1.)
        if metric['auc'] is not None:values.append(metric['auc'])
        accuracies.append(metric['accuracy'])
    return {'auc':float(np.mean(values)),'accuracy':float(np.mean(accuracies))}


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text())['code_sha256']==banks.split.parent.original.sha(__file__)
        return
    atomic_json(path,{'status':'FROZEN_BEFORE_TARGET_LABEL_ADAPTATION','seed':0,
        'datasets':72,'old_query_labels_reassigned_to_calibration_per_dataset':64,
        'nested_hyperparameter_adaptation':32,'nested_validation':32,
        'final_refit':'OOF predictions for all64 labels; base/branch contexts exclude each row from its own prediction',
        'main_comparator':'Native1 with source192+same target64 labels, total context256',
        'zero_target_Base_reported_separately':True,'alternative_TFM_branches':5,
        'target_context_only_control_reported':True,
        'A_C_calibration':['Clean','Shortcut','Domain','Covariate'],
        'B_D_calibration':['Clean','Domain','Covariate'],'B_D_Shortcut_calibration_excluded':True,
        'hyperparameters':list(REGULARIZATIONS),'hyperparameter_metric':'mean calibration-validation AUROC across allowed conditions',
        'candidates':['target-aware Base identity','target-labeled branch alone','cross-dataset prior policy','target-calibrated policy'],
        'priors_exclude_same_dataset':True,'fresh_evaluation_truth_loaded_by_fitting_code':False,
        'native_input_label_budget_matched':True,'extra_target_supervision_must_be_marked':True,
        'strict_C_D_all_feature_columns_retained':True,
        'new_backbone_training':False,'new_raw_feature_prediction_head':False,
        'original_paper_replication':False,'zero_target_label_generalization_claim':False,
        'scope':'few-shot target-supervised semisynthetic adaptation',
        'evaluation_labels_only_opened_after_all72_tasks_complete':True,
        'statistical_family':'16 planned A-D comparisons: four against each of target-native, zero-target and target-only controls, plus four legacy Concept comparisons',
        'code_sha256':banks.split.parent.original.sha(__file__),'Markdown_written':False})


def worker(task_id):
    torch.set_num_threads(1);torch.manual_seed(0)
    path=banks.split.OUT/'inputs'/f'{task_id}.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as z:data={k:z[k] for k in z.files}
    data['data_id']=int(meta['data_id'])
    folder=OUT/'tasks'/str(task_id)
    if (folder/'completion.json').exists():return {'task_id':task_id,'status':'CACHED'}
    for method,seen,matched in [('A',True,True),('B',False,True),('C',True,False),('D',False,False)]:
        training=cells_for(task_id,'adapt',matched,seen,data)
        validation=cells_for(task_id,'validation',matched,seen,data)
        refit=cells_for(task_id,'refit',matched,seen,data)
        if not seen:
            assert all(c['meta']['family']!='shortcut' for c in training+validation+refit)
        assert set(training[0]['positions']).isdisjoint(validation[0]['positions'])
        prior,origin=source_prior(meta,method)
        prior_policy={'kind':'agreement_gate_adapted','parameters':prior.tolist()}
        trials=[]
        for name,model in [('Base',{'kind':'base'}),('Target_only',{'kind':'member','slot':1}),('Prior',prior_policy)]:
            trials.append({'candidate':name,'validation_scores':scores(validation,model),'model':model})
        for reg in REGULARIZATIONS:
            model=fit(training,reg,prior)
            trials.append({'candidate':'Target_fit','regularization':reg,'validation_scores':scores(validation,model),'model':model})
        best=max(trials,key=lambda t:t['validation_scores']['auc'])
        policy=fit(refit,best['regularization'],prior) if best['candidate']=='Target_fit' else best['model']
        atomic_json(folder/f'{method}_policy.json',{'method':method,'seen':seen,'matched':matched,
            'unique_target_labels':64,'adapt_positions':data['adapt_positions'].tolist(),
            'validation_positions':data['validation_positions'].tolist(),'source_prior':origin,
            'allowed_families':sorted({c['meta']['family'] for c in training}),'trials':trials,
            'selected_candidate':best['candidate'],'selected_regularization':best.get('regularization'),
            'policy':policy,'fresh_evaluation_labels_used':False})
        for c in cells_for(task_id,'evaluation',matched,seen,data):
            m=c['meta'];prefix=f'{m["family"]}_{m["rho"]}'
            predictions={method:predict(c,policy)}
            if method=='A':predictions.update({'Base_target':c['base'],'Base_zero_target':c['zero'],
                'Target_only':c['bank'][1],'Single_matched':c['bank'][3]})
            for name,p in predictions.items():
                out=folder/'evaluation_predictions'/f'{prefix}_{name}.npz'
                out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,prediction=p)
        print('TARGET_ADAPTED_POLICY',task_id,method,best['candidate'],best['validation_scores']['auc'],flush=True)
    atomic_json(folder/'completion.json',{'status':'COMPLETE','task_id':task_id,'target_labels':64,
        'evaluation_labels_loaded':False,'all_four_policies_frozen':True})
    return {'task_id':task_id,'status':'COMPLETE'}


def launch():
    freeze()
    ids=sorted([int(p.stem) for p in (banks.split.OUT/'inputs').glob('*.npz')])
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(worker,i) for i in ids]):
            r=f.result();results.append(r);print(r,flush=True)
            atomic_json(OUT/'execution.json',{'status':'RUNNING','tasks':results})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','datasets':len(results),'tasks':results,
        'all_policies_frozen_before_evaluation_labels_opened':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--task-id',type=int)
    a=p.parse_args()
    if a.phase=='worker':worker(a.task_id)
    else:launch()
