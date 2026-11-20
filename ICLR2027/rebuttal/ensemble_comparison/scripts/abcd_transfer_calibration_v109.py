"""Regularized seen-mechanism adaptation from an unseen-mechanism prior.

B/D stay byte-identical to v100 classification/v107 regression. A/C priors
are trained only on allowed other mechanisms and exclude each inner
validation cohort. Seen updates use only calibration labels. No desired
between-method order or test metric is present in the loss/selection.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k,'1')
import numpy as np
import torch
from scipy.optimize import minimize

import abcd_metric_aligned_v100 as classification
import abcd_quality_nested_cv_v107 as regression
import abcd_balanced_quality_v106 as inputs
import abcd_source_quality_v104 as algorithm
from prepare_data import RUN,atomic_json

OUT=RUN/'results/abcd_transfer_calibration_v109'
SHRINKAGES=(.0001,.001,.01,.1,1.)
TEACHER_REGULARIZATION=.01


def freeze():
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/'plan.json'
    plan={'seed':0,'code_sha256':classification.origin.sha(__file__),
        'hypothesis':'Seen calibration can suffer negative transfer; regularize updates toward a prior trained on other mechanisms',
        'BD_classification_frozen_from':str(classification.OUT),'BD_regression_frozen_from':str(regression.OUT),
        'classification_inputs':'v98 nested calibration banks, target64; no extra target labels',
        'regression_inputs':'v105 paired bootstrap/source-heldout-quality banks; reference64 heldout source labels',
        'seen_updates':['classification Shortcut+Clean','regression Concept+Clean'],
        'unseen_priors':['classification Clean/Domain/Covariate','regression excludes Concept/Temporal'],
        'teacher_regularization_fixed':TEACHER_REGULARIZATION,
        'shrinkages':SHRINKAGES,'candidates_per_seen_method':7,
        'candidates':['previous seen policy','unseen-prior identity','five prior-centered updates'],
        'classification_selection':'nested32/32 calibration Accuracy then NLL',
        'regression_selection':'four-fold inner other-dataset OOF R2',
        'priors_exclude_inner_validation_labels':True,'outer_evaluation_dataset_excluded_from_regression_fit':True,
        'teacher_fixed_hyperparameter_not_selected_on_inner_validation':True,
        'no_between_method_margin_in_fit_or_selection':True,
        'reference_effect_size_for_reporting_only':.005,'context_cap':256,'backbone_updated':False,
        'both_query_cohorts_previously_evaluated_development':True,'independent_confirmation':False,
        'budget':'one transfer hypothesis, all shrinkages selected on calibration; preserve all outcomes',
        'original_paper_replication':False,'Markdown_written':False}
    if p.exists():assert json.loads(p.read_text())['code_sha256']==plan['code_sha256']
    else:atomic_json(p,plan)


def fit_regression(values,prior,regularization):
    counts={}
    for c in values:counts[c['meta']['data_id']]=counts.get(c['meta']['data_id'],0)+1
    columns={k:[] for k in ('M','D','Q','Y','weights')}
    for c in values:
        m,d,q=algorithm.features(c);n=len(c['y']);factor=c['sigma']**2/max(float(c['y'].var()),1e-9)
        columns['M'].append(m);columns['D'].append(d);columns['Q'].append(np.broadcast_to(q,m.shape))
        columns['Y'].append((c['y']-c['mu'])/c['sigma'])
        columns['weights'].append(np.full(n,factor/(len(counts)*counts[c['meta']['data_id']]*n)))
    t={k:torch.as_tensor(np.concatenate(v),dtype=torch.float64) for k,v in columns.items()}
    prior=np.asarray(prior,dtype=float)
    def objective(x):
        p=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        agreement=torch.exp(-t['D']/torch.exp(p[-1]))
        weights=torch.softmax(p[:6]+t['Q']*p[6:12]+agreement*p[12:18],dim=1)
        prediction=(weights*t['M']).sum(1)
        loss=((prediction-t['Y']).square()*t['weights']).sum()+regularization*(p-torch.as_tensor(prior)).square().mean()
        loss.backward();return float(loss.detach()),p.grad.detach().numpy()
    result=minimize(objective,prior.copy(),method='L-BFGS-B',jac=True,
        bounds=[(-8,8)]*6+[(0,10)]*6+[(-10,10)]*6+[(-4,4)],options={'maxiter':240,'ftol':1e-9,'gtol':1e-6})
    assert np.isfinite(result.x).all()
    return {'kind':'source_quality_fusion','parameters':result.x.tolist(),'regularization':regularization,
        'prior_parameters':prior.tolist(),'training_objective':float(result.fun),'optimizer_success':bool(result.success)}


def saved_inner(saved,g):
    p=saved['policy']
    if p['kind']!='source_quality_fusion':return p
    chosen=next(t for t in saved['trials'] if t['kind']=='fit' and t['scope']==saved['selected_scope'] and t['regularization']==saved['selected_regularization'])
    return next(t['policy'] for t in chosen['inner_folds'] if t['inner_fold']==g)


def regression_worker(fold):
    torch.set_num_threads(1);folder=OUT/'regression'/str(fold)
    for method,seen,matched in classification.METHODS:
        oldfile=regression.OUT/'folds'/str(fold)/f'{method}_policy.json'
        saved=json.loads(oldfile.read_text())
        if not seen:
            saved['frozen_unseen_control']=True;saved['frozen_source_sha256']=classification.origin.sha(oldfile)
            atomic_json(folder/f'{method}_policy.json',saved);continue
        bank=inputs.load_cells(matched)
        source=[c for c in bank if c['fold']!=fold]
        folds=sorted({c['fold'] for c in source});trials=[]
        scores={key:[] for key in ['previous','prior']+[str(x) for x in SHRINKAGES]}
        priors=[]
        for inner in folds:
            val=[c for c in source if c['fold']==inner and (c['meta']['rho']==0 or c['meta']['family']=='concept')]
            known=[c for c in source if c['fold']!=inner and (c['meta']['rho']==0 or c['meta']['family'] not in ('concept','temporal'))]
            target=[c for c in source if c['fold']!=inner and (c['meta']['rho']==0 or c['meta']['family']=='concept')]
            training_ids={c['meta']['data_id'] for c in known+target};validation_ids={c['meta']['data_id'] for c in val}
            assert training_ids.isdisjoint(validation_ids)
            teacher=algorithm.fit(known,TEACHER_REGULARIZATION)
            previous=saved_inner(saved,inner)
            n=len(validation_ids)
            scores['previous'].append((n,algorithm.score(val,previous)))
            scores['prior'].append((n,algorithm.score(val,teacher)))
            updates=[]
            for reg in SHRINKAGES:
                p=fit_regression(target,teacher['parameters'],reg);value=algorithm.score(val,p)
                scores[str(reg)].append((n,value));updates.append({'regularization':reg,'policy':p,'score':value})
            priors.append({'inner_fold':inner,'teacher':teacher,'updates':updates,
                'teacher_training_data_ids':sorted({c['meta']['data_id'] for c in known}),
                'update_training_data_ids':sorted({c['meta']['data_id'] for c in target}),
                'validation_data_ids':sorted(validation_ids)})
        for key,v in scores.items():trials.append({'candidate':key,'innerOOF_R2':sum(n*s for n,s in v)/sum(n for n,s in v)})
        best=max(trials,key=lambda t:t['innerOOF_R2'])
        if best['candidate']=='previous':policy=saved['policy'];teacher=None
        else:
            known=[c for c in source if c['meta']['rho']==0 or c['meta']['family'] not in ('concept','temporal')]
            teacher=algorithm.fit(known,TEACHER_REGULARIZATION)
            target=[c for c in source if c['meta']['rho']==0 or c['meta']['family']=='concept']
            policy=teacher if best['candidate']=='prior' else fit_regression(target,teacher['parameters'],float(best['candidate']))
        atomic_json(folder/f'{method}_policy.json',{'policy':policy,'selected_candidate':best['candidate'],
            'trials':trials,'inner_priors':priors,'final_teacher':teacher,'seen':True,'matched':matched,
            'training_data_ids':sorted({c['meta']['data_id'] for c in source}),
            'excluded_data_ids':sorted({c['meta']['data_id'] for c in bank if c['fold']==fold}),
            'heldout_query_labels_used':False})
        print('TRANSFER_REGRESSION',fold,method,best,flush=True)
    return {'task':'regression','id':fold}


def classification_worker(task_id):
    torch.set_num_threads(1);folder=OUT/'classification'/str(task_id)
    inp=classification.cls.banks.split.OUT/'inputs'/f'{task_id}.npz';meta=json.loads(inp.with_suffix('.json').read_text())
    with np.load(inp) as z:data={k:z[k] for k in z.files}
    data['data_id']=int(meta['data_id'])
    for method,seen,matched in classification.METHODS:
        original=classification.OUT/'classification'/str(task_id)/f'{method}_policy.json';saved=json.loads(original.read_text())
        if not seen:
            saved['frozen_unseen_control']=True;saved['frozen_source_sha256']=classification.origin.sha(original)
            atomic_json(folder/f'{method}_policy.json',saved);continue
        unknown_method='B' if matched else 'D'
        neutral,origin=classification.cls.source_prior(meta,unknown_method)
        unknown=classification.cls.cells_for(task_id,'adapt',matched,False,data)
        target=[c for c in classification.cls.cells_for(task_id,'adapt',matched,True,data) if c['meta']['family'] in ('clean','shortcut')]
        validation=[c for c in classification.cls.cells_for(task_id,'validation',matched,True,data) if c['meta']['family'] in ('clean','shortcut')]
        teacher=classification.fit(unknown,'classification',TEACHER_REGULARIZATION,neutral,'gated')
        old_trial=max(saved['trials'],key=lambda t:(t['validation']['primary'],t['validation']['secondary']))
        candidates=[('previous',old_trial['model']),('prior',teacher)]
        for reg in SHRINKAGES:candidates.append((str(reg),classification.fit(target,'classification',reg,np.asarray(teacher['parameters']),'gated')))
        trials=[{'candidate':name,'policy':p,'score':classification.score(validation,p)} for name,p in candidates]
        best=max(trials,key=lambda t:(t['score']['primary'],t['score']['secondary']))
        if best['candidate']=='previous':policy=saved['policy'];final_teacher=None
        else:
            full_unknown=classification.cls.cells_for(task_id,'refit',matched,False,data)
            full_target=[c for c in classification.cls.cells_for(task_id,'refit',matched,True,data) if c['meta']['family'] in ('clean','shortcut')]
            final_teacher=classification.fit(full_unknown,'classification',TEACHER_REGULARIZATION,neutral,'gated')
            policy=final_teacher if best['candidate']=='prior' else classification.fit(full_target,'classification',float(best['candidate']),np.asarray(final_teacher['parameters']),'gated')
        atomic_json(folder/f'{method}_policy.json',{'policy':policy,'seen':True,'matched':matched,
            'selected_candidate':best['candidate'],'trials':trials,'inner_teacher':teacher,'final_teacher':final_teacher,
            'teacher_prior_origin':origin,'target_calibration_labels':64,'evaluation_labels_used_for_fit':False})
    return {'task':'classification','id':task_id}


def train():
    freeze();tasks=[('classification',int(p.stem)) for p in (classification.cls.banks.split.OUT/'inputs').glob('*.npz')]
    tasks += [('regression',i) for i in range(5)];records=[]
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(classification_worker if t=='classification' else regression_worker,i) for t,i in tasks]
        for f in as_completed(futures):
            result=f.result();records.append(result);print(result,flush=True)
            atomic_json(OUT/'execution.json',{'status':'RUNNING','jobs':records})
    paths=list((OUT/'classification').glob('*/*_policy.json'))+list((OUT/'regression').glob('*/*_policy.json'));assert len(paths)==308
    atomic_json(OUT/'policy_seal.json',{str(p.relative_to(OUT)):classification.origin.sha(p) for p in paths})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE','jobs':records})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train']);p.parse_args();train()
