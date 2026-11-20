"""AUC-aligned calibration of the frozen strict-control v92 prediction bank.

Only calibration loss/selection changes from the v94 agreement gate. No new
context, feature suppression, raw-feature predictor, backbone training, or
benchmark modification. Outer-test outcomes never select a parameter.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

import semisynthetic_abcd_v92 as source
import tabace_fusion_v28 as fusion
from semisynthetic_feature_adapter_v93 import predict
from prepare_data import RUN,atomic_json
from run_tabicl import metrics

OUT=RUN/'results/semisynthetic_ranking_calibration_v96'
PAIRS_PER_CLASS=16


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text())['code_sha256']==source.original.sha(__file__)
        return
    atomic_json(p,{'status':'FROZEN_BEFORE_RANKING_CALIBRATION','seed':0,
        'fixed_benchmark_and_Base':source.selection(),'fixed_context_bank':'v92 density_adapted',
        'context_cap':256,'alternative_branches':5,'strict_no_feature_suppression_in_C_D':True,
        'frozen_forward_mapping':'v94 agreement-only evidence and entropy-or-disagreement gate',
        'calibration_loss':'classwise positive-negative logit-score pair logistic loss, macro dataset/condition/class weighting',
        'sampled_pairs_per_class_per_condition':PAIRS_PER_CLASS,
        'regularization_grid':source.original.REGS,
        'inner_selection':'macro dataset AUROC; identity Base fallback if no improvement',
        'outer_dataset_folds':5,'B_D_excludes_all_positive_five_source_perturbations':True,
        'own_dataset_query_labels_used_for_fit_or_selection':False,
        'new_raw_feature_supervised_head':False,'new_backbone_training':False,
        'new_constructor_or_benchmark':False,'calibration_adaptation_dagger':True,
        'all_validation_trials_and_Base_fallback_retained':True,
        'prior_development_outcomes_motivated_ranking_loss':True,
        'original_paper_replication':False,'independent_confirmation':False,
        'target_labels_used_for_declared_semisynthetic_generation':True,
        'target_labels_received_by_predictor_or_constructor':False,
        'statistical_family':'24 exploratory A-D comparisons including previous versions and Concept',
        'code_sha256':source.original.sha(__file__),'Markdown_written':False})


def pair_data(cells):
    assert all(c['role']=='validation' for c in cells)
    eligible=[c for c in cells if len(np.unique(c['y']))>1]
    counts={}
    for c in eligible:counts[c['meta']['data_id']]=counts.get(c['meta']['data_id'],0)+1
    rng=np.random.default_rng(0);values={};weights=[]
    def append(name,x):values.setdefault(name,[]).append(x)
    for c in eligible:
        y=c['y'].astype(int);features=fusion.features(c,5,'entropy_or_disagreement')
        labels=np.unique(y)
        for label in labels:
            positive=rng.choice(np.flatnonzero(y==label),PAIRS_PER_CLASS,replace=True)
            negative=rng.choice(np.flatnonzero(y!=label),PAIRS_PER_CLASS,replace=True)
            for side,ids in [('positive',positive),('negative',negative)]:
                for key in ('Q','S','U'):append(side+'_'+key,features[key][ids])
                append(side+'_members',c['bank'][:,ids,label].T)
                append(side+'_base',c['base'][ids,label])
            weight=1./(len(counts)*counts[c['meta']['data_id']]*len(labels)*PAIRS_PER_CLASS)
            weights.append(np.full(PAIRS_PER_CLASS,weight))
    tensors={k:torch.as_tensor(np.concatenate(v),dtype=torch.float64) for k,v in values.items()}
    tensors['weights']=torch.as_tensor(np.concatenate(weights),dtype=torch.float64)
    assert abs(float(tensors['weights'].sum())-1.)<1e-10
    return tensors


def mapped_probability(parameters,data,side):
    q,s,u=(data[side+'_'+k] for k in ('Q','S','U'))
    w=torch.softmax(parameters[:5]+q*parameters[5:10]+s*parameters[10:15],dim=1)
    mixed=(w*data[side+'_members']).sum(1)
    alpha=parameters[-1]*u*(w*s).sum(1)
    return (1-alpha)*data[side+'_base']+alpha*mixed


def fit(cells,regularization):
    data=pair_data(cells)
    def objective(values):
        p=torch.tensor(values,dtype=torch.float64,requires_grad=True)
        positive=mapped_probability(p,data,'positive').clamp(1e-7,1-1e-7)
        negative=mapped_probability(p,data,'negative').clamp(1e-7,1-1e-7)
        delta=torch.logit(positive)-torch.logit(negative)
        penalty=(p[:5]-p[:5].mean()).square().mean()+(p[5:15]-1).square().mean()
        loss=(torch.nn.functional.softplus(-delta)*data['weights']).sum()+regularization*penalty
        loss.backward();return float(loss.detach()),p.grad.detach().numpy()
    initial=np.r_[np.zeros(5),np.ones(10),.5]
    result=minimize(objective,initial,method='L-BFGS-B',jac=True,
        bounds=[(-8,8)]*5+[(-10,10)]*10+[(0,1)],options={'maxiter':220,'ftol':1e-9,'gtol':1e-6})
    assert np.isfinite(result.x).all()
    return {'kind':'agreement_gate_adapted','parameters':result.x.tolist(),'regularization':regularization,
        'gate_variant':'entropy_or_disagreement','calibration_objective':'sampled_pairwise_ranking',
        'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
        'training_objective':float(result.fun),'sampled_pairs':len(data['weights'])}


def auc_score(cells,model):
    datasets={}
    for c in cells:
        value=metrics(c['y'],predict(c,model),'classification',1.)['auc']
        if value is not None:datasets.setdefault(c['meta']['data_id'],[]).append(value)
    return float(np.mean([np.mean(v) for v in datasets.values()]))


def worker(fold):
    torch.set_num_threads(1);torch.manual_seed(0)
    folder=OUT/'folds/classification'/str(fold);rows=[]
    for method,seen,matched in [('A',True,True),('B',False,True),('C',True,False),('D',False,False)]:
        cells=source.load(matched,'density_adapted')
        train=[c for c in cells if c['fold']!=fold and (seen or not source.original.exposed(c,'classification'))]
        test=[c for c in cells if c['fold']==fold and c['meta']['track']=='five_semisynthetic']
        train_ids={c['meta']['data_id'] for c in train};test_ids={c['meta']['data_id'] for c in test}
        assert train_ids.isdisjoint(test_ids)
        assert seen or not any(source.original.exposed(c,'classification') for c in train)
        inner=min(c['fold'] for c in train)
        first=[c for c in train if c['fold']!=inner];validation=[c for c in train if c['fold']==inner]
        baseline={'kind':'base'};best_score=auc_score(validation,baseline);selected_reg=None
        trials=[{'candidate':'Base','inner_auc':best_score}]
        for reg in source.original.REGS:
            model=fit(first,reg);value=auc_score(validation,model)
            trials.append({'candidate':'ranking','regularization':reg,'inner_auc':value,'inner_model':model})
            if value>best_score+1e-12:best_score=value;selected_reg=reg
        policy=baseline if selected_reg is None else fit(train,selected_reg)
        atomic_json(folder/f'{method}_policy.json',{'method':method,'seen':seen,'matched':matched,
            'training_data_ids':sorted(train_ids),'excluded_data_ids':sorted(test_ids),'inner_fold':inner,
            'trials':trials,'selected_regularization':selected_reg,'selected_inner_auc':best_score,'policy':policy,
            'unseen_mechanism_audit':seen or not any(source.original.exposed(c,'classification') for c in train),
            'outer_test_metrics_used_for_selection':False})
        for c in test:
            m=c['meta'];predictions={method:predict(c,policy)}
            if method=='A':predictions.update({'Base':c['base'],'Single-matched':c['bank'][3]})
            for name,p in predictions.items():
                path=folder/'predictions'/str(m['task_id'])/f'{m["rho"]}_{name}.npz'
                path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prediction=p,y=c['y'])
                rows.append({'task':'classification','version':'ranking_selected','method':name,'data_id':m['data_id'],
                    'task_id':m['task_id'],'dataset':m['dataset'],'rho':m['rho'],'fold':fold,**metrics(c['y'],p,'classification',1.)})
        print('RANKING_POLICY',fold,method,selected_reg,best_score,flush=True)
    pd.DataFrame(rows).to_csv(folder/'detail.csv',index=False)
    atomic_json(folder/'completion.json',{'status':'COMPLETE','rows':len(rows)})


def launch():
    freeze()
    def execute(fold):
        log=OUT/f'worker_{fold}.log'
        with log.open('a') as handle:
            p=subprocess.run([sys.executable,__file__,'worker','--fold',str(fold)],stdout=handle,stderr=subprocess.STDOUT)
        return {'fold':fold,'returncode':p.returncode,'log':str(log)}
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(execute,i) for i in range(5)]):
            r=f.result();records.append(r);print(r,flush=True)
            atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':records})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if all(r['returncode']==0 for r in records) else 'FAILED','workers':records})
    if any(r['returncode'] for r in records):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--fold',type=int)
    a=p.parse_args()
    if a.phase=='worker':worker(a.fold)
    else:launch()
