"""Dataset-OOF GES and stacking. Outer-test labels never enter fitting."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from prepare_data import ROOT,RUN,atomic_json
from run_tabicl import normalize,metrics


def fold_mapping(task,track='five'):
    if track.startswith('seven'):
        manifest=json.loads((RUN/'configs/seven_reimplementation.json').read_text())['datasets']
        return {int(m['data_id']):int(m['outer_fold']) for m in manifest if m['task']==task},'original seven-point dataset folds; reconstructed operators'
    path=(ROOT/'fair_budget_experiments/limix_2m_tabace/results/oof_paper_final/classification/detail_oof.csv'
          if task=='classification' else ROOT/'PaperResults/failure_recovery_20260913/tabicl/five_targeted_final/regression/detail_oof.csv')
    mapping={}
    with path.open() as f:
        for row in csv.DictReader(f):
            key=int(row['data_id']);value=int(row['outer_fold'])
            if key in mapping and mapping[key]!=value:raise ValueError('conflicting fold')
            mapping[key]=value
    return mapping,str(path.relative_to(ROOT))


def load_cells(task,track='five'):
    mapping,source=fold_mapping(task,track)
    result=[]
    for path in sorted((RUN/'results/tabicl'/track/task).rglob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['data_id'] not in mapping:raise ValueError('missing original outer fold')
        with np.load(path) as raw:
            bank=raw['bootstrap_bank'].astype(np.float64)
            y=raw['y_target']
            mu=float(raw['source_y_mean']);sigma=float(raw['source_y_std'])
        result.append({'path':path,'meta':meta,'fold':mapping[meta['data_id']],
            'bank':bank,'y':y,'mu':mu,'sigma':sigma,'clean_weight':(5. if track.startswith('five') else 7.) if meta['rho']==0 else 1.})
    return result,source


def sample_weights(cells):
    # Equal dataset / family / strength weighting. Clean represents five families.
    totals={}
    for c in cells:totals[c['meta']['data_id']]=totals.get(c['meta']['data_id'],0.)+c['clean_weight']
    return [np.full(len(c['y']),c['clean_weight']/totals[c['meta']['data_id']]/len(c['y'])/len(totals)) for c in cells]


def ges(cells,task,iterations=25):
    values=[];truth=[]
    for c in cells:
        if task=='classification':
            values.append(c['bank'][:,np.arange(len(c['y'])),c['y'].astype(int)])
        else:
            values.append((c['bank']-c['mu'])/c['sigma']);truth.append((c['y']-c['mu'])/c['sigma'])
    matrix=np.concatenate(values,axis=1)
    weights=np.concatenate(sample_weights(cells))
    y=np.concatenate(truth) if truth else None
    running=np.zeros(matrix.shape[1]);counts=np.zeros(6)
    losses=[]
    for i in range(iterations):
        candidates=(running[None,:]+matrix)/(i+1)
        score=(-(np.log(np.clip(candidates,1e-12,1))*weights).sum(1) if task=='classification'
               else (((candidates-y[None,:])**2)*weights).sum(1))
        selected=int(np.argmin(score));running+=matrix[selected];counts[selected]+=1
        losses.append(float(score[selected]))
    best=int(np.argmin(losses))+1
    # Refit selection counts to the best training-loss prefix, as GES normally does.
    running[:]=0;counts[:]=0
    for i in range(best):
        candidates=(running[None,:]+matrix)/(i+1)
        score=(-(np.log(np.clip(candidates,1e-12,1))*weights).sum(1) if task=='classification'
               else (((candidates-y[None,:])**2)*weights).sum(1))
        selected=int(np.argmin(score));running+=matrix[selected];counts[selected]+=1
    return counts/counts.sum(),{'iterations':iterations,'best_prefix':best,'calibration_losses':losses}


def fit_stack(cells,task,regularization):
    weights=sample_weights(cells)
    if task=='regression':
        X=np.concatenate([((c['bank']-c['mu'])/c['sigma']).T for c in cells])
        y=np.concatenate([(c['y']-c['mu'])/c['sigma'] for c in cells]);w=np.concatenate(weights)
        mx=(X*w[:,None]).sum(0);my=(y*w).sum();Xc=X-mx
        beta=np.linalg.solve((Xc*w[:,None]).T@Xc+regularization*np.eye(6),Xc.T@(w*(y-my)))
        return {'beta':beta.tolist(),'intercept':float(my-mx@beta)}
    grouped={}
    for c,w in zip(cells,weights):
        k=c['bank'].shape[-1]
        grouped.setdefault(k,[]).append((np.log(np.clip(c['bank'].transpose(1,2,0),1e-12,1)),c['y'].astype(int),w))
    groups=[]
    for parts in grouped.values():groups.append(tuple(np.concatenate([p[i] for p in parts]) for i in range(3)))
    center=np.full(6,1/6)
    def objective(beta):
        loss=regularization*np.sum((beta-center)**2)
        gradient=2*regularization*(beta-center)
        for X,y,w in groups:
            logits=X@beta;lse=logsumexp(logits,axis=1)
            loss+=float(np.dot(w,lse-logits[np.arange(len(y)),y]))
            p=np.exp(logits-lse[:,None]);p[np.arange(len(y)),y]-=1
            gradient+=np.einsum('nc,ncm,n->m',p,X,w,optimize=True)
        return loss,gradient
    fit=minimize(objective,center,jac=True,method='L-BFGS-B',options={'maxiter':80,'ftol':1e-9})
    if not np.all(np.isfinite(fit.x)):raise ValueError('nonfinite stack coefficients')
    return {'beta':fit.x.tolist(),'intercept':0.,'optimizer_success':bool(fit.success),'message':str(fit.message)}


def stack_predict(c,model,task):
    beta=np.asarray(model['beta'])
    if task=='regression':return (((c['bank']-c['mu'])/c['sigma']).T@beta+model['intercept'])*c['sigma']+c['mu']
    z=np.einsum('m,mnc->nc',beta,np.log(np.clip(c['bank'],1e-12,1)))
    return normalize(np.exp(z-z.max(1,keepdims=True)))


def evaluate(cells,model,task):
    losses=[];weights=sample_weights(cells)
    for c,w in zip(cells,weights):
        pred=stack_predict(c,model,task)
        loss=(-np.log(np.clip(pred[np.arange(len(c['y'])),c['y'].astype(int)],1e-12,1))
              if task=='classification' else ((pred-c['y'])/c['sigma'])**2)
        losses.append(float(w@loss))
    return sum(losses)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true')
    parser.add_argument('--track',choices=['five','five_corrected','five_calibrated','five_feature_calibrated','seven_calibrated','seven_calibrated_v2','seven_reimplemented'],default='five');args=parser.parse_args()
    dimensions=([('classification',72*21),('regression',35*21)] if args.track=='five'
                else [('classification',35*29),('regression',33*29)])
    if args.track=='five_corrected':dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track in ['five_calibrated','five_feature_calibrated']:dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track in ['seven_calibrated','seven_calibrated_v2']:dimensions=[('regression',1155 if 'specific_reference' in __file__ else 957)]
    for task,expected in dimensions:
        root=RUN/'results/tabicl'/args.track/task
        def ready():
            paths=[p for p in root.rglob('*.json') if '.tmp' not in p.name]
            return len(paths)==expected and (task!='regression' or all(json.loads(p.read_text()).get('bootstrap_revision',1)>=2 for p in paths))
        while not ready():
            if not args.wait:raise RuntimeError('incomplete inference bank: '+task)
            time.sleep(10)
        cells,source=load_cells(task,args.track)
        assert len(cells)==expected
        for fold in range(5):
            train=[c for c in cells if c['fold']!=fold]
            test=[c for c in cells if c['fold']==fold]
            train_ids={c['meta']['data_id'] for c in train};test_ids={c['meta']['data_id'] for c in test}
            assert train_ids.isdisjoint(test_ids)
            output=RUN/'results/combiners'/args.track/task/f'fold_{fold}.json'
            if output.exists():continue
            print('FIT',task,fold,'train datasets',len(train_ids),'test datasets',len(test_ids),flush=True)
            weight,ges_meta=ges(train,task)
            inner_fold=min({c['fold'] for c in train})
            inner_train=[c for c in train if c['fold']!=inner_fold]
            inner_val=[c for c in train if c['fold']==inner_fold]
            candidates=[]
            for reg in [.001,.01,.1]:
                candidate=fit_stack(inner_train,task,reg)
                candidates.append((evaluate(inner_val,candidate,task),reg))
            _,best=min(candidates)
            stack=fit_stack(train,task,best)
            rows=[]
            for c in test:
                for name,pred in [('GES_25',np.einsum('m,m...->...',weight,c['bank'])),
                                  ('Stacking_OOF',stack_predict(c,stack,task))]:
                    evaluation_scale=(max(float(np.std(c['y'])),1e-12) if task=='regression' and args.track.startswith('seven') else c['meta']['scale'])
                    rows.append({**c['meta'],'method':name,'outer_fold':fold,
                                 **metrics(c['y'],pred,task,evaluation_scale)})
            atomic_json(output,{'status':'COMPLETE','task':task,'fold':fold,'seed':0,
                'fold_map_source':source,'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),
                'held_out_dataset_labels_used_for_fitting':False,
                'ges_weights':weight.tolist(),'ges':ges_meta,'stack':stack,
                'stack_regularization':best,'inner_validation_scores':candidates,'rows':rows})
            print('DONE',task,fold,flush=True)


if __name__=='__main__':main()
