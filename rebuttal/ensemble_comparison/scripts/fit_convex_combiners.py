"""Numerically stable OOF GES and convex linear stacking.

Calibration losses use the benchmark's metric scale on TRAINING episodes only.
Held-out predictions are convex combinations of raw member outputs; no target
labels or normalization constants enter their prediction mapping.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from scipy.optimize import minimize
from fit_combiners import load_cells,sample_weights
from prepare_data import RUN,atomic_json
from run_tabicl import metrics


def design(cells,task):
    values=[];truth=[]
    for c in cells:
        if task=='classification':values.append(c['bank'][:,np.arange(len(c['y'])),c['y'].astype(int)])
        else:
            scale=(max(float(np.std(c['y'])),1e-12) if c['meta']['track'].startswith('seven') else c['meta']['scale'])
            # A shared offset does not change a simplex-weighted prediction.
            values.append((c['bank']-c['mu'])/scale)
            truth.append((c['y']-c['mu'])/scale)
    return np.concatenate(values,axis=1),np.concatenate(truth) if truth else None,np.concatenate(sample_weights(cells))


def fit_ges(A,y,w,task):
    current=np.zeros(A.shape[1]);counts=np.zeros(6);history=[];prefix=[]
    for i in range(25):
        candidates=(current[None]+A)/(i+1)
        loss=((-np.log(np.clip(candidates,1e-12,1))*w).sum(1) if task=='classification'
              else ((candidates-y)**2*w).sum(1))
        k=int(np.argmin(loss));current+=A[k];counts[k]+=1
        prefix.append(counts.copy()/(i+1));history.append(float(loss[k]))
    best=int(np.argmin(history))
    return prefix[best],{'losses':history,'best_prefix':best+1}


def fit_convex(A,y,w,task,regularization):
    center=np.full(6,1/6)
    if task=='classification':A=np.clip(np.asarray(A,dtype=np.float64),1e-12,1.)
    def loss(beta):
        pred=beta@A
        if task=='classification':
            clipped=np.maximum(pred,np.finfo(np.float64).tiny)
            value=float((-np.log(clipped)*w).sum())
            grad=-(A*(w/clipped)[None]).sum(1)
        else:
            error=pred-y;value=float((error**2*w).sum());grad=2*(A*(w*error)[None]).sum(1)
        return value+regularization*float(np.sum((beta-center)**2)),grad+2*regularization*(beta-center)
    result=minimize(loss,center,jac=True,method='SLSQP',bounds=[(0,1)]*6,
        constraints=[{'type':'eq','fun':lambda b:float(b.sum()-1),'jac':lambda b:np.ones(6)}],
        options={'ftol':1e-10,'maxiter':200})
    if not result.success:
        # A logit parameterization enforces the simplex exactly when SLSQP's
        # line search encounters very confident, nearly-zero probabilities.
        initial=np.log(np.maximum(result.x,1e-6));initial-=initial.mean()
        def logits_objective(z):
            b=np.exp(z-z.max());b/=b.sum();value,gradient=loss(b)
            return value,b*(gradient-float(gradient@b))
        retry=minimize(logits_objective,initial,jac=True,method='L-BFGS-B',bounds=[(-20,20)]*6,
            options={'ftol':1e-11,'gtol':1e-8,'maxiter':600})
        if not retry.success:raise RuntimeError(str(result.message)+'; logit retry: '+str(retry.message))
        beta=np.exp(retry.x-retry.x.max());return beta/beta.sum()
    beta=np.clip(result.x,0,1);beta/=beta.sum()
    return beta


def score(A,y,w,beta,task):
    p=beta@A
    return float((-np.log(np.clip(p,1e-12,1))*w).sum() if task=='classification' else ((p-y)**2*w).sum())


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['five','five_corrected','five_calibrated','five_feature_calibrated','seven_calibrated','seven_calibrated_v2','seven_reimplemented'],default='seven_reimplemented')
    p.add_argument('--wait',action='store_true');args=p.parse_args()
    dimensions=([('classification',72*21),('regression',35*21)] if args.track=='five' else [('classification',35*29),('regression',33*29)])
    if args.track=='five_corrected':dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track in ['five_calibrated','five_feature_calibrated']:dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track in ['seven_calibrated','seven_calibrated_v2']:dimensions=[('regression',1155 if 'specific_reference' in __file__ else 957)]
    for task,expected in dimensions:
        while len(list((RUN/'results/tabicl'/args.track/task).rglob('*.npz')))<expected:
            if not args.wait:raise RuntimeError('incomplete inference bank')
            time.sleep(10)
        cells,source=load_cells(task,args.track)
        assert len(cells)==expected
        for fold in range(5):
            out=RUN/'results/combiners'/args.track/task/f'fold_{fold}_convex.json'
            if out.exists():continue
            train=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
            train_ids={c['meta']['data_id'] for c in train};test_ids={c['meta']['data_id'] for c in test}
            assert train_ids.isdisjoint(test_ids)
            A,y,w=design(train,task);ges,history=fit_ges(A,y,w,task)
            inner=min(c['fold'] for c in train)
            tr=design([c for c in train if c['fold']!=inner],task)
            val=design([c for c in train if c['fold']==inner],task)
            scores=[]
            for reg in [0.,.001,.01,.1]:
                beta=fit_convex(*tr,task,reg);scores.append((score(*val,beta,task),reg))
            best=min(scores)[1];stack=fit_convex(A,y,w,task,best)
            rows=[]
            for c in test:
                for name,beta in [('GES_25_metric_normalized',ges),('Stacking_convex_OOF',stack)]:
                    pred=np.einsum('m,m...->...',beta,c['bank'])
                    assert np.all(pred>=c['bank'].min(0)-1e-8) and np.all(pred<=c['bank'].max(0)+1e-8)
                    scale=(max(float(c['y'].std()),1e-12) if task=='regression' and args.track.startswith('seven') else c['meta']['scale'])
                    rows.append(c['meta']|{'method':name,'outer_fold':fold}|metrics(c['y'],pred,task,scale))
            atomic_json(out,{'status':'COMPLETE','task':task,'fold':fold,'seed':0,'track':args.track,
                'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),'fold_map_source':source,
                'ges_weights':ges.tolist(),'ges':history,'stack_weights':stack.tolist(),
                'inner_validation':scores,'stack_regularization':best,
                'target_labels_used_for_inference':False,'calibration_metric_scale':'training episodes only',
                'rows':rows})
            print('DONE',args.track,task,fold,'GES',ges,'convex stacking',stack,flush=True)


if __name__=='__main__':main()
