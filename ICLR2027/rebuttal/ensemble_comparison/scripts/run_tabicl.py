"""Resumable, single-seed TabICL ensemble comparisons on the original five track.

No target labels are passed to inference or ensemble construction. Supervised
combiners are fitted separately, after dataset-level out-of-fold partitioning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier, TabICLRegressor

from prepare_data import RUN, ROOT, atomic_json, protocol

SEED = 0
CHECKPOINTS = ROOT.parent / 'TabSwift/artifacts/checkpoints'


def normalize(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    return p / p.sum(axis=-1, keepdims=True)


def metrics(y, pred, task, scale):
    if task == 'classification':
        p = normalize(pred)
        aucs = [roc_auc_score(y == i, p[:, i]) for i in range(p.shape[1])
                if np.any(y == i) and np.any(y != i)]
        return {'accuracy': float(np.mean(p.argmax(-1) == y)),
                'auc': float(np.mean(aucs)) if aucs else None}
    error = np.asarray(pred) - y
    denominator = float(np.sum((y-y.mean())**2))
    return {'r2': 1-float(np.sum(error**2))/max(denominator, 1e-12),
            'nrmse': float(np.sqrt(np.mean(error**2)))/scale,
            'nmae': float(np.mean(np.abs(error)))/scale}


def preserving_bootstrap(y, rng, size=None):
    size=len(y) if size is None else size
    mandatory = [rng.choice(np.flatnonzero(y == c)) for c in np.unique(y)]
    rows = np.r_[mandatory, rng.choice(len(y), size-len(mandatory), replace=True)]
    rng.shuffle(rows)
    return rows.astype(int)


def dense_groups(y, rng, count=6):
    groups = [[] for _ in range(count)]
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        rng.shuffle(rows)
        for i, part in enumerate(np.array_split(rows, count)):
            groups[i].extend(part.tolist() if len(part) else [int(rng.choice(rows))])
    return [np.asarray(x, dtype=int) for x in groups]


class Predictor:
    def __init__(self, task, device):
        self.task = task
        self.device = device
        self.models = {}

    def __call__(self, X, y, Q, *, members=1, random_state=SEED, n_classes=None, quantiles=False):
        if X.shape[1]==0 or np.all(np.ptp(np.asarray(X,dtype=np.float64),axis=0)==0):
            if self.task=='classification':
                prior=np.bincount(np.asarray(y,dtype=int),minlength=n_classes)/len(y)
                return normalize(np.broadcast_to(prior,(len(Q),n_classes)).copy()),None
            spread=float(np.quantile(y,.85)-np.quantile(y,.15))
            return np.full(len(Q),float(np.mean(y))),np.full(len(Q),spread) if quantiles else None
        if self.task == 'classification' and len(np.unique(y)) == 1:
            out = np.zeros((len(Q), n_classes))
            out[:, int(y[0])] = 1
            return out, None
        if self.task == 'regression' and float(np.std(y)) < 1e-12:
            return np.full(len(Q), np.mean(y)), np.zeros(len(Q))
        if members not in self.models:
            cls = TabICLClassifier if self.task == 'classification' else TabICLRegressor
            name = f'tabicl-{ "classifier" if self.task == "classification" else "regressor"}-v2-20260212.ckpt'
            self.models[members] = cls(n_estimators=members, device=self.device,
                model_path=str(CHECKPOINTS/name), allow_auto_download=False,
                n_jobs=1, batch_size=1, random_state=random_state)
        model = self.models[members]
        model.random_state = random_state
        model.fit(np.asarray(X, dtype=np.float32), y)
        # Reuse exactly the loaded frozen weights; do not reload 110 MB on each fit.
        if not getattr(model, '_weights_cached_for_rebuttal', False):
            model._load_model = lambda: None
            model._weights_cached_for_rebuttal = True
        if self.task == 'classification':
            pred = model.predict_proba(np.asarray(Q, dtype=np.float32))
            out = np.zeros((len(Q), n_classes), dtype=np.float64)
            out[:, np.asarray(model.classes_, dtype=int)] = pred
            return normalize(out), None
        if quantiles:
            out = model.predict(np.asarray(Q, dtype=np.float32),
                                output_type=['mean', 'quantiles'], alphas=[.15, .85])
            return np.asarray(out['mean'], dtype=np.float64), np.maximum(
                out['quantiles'][:, 1]-out['quantiles'][:, 0], 0)
        return np.asarray(model.predict(np.asarray(Q, dtype=np.float32)), dtype=np.float64), None


def run_cell(predictor, data, meta, family, rho, output):
    task = meta['task']
    X0, y0, Q, truth = (data[k] for k in ['X_source','y_source','X_target','y_target'])
    if str(meta.get('track','')).startswith('seven') or meta.get('condition_materialized',False):
        X,y=X0,y0
    else:
        changed = protocol.manipulate_source_context(task, X0, y0, family=family,
            rho=rho/100, seed=1729+rho, shift_scale=6.0 if task=='classification' else 10.0,
            label_scale=1.0)
        X, y = changed.X_context, changed.y_context
    pool_X=data['X_pool'] if 'X_pool' in data else X
    pool_y=data['y_pool'] if 'y_pool' in data else y
    n_classes = meta.get('n_classes',2 if str(meta.get('track','')).startswith('seven') else len(np.unique(y0))) if task == 'classification' else None
    predictions, summaries, arrays = {}, {}, {}
    call_times = {}

    def predict(name, ctx_X, ctx_y, *, members=1, quantiles=False):
        torch.cuda.synchronize()
        start = time.perf_counter()
        pred, uncertainty = predictor(ctx_X, ctx_y, Q, members=members,
              random_state=SEED, n_classes=n_classes, quantiles=quantiles)
        torch.cuda.synchronize()
        call_times[name] = time.perf_counter()-start
        return pred, uncertainty

    # Base is solely a protocol check/anchor and is never tuned toward the paper score.
    pred, _ = predict('base_protocol_check', X, y)
    predictions['base_protocol_check'] = pred
    for count in meta.get('native_ensemble_sizes',[6,8]):
        pred, _ = predict(f'native_{count}', X, y, members=count)
        predictions[f'native_{count}'] = pred

    bootstrap, iqrs = [], []
    for i in range(6):
        rng=np.random.default_rng(SEED+1009*(i+1))
        rows = (preserving_bootstrap(pool_y,rng,size=len(y)) if task=='classification'
                else rng.choice(len(pool_y),size=len(y),replace=True))
        pred, uncertainty = predict(f'bootstrap_member_{i}', pool_X[rows], pool_y[rows], quantiles=task=='regression')
        bootstrap.append(pred)
        if uncertainty is not None: iqrs.append(uncertainty)
    bank = np.stack(bootstrap)
    arrays['bootstrap_bank'] = bank
    predictions['bootstrap_mean_6'] = bank.mean(0)
    call_times['bootstrap_mean_6'] = sum(call_times[f'bootstrap_member_{i}'] for i in range(6))
    if task == 'regression':
        iqrs = np.stack(iqrs)
        arrays['bootstrap_iqr'] = iqrs
        selected = int(np.argmin(iqrs.mean(axis=1)))
        # Original 10% retained fraction, restricted to six candidates: keep one.
        predictions['oligoicp_iqr_6_keep1_adapted'] = bank[selected]
        call_times['oligoicp_iqr_6_keep1_adapted'] = call_times['bootstrap_mean_6']

    if task == 'classification':
        groups = dense_groups(y, np.random.default_rng(SEED+2718))
        dense, similarity = [], []
        loc, scale = np.mean(X, axis=0), np.maximum(np.std(X, axis=0), 1e-6)
        q = (Q-loc)/scale
        qnorm = np.maximum(np.linalg.norm(q, axis=1), 1e-12)
        for i, rows in enumerate(groups):
            pred, _ = predict(f'dense_member_{i}', X[rows], y[rows])
            dense.append(pred)
            z = (X[rows]-loc)/scale
            znorm = np.maximum(np.linalg.norm(z, axis=1), 1e-12)
            cos = (q @ (z/znorm[:,None]).T)/qnorm[:,None]
            similarity.append(np.maximum(cos.mean(1), 0)+1e-6)
        dense = np.stack(dense)
        w = np.stack(similarity)
        w /= w.sum(0,keepdims=True)
        arrays['dense_bank'], arrays['dense_weights'] = dense, w
        predictions['dense_moe_adapted'] = normalize((dense*w[:,:,None]).sum(0))
        logp = (np.log(np.clip(dense,1e-12,1))*w[:,:,None]).sum(0)
        predictions['dense_poe_adapted'] = normalize(np.exp(logp-logp.max(1,keepdims=True)))
        predictions['dense_max_adapted'] = normalize((dense*w[:,:,None]).max(0))
        dense_time = sum(call_times[f'dense_member_{i}'] for i in range(6))
        for name in ['dense_moe_adapted','dense_poe_adapted','dense_max_adapted']:
            call_times[name] = dense_time

    # Test-time feature divide-and-conquer; unchanged low-dimensional views are reused.
    columns = X.shape[1]
    count = min(6, max(1, int(np.ceil(columns/500))))
    if count == 1:
        predictions['divide_conquer_feature_adapted'] = predictions['base_protocol_check']
        call_times['divide_conquer_feature_adapted'] = call_times['base_protocol_check']
    else:
        views=[]
        for i in range(count):
            ids=np.random.default_rng(SEED+5003+i).choice(columns,min(columns,500),replace=False)
            # Predictor accepts a different query projection for this branch.
            torch.cuda.synchronize();start=time.perf_counter()
            value,_=predictor(X[:,ids],y,Q[:,ids],n_classes=n_classes)
            torch.cuda.synchronize();call_times[f'divide_conquer_member_{i}']=time.perf_counter()-start
            views.append(value)
        predictions['divide_conquer_feature_adapted']=np.mean(views,axis=0)
        call_times['divide_conquer_feature_adapted']=sum(call_times[f'divide_conquer_member_{i}'] for i in range(count))

    evaluation_scale=(max(float(np.std(truth)),1e-12) if task=='regression' and str(meta.get('track','')).startswith('seven') else meta['scale'])
    for name,pred in predictions.items():
        if not np.all(np.isfinite(pred)): raise ValueError(f'nonfinite {name}')
        summaries[name] = {**metrics(truth,pred,task,evaluation_scale), 'seconds':call_times[name]}
        arrays['prediction__'+name] = pred
    arrays['y_target'] = truth
    arrays['source_y_mean'] = np.asarray(float(np.mean(y)))
    arrays['source_y_std'] = np.asarray(max(float(np.std(y)),1e-12))
    arrays['metadata_json'] = np.asarray(json.dumps({**meta,'family':family,'rho':rho,'seed':SEED}))
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays);temp.replace(output)
    atomic_json(output.with_suffix('.json'),{**meta,'family':family,'rho':rho,'seed':SEED,
        'status':'COMPLETE','methods':summaries,'feature_dac_members':count,'bootstrap_revision':2,
        'held_out_labels_used_for_fitting':False,'adaptation_training':False,
        'notes':['single seed','DENSE uses class-covering groups and tabular cosine similarity',
                 'D&C uses soft-vote probabilities for AUROC; low-dimensional view equals Base']})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--task',choices=['classification','regression','both'],default='both')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(args.gpu)
    predictors={}
    jobs=[]
    for task,suite in [('classification',99),('regression',353)]:
        if args.task not in ['both',task]:continue
        raw=json.loads((RUN/f'cache/suite_{suite}.json').read_text())['study']['tasks']['task_id']
        jobs.extend((task,int(i)) for i in raw)
    jobs=jobs[args.shard::args.shards]
    if args.limit:jobs=jobs[:args.limit]
    failures=0
    for task,task_id in jobs:
        source=RUN/'cache/five'/task/f'{task_id}.npz'
        waited=0
        while not source.exists() and waited<1800:
            if (RUN/'results/preparation_failures'/f'{task}_{task_id}.json').exists():break
            time.sleep(5);waited+=5
        if not source.exists():
            print('MISSING_DATA',task,task_id,flush=True);failures+=1;continue
        meta=json.loads(source.with_suffix('.json').read_text())
        conditions=[('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        if args.smoke:conditions=conditions[:2]
        with np.load(source) as data:
            for family,rho in conditions:
                output=RUN/'results/tabicl/five'/task/str(task_id)/f'{family}_{rho}.npz'
                if output.exists() and output.with_suffix('.json').exists():continue
                print('START',task,task_id,meta['dataset'],family,rho,flush=True)
                try:
                    if task not in predictors:predictors[task]=Predictor(task,f'cuda:{args.gpu}')
                    run_cell(predictors[task],data,meta,family,rho,output)
                    print('DONE',task,task_id,family,rho,flush=True)
                except Exception as e:
                    failures+=1
                    row={'task':task,'task_id':task_id,'family':family,'rho':rho,
                         'error':repr(e),'traceback':traceback.format_exc()}
                    atomic_json(RUN/'results/inference_failures'/f'tabicl_{task}_{task_id}_{family}_{rho}.json',row)
                    print('FAILED',json.dumps(row),flush=True)
                    torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
