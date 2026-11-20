"""Observed-only / limited-reference TabACE on immutable Table-28 inputs.

Reference treatment replaces <=20% rows in five registered Shared views. Specific
inherits these views and uses the five cached family-directed observed views.
No benchmark edits, target-label fitting, target-score or paper-number selection.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

import finetune_tabace_main_v1 as main
import finetune_source_tabace_v5 as source
import finetune_tabace_reference_v3 as references
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor, metrics

OUT = RUN/'results/paperish_fixed_benchmark_reference020_v25'
RAW = RUN/'results/pretrained_tabace_rerun_v21'
FT = RUN/'results/finetune_final_selected_v13'
RATE = 0.20


def rel(job):
    return Path('jobs')/job['task']/job['track']/str(job['meta']['task_id'])


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'plan.json'
    if path.exists():
        return
    jobs = main.jobs()
    folds = {}
    for task in ('classification', 'regression'):
        ids = sorted({int(j['meta']['data_id']) for j in jobs if j['task'] == task})
        folds[task] = {str(data_id): i % 5 for i, data_id in enumerate(ids)}
    atomic_json(path, dict(
        status='FROZEN_BEFORE_NEW_INFERENCE_AND_QUERY_SCORING', seed=0,
        benchmark_frozen=True, benchmark_hash=main.sha(main.frozen.CONFIG),
        classification_five_hash=main.sha(RUN/'results/finetune_source_benchmark_v5/frozen.json'),
        source_cache=str(RAW), reference_fraction_per_view_upper_bound=RATE,
        source_reference_pool_cap=1024, per_context_cap=256,
        Shared_alternatives=5, Specific_alternatives=10,
        Specific_pool_scope='five recovered family-directed views plus all five Shared views; original complete selected lists not recovered',
        variants=['observed_only', 'reference020'],
        unchanged_original_gate=True, classification_gate='normalized Base entropy',
        regression_gate='Base-aggregate discrepancy times alternative agreement',
        extra_gain=1.0, regularization=0.01,
        calibration='five-fold dataset-level OOF, excluding every held-out data_id',
        calibration_strengths_available=[0.0,0.5,1.0], evaluation_strengths=[0.2,0.5,0.8,1.0],
        paper_values_not_used_for_selection=True, final_query_labels_used_for_selection=False,
        exact_archived_main_table_reproduction=False,
        semisynthetic_regression_blocks=['clean','five'],
        folds=folds, start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        jobs=len(jobs), stage='controlled replay on fixed development benchmark'))


def read_cell(path, job, role, light=False):
    meta = json.loads(path.with_suffix('.json').read_text())
    with np.load(path) as z:
        c = {k:z[k] for k in ['base','bank','y','mu','sigma']}
    if light:
        other = OUT/rel(job)/path.relative_to(RAW/rel(job))
        with np.load(other) as z:
            c['bank'] = c['bank'].copy()
            c['bank'][:5] = z['members']
    return dict(base=c['base'], bank=c['bank'], y=c['y'],
                mu=float(c['mu']), sigma=max(float(c['sigma']),1e-8),
                task=job['task'], role=role, meta=meta|dict(data_id=job['meta']['data_id']))


def replace_rows(C, d, A, b, task, branch):
    rng = np.random.default_rng(1729+branch)
    n = int(np.floor(RATE*len(d)))
    protected = set()
    if task == 'classification':
        protected = {int(np.flatnonzero(d == label)[0]) for label in np.unique(d)}
    eligible = np.array([i for i in range(len(d)) if i not in protected], dtype=int)
    take = rng.permutation(eligible)[:n]
    ref = []
    for index in take:
        pool = np.flatnonzero(b == d[index]) if task == 'classification' else np.arange(len(b))
        if not len(pool):
            pool = np.arange(len(b))
        ref.append(int(rng.choice(pool)))
    X, y = C.copy(), d.copy()
    X[take], y[take] = A[ref], b[ref]
    assert len(y) == len(d) <= 256 and len(take)/len(d) <= RATE+1e-12
    return X, y, len(take)


def make_light(job, role, block, family, rho, predictor, pool, split, ref):
    sub = Path('calibration') if role == 'validation' else Path('query')/block
    target = OUT/rel(job)/sub/f'{family}_{rho}.npz'
    if target.exists() and target.with_suffix('.json').exists():
        return
    old_path = RAW/rel(job)/sub/f'{family}_{rho}.npz'
    meta = json.loads(old_path.with_suffix('.json').read_text())
    if role == 'query':
        data, _ = source.bank(job['task'],block,job['meta'],family,rho)
    else:
        data = source.materialize(job,pool,np.r_[split['train'],split['val']],split['cal'],family,rho)
    assert meta['input_hashes'] == {k:protocol.array_sha256(v) for k,v in data.items()}
    with np.load(old_path) as z:
        base = z['base']
    X, y, Q = (data[k] for k in ['X_source','y_source','X_target'])
    views = main.contexts.shared_contexts(X,y,Q,len(y),job['task'],base)
    A, b, ids = ref
    _, _, center, scale = main.contexts.shared.geometry(X,Q)
    target_center = np.median(Q,axis=0)
    target_scale = np.maximum(np.percentile(Q,75,axis=0)-np.percentile(Q,25,axis=0),1e-6)
    aligned_A = (A-center)/scale*target_scale+target_center
    predictions, counts, hashes = [], [], []
    started = time.monotonic()
    for branch, (name,C,d,V) in enumerate(views):
        clean = A
        if name == 'feature_subspace':
            top = int(main.contexts.specific.importance(X,y.astype(float))[0])
            cols = [j for j in range(Q.shape[1]) if j != top] or [0]
            assert np.array_equal(Q[:,cols],V)
            clean = A[:,cols]
        elif name in ('distribution_aligned','aligned_target_matched'):
            clean = aligned_A
        changed, labels, count = replace_rows(C,d,clean,b,job['task'],branch)
        prediction, _ = predictor(changed,labels,V,members=1,random_state=0,
                                  n_classes=base.shape[1] if job['task']=='classification' else None)
        assert np.isfinite(prediction).all()
        predictions.append(prediction)
        counts.append(dict(role=name,rows=len(labels),clean_reference_rows=count,
                           fraction=count/len(labels)))
        hashes.append([protocol.array_sha256(a) for a in (changed,labels,V)])
    main.scaling.npwrite(target,members=np.stack(predictions))
    atomic_json(target.with_suffix('.json'), dict(
        input_hashes=meta['input_hashes'], member_input_hashes=hashes, roles=counts,
        reference_original_ids_hash=protocol.array_sha256(ids),
        reference_rows_disjoint_query_and_calibration=True, native_members=1,
        frozen_pretrained_weights=True, seconds=time.monotonic()-started))


def inference_worker(shard, limit=0):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    jobs = main.jobs()[shard::8]
    if limit:
        jobs = jobs[:limit]
    predictors = {}
    for index,job in enumerate(jobs):
        dest = OUT/rel(job)
        if (dest/'inference_complete.json').exists():
            continue
        try:
            pool = main.scaling.prepare(job['task'],job['track'],job['meta'])
            with np.load(FT/rel(job)/'split.npz') as z:
                split = {k:z[k] for k in z.files}
            ref = references.references(job,pool,split)
            if job['task'] not in predictors:
                predictors[job['task']] = Predictor(job['task'],'cuda:0')
            predictor = predictors[job['task']]
            for family,rho in [('original',0)]+[(f,r) for f in main.families(job['track']) for r in [50,100]]:
                block = 'clean' if rho == 0 else job['track']
                make_light(job,'validation',block,family,rho,predictor,pool,split,ref)
            for block in (['clean','five'] if job['track']=='five' else ['seven']):
                for family,rho in main.contexts.case_grid(block):
                    make_light(job,'query',block,family,rho,predictor,pool,split,ref)
            atomic_json(dest/'inference_complete.json',dict(task=job['task'],track=job['track'],task_id=job['meta']['task_id']))
            print('INFERENCE_COMPLETE',shard,index+1,len(jobs),job['task'],job['track'],job['meta']['task_id'],flush=True)
        except Exception:
            error = traceback.format_exc()
            atomic_json(dest/'error.json',dict(error=error))
            print('ERROR',shard,error,flush=True)
    print('WORKER_DONE',shard,flush=True)


def arrays(cells, m, device):
    parts, weights = {}, []
    counts = {}
    for c in cells:
        counts[c['meta']['data_id']] = counts.get(c['meta']['data_id'],0)+1
    for c in cells:
        bank, base = c['bank'][:m], c['base']
        if c['task']=='classification':
            p = np.clip(bank,1e-12,1.)
            Q = 1+np.sum(p*np.log(p),axis=2)/np.log(max(2,base.shape[1]))
            S = 1-.5*np.sum(np.abs(bank-bank.mean(0)[None]),axis=2)
            pb = np.clip(base,1e-12,1.)
            U = -np.sum(pb*np.log(pb),axis=1)/np.log(max(2,base.shape[1]))
            y = c['y'].astype(int)
            item=dict(Q=Q.T,S=S.T,U=U,
                      T=bank[:,np.arange(len(y)),y].T,B=base[np.arange(len(y)),y])
            factor=1.
        else:
            B=(base-c['mu'])/c['sigma']
            M=(bank-c['mu'])/c['sigma']
            item=dict(M=M.T,B=B,D=np.abs(M-np.median(M,axis=0)[None]).T,Y=(c['y']-c['mu'])/c['sigma'])
            factor=(c['sigma']/c['meta']['scale'])**2
        for k,v in item.items():
            parts.setdefault(k,[]).append(v)
        weights.append(np.full(len(c['y']),factor/(len(counts)*counts[c['meta']['data_id']]*len(c['y']))))
    return {k:torch.as_tensor(np.concatenate(v),dtype=torch.float64,device=device) for k,v in parts.items()}|dict(weights=torch.as_tensor(np.concatenate(weights),dtype=torch.float64,device=device))


def fit(cells,task,m,device):
    d=arrays(cells,m,device)
    initial=np.r_[np.zeros(m),np.ones(2*m),.5] if task=='classification' else np.r_[np.zeros(m),np.ones(m),0.,.5]
    bounds=[(-8,8)]*m+([(0,10)]*(2*m)+[(0,1)] if task=='classification' else [(0,10)]*m+[(-4,4),(0,1)])
    def objective(x):
        p=torch.tensor(x,dtype=torch.float64,device=device,requires_grad=True)
        if task=='classification':
            w=torch.softmax(p[:m]+d['Q']*p[m:2*m]+d['S']*p[2*m:3*m],dim=1)
            alpha=p[-1]*d['U']*(w*d['Q']*d['S']).sum(1)
            out=(1-alpha)*d['B']+alpha*(w*d['T']).sum(1)
            raw=-torch.log(out.clamp(1e-12,1.))
            coef=p[m:3*m]
        else:
            tau=torch.exp(p[-2]);s=torch.exp(-d['D']/tau)
            w=torch.softmax(p[:m]+s*p[m:2*m],dim=1)
            avg=(w*d['M']).sum(1)
            alpha=p[-1]*(1-torch.exp(-torch.abs(avg-d['B'])/tau))*(w*s).sum(1)
            out=(1-alpha)*d['B']+alpha*avg
            raw=(out-d['Y'])**2;coef=p[m:2*m]
        loss=(raw*d['weights']).sum()+.01*((p[:m]-p[:m].mean()).square().mean()+(coef-1).square().mean())
        loss.backward()
        return float(loss.detach()),p.grad.detach().cpu().numpy()
    result=minimize(objective,initial,method='L-BFGS-B',jac=True,bounds=bounds,
                    options=dict(maxiter=120,ftol=1e-9,gtol=1e-6))
    assert np.isfinite(result.x).all()
    return dict(kind='gated_fusion',m=m,gain=1.,regularization=.01,parameters=result.x.tolist(),gate_variant='entropy',
                optimization=dict(success=bool(result.success),iterations=int(result.nit),loss=float(result.fun)))


def load_cells(task, variant, role):
    cells=[]
    for job in [j for j in main.jobs() if j['task']==task]:
        root=RAW/rel(job)
        paths=sorted((root/'calibration').glob('*.npz')) if role=='validation' else sorted((root/'query').glob('*/*.npz'))
        for p in paths:
            c=read_cell(p,job,role,variant=='reference020')
            c['path']=str(p);c['group']=(job['track'],c['meta']['family'])
            cells.append(c)
    return cells


def fit_worker(task, variant):
    torch.set_num_threads(1)
    import tabace_fusion_v28 as fusion
    cal=load_cells(task,variant,'validation')
    queries=load_cells(task,variant,'query')
    folds=json.loads((OUT/'plan.json').read_text())['folds'][task]
    rows=[]
    for fold in range(5):
        train=[c for c in cal if folds[str(c['meta']['data_id'])]!=fold]
        test=[c for c in queries if folds[str(c['meta']['data_id'])]==fold]
        assert not ({c['meta']['data_id'] for c in train}&{c['meta']['data_id'] for c in test})
        shared=fit(train,task,5,'cuda:0')
        specific={}
        for group in sorted({c['group'] for c in test}):
            group_cal=[c for c in train if c['group']==group]
            specific['/'.join(group)]=fit(group_cal,task,10,'cuda:0')
        dest=OUT/'policies'/variant/task/f'fold{fold}'
        atomic_json(dest/'policies.json',dict(Shared=shared,Specific=specific,
                    outer_train_data_ids=sorted({c['meta']['data_id'] for c in train}),
                    outer_test_data_ids=sorted({c['meta']['data_id'] for c in test}),
                    final_query_labels_used_for_fit=False))
        for c in test:
            preds=dict(Base=c['base'],Shared=fusion.predict(c,shared),Specific=fusion.predict(c,specific['/'.join(c['group'])]))
            for name,p in preds.items():
                row={k:c['meta'][k] for k in ['task','block','task_id','dataset','family','rho']}
                rows.append(row|dict(data_id=c['meta']['data_id'],fold=fold,variant=variant,method=name,
                                     **metrics(c['y'],p,task,c['meta']['scale'])))
        pd.DataFrame(rows).to_csv(OUT/f'progress_{variant}_{task}.csv',index=False)
        print('FOLD_COMPLETE',variant,task,fold,'conditions',len(test),flush=True)
    detail=pd.DataFrame(rows)
    detail.to_csv(OUT/f'detail_{variant}_{task}.csv',index=False)
    avg=detail.groupby(['task','block','variant','method'],as_index=False).agg(
        datasets=('data_id','nunique'),conditions=('task_id','size'),
        **{k:(k,'mean') for k in (['accuracy','auc'] if task=='classification' else ['r2','nrmse'])})
    avg.to_csv(OUT/f'averages_{variant}_{task}.csv',index=False)
    atomic_json(OUT/f'complete_{variant}_{task}.json',dict(status='COMPLETE',datasets=int(detail.data_id.nunique()),
                 conditions=len(detail)//3,OOF_dataset_disjoint=True,benchmark_unchanged=True))
    print(avg.to_string(index=False),flush=True)


def launch(phase):
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    jobs=[]
    for gpu in range(8):
        if phase=='inference':
            args=['inference','--shard',str(gpu)]
        else:
            if gpu>=4:
                continue
            args=['fit','--task',('classification' if gpu%2==0 else 'regression'),
                  '--variant',('observed_only' if gpu<2 else 'reference020')]
        logfile=OUT/'logs'/f'{phase}_{gpu}.log'
        with logfile.open('a',buffering=1) as log:
            p=subprocess.Popen([sys.executable,'-u',__file__,*args],env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(dict(gpu=gpu,pid=p.pid,log=str(logfile)))
    atomic_json(OUT/f'{phase}_workers.json',jobs)
    print(json.dumps(jobs),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['declare','inference','fit','launch-inference','launch-fit'])
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--task',choices=['classification','regression'])
    parser.add_argument('--variant',choices=['observed_only','reference020'])
    args=parser.parse_args()
    declare()
    if args.phase=='inference': inference_worker(args.shard,args.limit)
    elif args.phase=='fit': fit_worker(args.task,args.variant)
    elif args.phase.startswith('launch-'): launch(args.phase[7:])
