"""A separate observed-source adaptation registry; immutable v28 Base and banks.

Candidate construction never receives final query labels or actual severity.
Shared uses ten roles. Specific retains all ten, adds the original five roles
and six family-aware roles. Every TFM branch uses at most the paired row budget.
Clean TRAIN holdout labels are used only later for the declared fusion fit.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer
import tabace_context_reproduction_v28 as raw
import tabace_validation_banks_v28 as validation
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor

OUT = RUN / 'results/tabace_context_adaptation_v29'
_CACHE = {}


def robust(X, Q):
    center = np.median(X, axis=0)
    scale = np.percentile(X, 75, axis=0)-np.percentile(X, 25, axis=0)
    scale = np.where(scale > 1e-6, scale, np.maximum(X.std(0), 1e-6))
    return (X-center)/scale, (Q-center)/scale


def transport(X, Q, amount=1.):
    a, b = robust(X, Q)
    center = np.median(Q, axis=0)
    scale = np.percentile(Q, 75, axis=0)-np.percentile(Q, 25, axis=0)
    scale = np.where(scale > 1e-6, scale, np.maximum(Q.std(0), 1e-6))
    aligned = a*scale+center
    return np.asarray((1-amount)*X+amount*aligned, dtype=np.float32)


def stable_columns(X, Q):
    Z, V = robust(X, Q)
    # Unlabeled source/query marginal discrepancies, never target-label scores.
    ratio = np.maximum(V.std(0), 1e-6)/np.maximum(Z.std(0), 1e-6)
    score = np.abs(np.median(V, axis=0))+np.abs(np.log(np.clip(ratio, 1e-6, 1e6)))
    # Two-atom shortcut corruption is detectable without knowing the generator.
    if len(X) >= 32:
        for j in range(X.shape[1]):
            _, count = np.unique(X[:, j], return_counts=True)
            if len(count) > 2 and count.max()/len(X) > .20:
                score[j] += 2*count.max()/len(X)
    keep = np.flatnonzero(score <= 1.5)
    if len(keep) < max(1, X.shape[1]//4):
        keep = np.argsort(score, kind='stable')[:max(1, X.shape[1]//2)]
    return np.sort(keep)


def reflect(y, q=.49, shrink=.4):
    values, counts = np.unique(y, return_counts=True)
    lo, hi = np.quantile(y, [q, 1-q])
    # Chunkwise corruptions repeat labels on a 256-row scale, not on total N.
    threshold = max(2, int(np.ceil(.01*min(len(y), 256))))
    repeated = values[(counts >= threshold) & ((values <= lo) | (values >= hi))]
    mask = np.isin(y, repeated)
    clean = y[~mask]
    center = float(np.median(clean)) if len(clean) >= max(8, int(.1*len(y))) else .5*(float(values.min())+float(values.max()))
    result = y.copy()
    result[mask] = center-shrink*(y[mask]-center)
    return result


def denoise(X, y, count=None, alpha=1.):
    order = raw.specific.importance(X, y)
    columns = np.sort(order[:min(len(order), count or len(order))])
    Z, _ = robust(X[:, columns], X[:1, columns])
    Z = np.clip(Z, -10., 10.)
    return Ridge(alpha=alpha, solver='lsqr', tol=1e-6).fit(Z, y).predict(Z)


def generic(X, y, Q, task):
    columns = stable_columns(X, Q)
    A, T = X[:, columns], Q[:, columns]
    if task == 'regression':
        repaired = reflect(y)
        return [('repeat_mode_reflection', X, repaired, Q),
                ('reflected_marginal_transport', transport(X,Q), repaired, Q),
                ('stable_feature_subspace', A, y, T),
                ('stable_subspace_reflection', A, repaired, T),
                ('source_ridge_denoising', X, denoise(X,y), Q)]
    Z, V = robust(X, Q)
    small, _ = robust(A, T)
    repaired = raw.safe_repair_labels(small, y, .7)
    return [('unlabeled_marginal_transport', transport(X,Q), y, Q),
            ('stable_feature_subspace', A, y, T),
            ('stable_subspace_consensus', A, repaired, T),
            ('all_feature_consensus', X, raw.safe_repair_labels(Z,y,.7), Q),
            ('source_robust_asinh', np.arcsinh(Z), y, np.arcsinh(V))]


def augmented_source(meta, family, strength, X, y):
    # Exactly the registered feature-corruption law, applied to SOURCE labels.
    from scaled_historical_feature_protocol import apply_feature_scaled_shift
    cfg = raw.frozen.configuration()['classification_five']['family_settings'][family]
    kind = 'aucband' if family == 'band' else family
    op = f'publicshift_d{meta["data_id"]}_{kind}_rho{strength}_beta{int(cfg["beta"]*100)}'
    shifted, _ = apply_feature_scaled_shift(op, X, y, X, y, 0,
                                           feature_multiplier=cfg['feature_multiplier'])
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) > 1:
        positive = labels[1] if len(labels)==2 else labels[int(np.argmax(counts))]
        direction = X[y==positive].mean(0)-X[y!=positive].mean(0)
        shifted = shifted+cfg.get('location_delta',0.)*(strength/100.)*direction
    return np.asarray(shifted, dtype=np.float32)


def family_views(X, y, Q, task, block, family, meta):
    if block == 'clean':
        if task == 'regression':
            return [(f'ridge_k{k}_a{a}', X, denoise(X,y,k,a), Q)
                    for k,a in [(3,1.),(5,1.),(8,1.),(16,1.),(None,.1),(None,10.)]]
        Z, V = robust(X,Q)
        views = [(f'asinh_scale{s}', np.arcsinh(Z/s), y, np.arcsinh(V/s)) for s in [.5,2.]]
        for normal in ['normal','uniform']:
            transform = QuantileTransformer(n_quantiles=min(128,len(y)),output_distribution=normal,
                                             random_state=0,subsample=None).fit(X)
            views.append((f'quantile_{normal}',transform.transform(X),y,transform.transform(Q)))
        for seed in [0,1]:
            cols = np.sort(np.random.default_rng(seed).choice(X.shape[1],max(1,int(.75*X.shape[1])),replace=False))
            views.append((f'feature_subset75_stream{seed}', X[:,cols], y, Q[:,cols]))
        return views
    if task=='classification' and block=='five':
        result = [(f'source_augmentation_strength{s}', augmented_source(meta,family,s,X,y), y, Q)
                  for s in [20,50,80,100]]
        for amount in [.75,1.25]:
            result.append((f'augmentation50_transport{amount}',transport(augmented_source(meta,family,50,X,y),Q,amount),y,Q))
        return result
    if family=='domain' or (family=='temporal' and task=='regression'):
        return [(f'marginal_transport{a}',transport(X,Q,a),y,Q) for a in [.5,.75,1.,1.25,1.5,2.]]
    if block=='seven' and family in ['concept','temporal']:
        Z, _ = robust(X,Q)
        j = int(np.argmax(X[:min(256,len(y))].var(0)))
        if task=='regression':
            return [(f'observed_response_drift{a}',X,y-a*max(float(y.std()),1e-8)*np.tanh(Z[:,j]),Q)
                    for a in [.25,.5,1.,2.,4.,8.]]
        h = max(2,3*len(y)//4)
        history, labels = X[:h], y[:h]
        result=[]
        for fraction in [.05,.1,.15,.2,.25,.3]:
            order=np.argsort(np.abs(history[:,j]-np.median(history[:,j])),kind='stable')
            revised=labels.copy();chosen=order[:max(1,int(fraction*len(order)))]
            if len(np.unique(y))==2:revised[chosen]=1-revised[chosen]
            A=transport(history,Q) if family=='temporal' else history
            result.append((f'history_boundary_flip{fraction}',A,revised,Q))
        return result
    if task=='regression' and block=='five':
        if family=='shortcut':
            order=raw.specific.importance(X,y)
            return [(f'drop_dominant{k}',X[:,np.sort(order[min(k,len(order)-1):])],y,Q[:,np.sort(order[min(k,len(order)-1):])])
                    for k in [1,2,3,4,8,max(1,X.shape[1]//2)]]
        columns=stable_columns(X,Q) if family in ['local','mixed'] else np.arange(X.shape[1])
        return [(f'repeat_reflection_q{q}_a{a}',X[:,columns],reflect(y,q,a),Q[:,columns])
                for q in [.45,.49] for a in [.25,.5,.75]]
    # Genuine bootstrap streams after observed-pair deduplication; no row padding.
    _, unique=np.unique(np.column_stack([X,y]),axis=0,return_index=True)
    unique=np.sort(unique)
    result=[]
    for seed in range(6):
        rows=np.random.default_rng(np.random.SeedSequence([0,2900,seed])).choice(unique,len(y),replace=True)
        result.append((f'unique_pair_bootstrap{seed}',X[rows],y[rows],Q))
    return result


def new_views(X,y,Q,task,block,family,meta):
    key=(task,block,family,int(meta['task_id']),*(protocol.array_sha256(a) for a in [X,y,Q]))
    if key not in _CACHE:
        if len(_CACHE)>=2:_CACHE.pop(next(iter(_CACHE)))
        _CACHE[key]=generic(X,y,Q,task)+family_views(X,y,Q,task,block,family,meta)
    assert len(_CACHE[key])==11
    return _CACHE[key]


def run_case(predictor,data,rawpath,out,task,block,meta,family,rho,budget):
    original=json.loads(rawpath.with_suffix('.json').read_text())
    assert original['source_input_hashes']=={k:protocol.array_sha256(v) for k,v in data.items()}
    with np.load(rawpath) as z:
        base=z['base'];old=z['members'];truth=z['y_target'];mu=z['source_mean'];sigma=z['source_scale']
    assert np.array_equal(truth,data['y_target'])
    X,y,Q=(data[k] for k in ['X_source','y_source','X_target'])
    before=[protocol.array_sha256(a) for a in [X,y,Q]]
    cached={tuple(before):base}
    for hashes,p in zip(original['member_input_hashes'],old):cached[tuple(hashes)]=p
    predictions=[];names=[];hashes=[];rows=[];aliases=[];seconds=[]
    for name,A,b,T in new_views(X,y,Q,task,block,family,meta):
        A=np.asarray(A,dtype=np.float32);T=np.asarray(T,dtype=np.float32);b=np.asarray(b)
        assert 0<len(b)<=len(y) and A.shape[1]==T.shape[1]>0
        assert all(np.isfinite(v).all() for v in [A,b,T])
        key=tuple(protocol.array_sha256(v) for v in [A,b,T]);start=time.monotonic();alias=key in cached
        if alias:p=cached[key]
        else:p,_=predictor(A,b,T,members=1,random_state=0,n_classes=base.shape[1] if task=='classification' else None)
        assert np.isfinite(p).all();cached[key]=p
        predictions.append(p);names.append(name);hashes.append(key);rows.append(len(b));aliases.append(alias);seconds.append(time.monotonic()-start)
    assert before==[protocol.array_sha256(a) for a in [X,y,Q]],'Candidate construction mutated paired inputs'
    bank=np.concatenate([old[:5],np.asarray(predictions[:5]),old[5:],np.asarray(predictions[5:])])
    raw.scaling.npwrite(out,base=base,members=bank,y_target=truth,source_mean=mu,source_scale=sigma)
    reorder=lambda oldvalues,newvalues:oldvalues[:5]+newvalues[:5]+oldvalues[5:]+newvalues[5:]
    record=original|dict(status='ADAPTED_BANK_COMPLETE_FUSION_PENDING',raw_bank=str(rawpath),
        shared_roles=original['shared_roles']+names[:5],specific_additional_roles=original['specific_additional_roles']+names[5:],
        context_rows=reorder(original['context_rows'],rows),aliases=reorder(original['aliases'],aliases),
        member_input_hashes=reorder(original['member_input_hashes'],hashes),
        member_seconds=reorder(original['member_seconds'],seconds),
        observed_source_only_adaptation=True,extra_clean_labels_used_by_constructors=False,
        actual_severity_received_by_adapted_constructors=False,
        clean_TRAIN_labels_used_for_later_fusion=True,
        source_augmentation_uses_observed_SOURCE_labels_only=True,
        checkpoint_finetuned=False,Specific_contains_all_Shared=True,adaptation_version='v29')
    record.pop('diagnostic_mean_metrics',None)
    atomic_json(out.with_suffix('.json'),record)


def worker(role,shard,shards,budgets):
    import torch,warnings
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictors={};failures=0
    for task,track,meta in raw.jobs()[shard::shards]:
        try:
            pool=raw.scaling.prepare(task,track,meta)
            split=validation.validation_split(task,track,meta,pool) if role=='validation' else None
            for budget in budgets:
                for block in (['clean','five'] if track=='five' else ['seven']):
                    for family,rho in raw.case_grid(block):
                        suffix=Path(str(budget))/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                        rawpath=raw.OUT/'banks'/role/suffix;out=OUT/'banks'/role/suffix
                        if out.exists() and out.with_suffix('.json').exists():continue
                        assert rawpath.exists() and rawpath.with_suffix('.json').exists(),str(rawpath)
                        if role=='query':data,*_=raw.query_input(task,block,meta,family,rho,budget,pool)
                        else:data,_=validation.materialize(task,block,meta,family,rho,budget,pool,split)
                        if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
                        run_case(predictors[task],data,rawpath,out,task,block,meta,family,rho,budget)
                    print('ADAPTED_BANK_BLOCK_COMPLETE',role,budget,task,block,meta['dataset'],flush=True)
        except Exception:
            failures+=1;error=traceback.format_exc();print(error,flush=True)
            atomic_json(OUT/'errors'/f'{role}_{task}_{track}_{meta["task_id"]}.json',dict(error=error))
    if failures:raise SystemExit(1)


def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    destination=OUT/'plan.json'
    assert not destination.exists(),'Registry already declared'
    atomic_json(destination,dict(status='DECLARED_BEFORE_NEW_CANDIDATE_PREDICTIONS',seed=0,
        budgets=raw.BUDGETS,shared_members=10,specific_total_members=21,
        purpose='Repair observed candidate deficiencies diagnosed only on held-out TRAIN',
        immutable_Base=True,raw_v28_retained=True,Specific_inherits_all_Shared=True,
        constructors_receive_query_labels=False,constructors_receive_actual_severity=False,
        family_specific_augmentation_grid=[.2,.5,.8,1.],observed_source_labels_only=True,
        clean_TRAIN_supervision_for_fusion=True,query_label_parameter_selection=False,
        checkpoint_finetuning=False,development_only=True,original_paper_replication=False,
        methods_are_adapted_reconstructions=True,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((raw.OUT/'paper_targets.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker']);p.add_argument('--role',choices=['query','validation']);p.add_argument('--budgets',default='256');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    plan() if a.phase=='plan' else worker(a.role,a.shard,a.shards,[int(n) for n in a.budgets.split(',')])
