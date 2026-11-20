"""Held-out TRAIN candidate banks for v28 Shared/Specific fusion fitting.

Constructors see only observed source labels and unlabeled validation features.
Clean TRAIN labels provide supervision for the fusion policy, not final-query
labels. Classification validation perturbations reuse frozen generator rules.
"""
import argparse
import json
import os
from pathlib import Path
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
from sklearn.model_selection import train_test_split
import tabace_context_reproduction_v28 as main
from prepare_data import atomic_json,protocol
_PSEUDO_CACHE={}


def validation_split(task,track,meta,pool):
    path=main.OUT/'validation_splits'/task/track/f'{meta["task_id"]}.npz'
    if path.exists():
        with np.load(path) as z:return {k:z[k] for k in z.files}
    if task=='regression':
        split=main.repaired.split(meta,track,pool)
        result=dict(train_positions=split['train_positions'],held_positions=split['held_positions'],
            X_validation=split['X_validation'],y_validation=split['y_validation'],
            source_original_ids=split['source_original_ids'],validation_original_ids=split['validation_original_ids'])
    elif track=='five':
        candidates=np.arange(256,len(pool['y']));n=min(128,len(candidates)//2)
        assert n>=8,(meta['dataset'],len(candidates))
        y=pool['y'][candidates];labels,counts=np.unique(y,return_counts=True)
        stratify=y if counts.min()>=2 and n>=len(labels) and len(candidates)-n>=len(labels) else None
        _,held=train_test_split(candidates,test_size=n,random_state=0,stratify=stratify)
        train=np.setdiff1d(np.arange(len(pool['y'])),held)
        result=dict(train_positions=train,held_positions=np.sort(held),X_validation=pool['X'][np.sort(held)],
            y_validation=pool['y'][np.sort(held)],source_original_ids=pool['source_ids'][train],validation_original_ids=pool['source_ids'][np.sort(held)])
    else:
        ids=np.r_[pool['h_ids'],pool['r_ids']];X=np.vstack([pool['H'],pool['R']]);y=np.r_[pool['hy'],pool['ry']]
        protected=np.r_[pool['h_ids'][:256],pool['r_ids'][:64]]
        candidates=np.flatnonzero(~np.isin(ids,protected));n=min(128,len(candidates)//2)
        if set(np.unique(y[candidates]))!=set(np.unique(y)):
            # Small imbalanced tasks may have every minority example in the
            # original320 reference rows. Use an ordinary source cross-fit;
            # final-query inputs keep their original canonical core.
            candidates=np.arange(len(y));n=min(64,len(y)-256)
        assert n>=8,(meta['dataset'],len(candidates))
        labels,counts=np.unique(y[candidates],return_counts=True)
        stratify=y[candidates] if counts.min()>=2 and n>=len(labels) else None
        _,held=train_test_split(candidates,test_size=n,random_state=0,stratify=stratify);held=np.sort(held)
        # Quantile/proportional splits can miss a class represented by a single
        # candidate. Label-prior validation must retain every available class.
        held=held.copy()
        for label in np.unique(y[candidates]):
            if np.any(y[held]==label):continue
            replacement=int(candidates[np.flatnonzero(y[candidates]==label)[0]])
            eligible=[j for j,i in enumerate(held) if np.sum(y[held]==y[i])>1]
            assert eligible,'Validation class coverage cannot be maintained'
            held[eligible[-1]]=replacement
        held=np.sort(held)
        assert set(np.unique(y[held]))==set(np.unique(y[candidates]))
        train=np.setdiff1d(np.arange(len(y)),held);validation_ids=ids[held]
        result=dict(train_positions=train,held_positions=held,X_validation=X[held],y_validation=y[held],
            source_original_ids=ids[train],validation_original_ids=validation_ids,
            h_keep=~np.isin(pool['h_ids'],validation_ids),r_keep=~np.isin(pool['r_ids'],validation_ids))
    assert not np.intersect1d(result['source_original_ids'],result['validation_original_ids']).size
    assert not np.intersect1d(result['validation_original_ids'],pool['query_ids']).size
    main.scaling.npwrite(path,**result)
    atomic_json(path.with_suffix('.json'),dict(task=task,track=track,task_id=meta['task_id'],
        validation_rows=len(result['y_validation']),source_rows=len(result['source_original_ids']),
        held_out_TRAIN_source_disjoint=True,original_query_disjoint=True,
        clean_TRAIN_labels_used_for_fusion_calibration=True,query_labels_used=False,
        hashes={k:protocol.array_sha256(v) for k,v in result.items()}))
    return result


def class_five_shift(meta,family,rho,core_X,core_y,Q):
    from sklearn.ensemble import ExtraTreesClassifier
    from scaled_historical_feature_protocol import apply_feature_scaled_shift
    cfg=main.frozen.configuration()['classification_five']['family_settings'][family]
    key=(meta['task_id'],protocol.array_sha256(core_X),protocol.array_sha256(core_y),protocol.array_sha256(Q))
    if key not in _PSEUDO_CACHE:
        if len(_PSEUDO_CACHE)>=16:_PSEUDO_CACHE.pop(next(iter(_PSEUDO_CACHE)))
        _PSEUDO_CACHE[key]=ExtraTreesClassifier(n_estimators=128,random_state=0,n_jobs=1).fit(core_X,core_y).predict(Q)
    pseudo=_PSEUDO_CACHE[key]
    kind='aucband' if family=='band' else family
    op=f'publicshift_d{meta["data_id"]}_{kind}_rho{rho}_beta{int(cfg["beta"]*100)}'
    shifted,_=apply_feature_scaled_shift(op,core_X,core_y,Q,pseudo,0,feature_multiplier=cfg['feature_multiplier'])
    labels,counts=np.unique(core_y,return_counts=True);positive=labels[1] if len(labels)==2 else labels[int(np.argmax(counts))]
    direction=core_X[core_y==positive].mean(0)-core_X[core_y!=positive].mean(0)
    return shifted+cfg.get('location_delta',0.)*(rho/100.)*direction


def materialize(task,block,meta,family,rho,budget,pool,split):
    Q=split['X_validation'].copy();truth=split['y_validation'].copy()
    if task=='classification' and block in ['clean','five']:
        pos=split['train_positions'];X=pool['X'][pos][:budget];y=pool['y'][pos][:budget]
        if block=='five':Q=class_five_shift(meta,family,rho,pool['X'][:256],pool['y'][:256],Q)
    elif task=='classification':
        filtered=pool|dict(H=pool['H'][split['h_keep']],hy=pool['hy'][split['h_keep']],
                           R=pool['R'][split['r_keep']],ry=pool['ry'][split['r_keep']],
                           H0=pool['H0'][split['h_keep'][:len(pool['H0'])]],hy0=pool['hy0'][split['h_keep'][:len(pool['H0'])]],
                           R0=pool['R0'][split['r_keep'][:len(pool['R0'])]],ry0=pool['ry0'][split['r_keep'][:len(pool['R0'])]])
        X,y=main.scaling.class_seven(filtered,family,rho,budget)
        from internal_seven_fixed_strength import make_condition
        clean=(filtered['H0'],filtered['hy0'],filtered['R0'],filtered['ry0'],Q,truth)
        condition,_=make_condition(meta,family,rho,main.frozen.configuration()['classification_seven'],clean)
        Q,truth=condition['X_target'],condition['y_target']
    elif block=='clean':
        pos=split['train_positions'];X=pool['X'][pos][:budget];y=pool['y'][pos][:budget]
    elif block=='five':
        env=main.repaired.environment(pool,meta,block,family,rho,split['train_positions'])
        rate=json.loads((main.repaired.OUT/'frozen_five_query/parameters.json').read_text())['repair_probability']
        X,y,_=main.repaired.context(env,rate,budget)
    else:
        pos=split['train_positions'];filtered=pool|dict(X=pool['X'][pos],y=pool['y'][pos])
        # The benchmark's unlabeled query-feature-dependent selection statistics
        # remain fixed. Original query labels are not read here.
        original_features=main.repaired.core_arrays(block,meta,family,rho)['X_target']
        X,y=main.scaling.reg_seven(filtered,family,rho,budget,original_features)
    data=dict(X_source=X,y_source=y,X_target=Q,y_target=truth)
    scale=1. if task=='classification' else max(float(truth.std()),1e-12)
    return data,scale


def worker(shard,shards,budgets):
    import torch,warnings
    from run_tabicl import Predictor
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore');predictors={}
    for task,track,meta in main.jobs()[shard::shards]:
        try:
            pool=main.scaling.prepare(task,track,meta);split=validation_split(task,track,meta,pool)
            if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
            predictor=predictors[task]
            if task=='classification':nc=int(pool['y'].max()+1) if 'y' in pool else 2
            else:nc=None
            blocks=['clean','five'] if track=='five' else ['seven']
            for budget in budgets:
                for block in blocks:
                    for family,rho in main.case_grid(block):
                        out=main.OUT/'banks/validation'/str(budget)/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                        if out.exists() and out.with_suffix('.json').exists():continue
                        data,scale=materialize(task,block,meta,family,rho,budget,pool,split)
                        base,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,random_state=0,n_classes=nc)
                        main.run_case(predictor,data,base,task,block,meta,family,rho,budget,scale,out,
                            str(main.OUT/'validation_splits'/task/track/f'{meta["task_id"]}.npz'))
                    print('VALIDATION_BANK_BLOCK_COMPLETE',budget,task,block,meta['dataset'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(main.OUT/'errors'/f'validation_{task}_{track}_{meta["task_id"]}.json',dict(error=error))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--budgets',default='256');a=p.parse_args()
    worker(a.shard,a.shards,[int(n) for n in a.budgets.split(',')])
