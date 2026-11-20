"""TRAIN-supervised residual calibration of frozen classifier probabilities.

Uses the existing matched pseudo-label perturbation protocol. Source sample IDs
are grouped across all perturbations and label-prior resamples. Specific's
outer-fold parent is trained without that fold, including for training anchors.
No final-query bank is loaded until every selected head is frozen.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import torch
from torch import nn
import recovery_contexts_v44 as source
from prepare_data import RUN,atomic_json
from run_tabicl import normalize,metrics

OUT=RUN/'results/tabace_probability_head_v51'
PROFILES=[dict(hidden=0,weight_decay=1.,epochs=120,learning_rate=.02),
          dict(hidden=32,weight_decay=1.,epochs=120,learning_rate=.02),
          dict(hidden=32,weight_decay=.1,epochs=120,learning_rate=.01)]
BLENDS=[.25,.5,.75,1.]


def original_ids(meta,block,truth):
    track='seven' if block=='seven' else 'five'
    path=source.adaptation.raw.OUT/'validation_splits/classification'/track/f'{meta["task_id"]}.npz'
    with np.load(path) as z:ids=z['validation_original_ids'];labels=z['y_validation']
    if block=='seven' and meta['family']=='label':
        cfg=source.adaptation.raw.frozen.configuration()['classification_seven']['generator_parameters']
        mu=float(labels.mean());extreme=.9 if mu>=.5 else .1
        if cfg['label_prior_direction']==-1:extreme=1-extreme
        prior=mu+(meta['rho']/100.)*cfg['label_prior_displacement_coefficient']*(extreme-mu)
        rng=np.random.default_rng(0);sample=(rng.random(len(labels))<prior).astype(int)
        selected=np.asarray([rng.choice(np.flatnonzero(labels==label)) for label in sample])
        ids=ids[selected];labels=labels[selected]
    if not (block=='seven' and meta['family'] in ['concept','temporal']):assert np.array_equal(labels,truth)
    assert len(ids)==len(truth)
    return ids


def load(role,budget,block):
    cohort={str(m['task_id']):m for m in source.adaptation.raw.frozen.cohort('classification',block)}
    order=list(range(21))
    if block=='clean':
        choice=source.full.OUT/'policies'/str(budget)/'classification/clean/generic_scale_selection.json'
        selected=json.loads(choice.read_text());assert selected['query_labels_used'] is False
        order=selected['permutation']
    cells=[]
    folder=source.OUT/'banks'/role/str(budget)/'classification'/block
    for path in sorted(folder.glob('*/*.npz')):
        if '.tmp.' in path.name:continue
        meta=json.loads(path.with_suffix('.json').read_text());meta['data_id']=cohort[str(meta['task_id'])]['data_id']
        with np.load(path) as z:c=dict(base=z['base'],bank=z['members'][order],y=z['y_target'].astype(int))
        assert c['bank'].shape[0]==21 and c['y'].min()>=0 and c['y'].max()<c['base'].shape[1]
        c.update(meta=meta,path=path,key=str(path.relative_to(source.OUT/'banks'/role/str(budget))),order=order)
        if role=='validation':
            c['ids']=original_ids(meta,block,c['y'])
            c['folds']=np.asarray([int.from_bytes(hashlib.sha256(f'0:{meta["data_id"]}:{int(i)}'.encode()).digest()[:8],'big')%4 for i in c['ids']])
        cells.append(c)
    expected=len(cohort)*dict(clean=1,five=20,seven=28)[block];assert len(cells)==expected
    return cells


def feature(c,m):
    B=np.log(np.clip(c['base'],1e-12,1.));B=B-B.mean(1,keepdims=True)
    P=np.clip(c['bank'][:m],1e-12,1.)
    logp=np.log(P);logp=logp-logp.mean(2,keepdims=True)
    delta=np.clip(logp-B[None],-12.,12.).transpose(1,0,2).reshape(len(B),-1)
    entropy=-(P*np.log(P)).sum(2).T/np.log(max(2,B.shape[1]))
    base_entropy=-(c['base']*np.log(np.clip(c['base'],1e-12,1.))).sum(1,keepdims=True)/np.log(max(2,B.shape[1]))
    return np.column_stack([B,delta,base_entropy,entropy]).astype(np.float32)


def bundles(cells):
    grouped={}
    for c in cells:grouped.setdefault(str(c['meta']['task_id']),[]).append(c)
    result={}
    for task,group in grouped.items():
        lengths=[len(c['y']) for c in group];starts=np.cumsum([0]+lengths[:-1])
        result[task]=dict(cells=group,starts=starts,lengths=lengths,
            X10=np.concatenate([feature(c,10) for c in group]),X21=np.concatenate([feature(c,21) for c in group]),
            base=np.concatenate([c['base'] for c in group]),y=np.concatenate([c['y'] for c in group]),
            ids=np.concatenate([c['ids'] for c in group]),folds=np.concatenate([c['folds'] for c in group]),
            case_ids=np.concatenate([np.full(len(c['y']),i,dtype=int) for i,c in enumerate(group)]))
    return result


def weights(case_ids):
    # Preserve the actual label-prior resampling distribution within each case.
    # Inverse ORIGINAL-row-frequency weights would incorrectly undo that shift.
    _,inverse,count=np.unique(case_ids,return_inverse=True,return_counts=True)
    return 1./count[inverse]


def network(d,k,hidden,device):
    if hidden:
        model=nn.Sequential(nn.Linear(d,hidden),nn.ReLU(),nn.Linear(hidden,k))
    else:model=nn.Sequential(nn.Linear(d,k))
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    return model.to(device)


def fit_net(X,y,anchor,ids,profile,device,case_ids=None):
    torch.manual_seed(0)
    w=weights(case_ids) if case_ids is not None else np.ones(len(ids))
    mean=np.average(X,axis=0,weights=w);variance=np.average((X-mean)**2,axis=0,weights=w)
    scale=np.sqrt(variance);scale=np.where(scale>1e-6,scale,1.)
    mean,scale=mean.astype(np.float32),scale.astype(np.float32)
    Z=torch.as_tensor(np.clip((X-mean)/scale,-10.,10.),device=device,dtype=torch.float32)
    Y=torch.as_tensor(y,device=device,dtype=torch.long)
    W=torch.as_tensor(w,device=device,dtype=torch.float32);W=W/W.sum()
    offset=torch.as_tensor(np.log(np.clip(anchor,1e-12,1.)),device=device,dtype=torch.float32)
    model=network(X.shape[1],anchor.shape[1],profile['hidden'],device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=profile['learning_rate'],weight_decay=profile['weight_decay'])
    model.train()
    for _ in range(profile['epochs']):
        optimizer.zero_grad(set_to_none=True)
        loss=(torch.nn.functional.cross_entropy(offset+model(Z),Y,reduction='none')*W).sum()
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
    return dict(mean=torch.from_numpy(mean),scale=torch.from_numpy(scale),profile=profile,d=X.shape[1],k=anchor.shape[1],
        state={k:v.detach().cpu() for k,v in model.state_dict().items()},unique_TRAIN_rows=int(len(np.unique(ids))))


def infer(X,anchor,saved,device):
    model=network(saved['d'],saved['k'],saved['profile']['hidden'],device);model.load_state_dict(saved['state']);model.eval()
    Z=torch.as_tensor(X,device=device,dtype=torch.float32)
    Z=((Z-saved['mean'].to(device))/saved['scale'].to(device)).clamp(-10,10)
    offset=torch.as_tensor(np.log(np.clip(anchor,1e-12,1.)),device=device,dtype=torch.float32)
    with torch.no_grad():p=torch.softmax(offset+model(Z),1).cpu().numpy()
    return normalize(p)


def unpack(bundle,predictions):
    return {c['key']:predictions[s:s+n] for c,s,n in zip(bundle['cells'],bundle['starts'],bundle['lengths'])}


def score(cells,predictions):
    values=[metrics(c['y'],predictions[c['key']],'classification',1.) for c in cells]
    accuracy=float(np.mean([v['accuracy'] for v in values]));aucs=[v['auc'] for v in values if v['auc'] is not None]
    auc=float(np.mean(aucs)) if aucs else None
    return dict(accuracy=accuracy,auc=auc),accuracy if auc is None else .5*(accuracy+auc)


def parent_prediction(bundle,shared,root,fold,device):
    if shared['chosen'] is None:return bundle['base']
    task=str(bundle['cells'][0]['meta']['task_id']);chosen=shared['chosen']
    path=(root/'Shared/cv'/str(chosen['profile_index'])/f'{task}_{fold}.pt') if fold is not None else Path(shared['models'][task])
    model=torch.load(path,map_location='cpu',weights_only=True)
    p=infer(bundle['X10'],bundle['base'],model,device);a=chosen['blend']
    return normalize((1-a)*bundle['base']+a*p)


def fit_policy(cells,m,root,name,device,shared=None):
    groups=bundles(cells)
    baseline={};anchors={}
    for task,b in groups.items():
        if shared is None:baseline.update(unpack(b,b['base']))
        else:
            p=np.empty_like(b['base'],dtype=np.float64)
            for fold in range(4):
                mask=b['folds']==fold
                if not mask.any():continue
                parent=parent_prediction(b,shared,root,fold,device)
                anchors[task,fold]=parent;p[mask]=parent[mask]
            baseline.update(unpack(b,p))
    initial,best_value=score(cells,baseline);best=None;best_oof=baseline;trials=[]
    for index,profile in enumerate(PROFILES):
        predictions={}
        for task,b in groups.items():
            p=np.empty_like(b['base'],dtype=np.float64);X=b['X10'] if m==10 else b['X21']
            for fold in range(4):
                train=b['folds']!=fold;valid=~train
                if not valid.any():continue
                assert train.any() and not np.intersect1d(b['ids'][train],b['ids'][valid]).size
                anchor=b['base'] if shared is None else anchors[task,fold]
                model=fit_net(X[train],b['y'][train],anchor[train],b['ids'][train],profile,device,b['case_ids'][train])
                p[valid]=infer(X[valid],anchor[valid],model,device)
                if m==10:
                    path=root/name/'cv'/str(index)/f'{task}_{fold}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(model,path)
            predictions.update(unpack(b,p))
        for blend in BLENDS:
            combined={c['key']:normalize((1-blend)*baseline[c['key']]+blend*predictions[c['key']]) for c in cells}
            values,utility=score(cells,combined);trials.append(dict(profile_index=index,blend=blend,source_OOF_scores=values,utility=utility))
            if utility>best_value+1e-10:best_value=utility;best=dict(profile_index=index,blend=blend);best_oof=combined
        print('PROBABILITY_PROFILE_COMPLETE',name,index,flush=True)
    models={}
    if best is not None:
        for task,b in groups.items():
            X=b['X10'] if m==10 else b['X21']
            anchor=b['base'] if shared is None else parent_prediction(b,shared,root,None,device)
            model=fit_net(X,b['y'],anchor,b['ids'],PROFILES[best['profile_index']],device,b['case_ids'])
            path=root/name/'models'/f'{task}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(model,path);models[task]=str(path)
    policy=dict(chosen=best,models=models,m=m)
    atomic_json(root/name/'policy.json',dict(status='FROZEN_BEFORE_QUERY',model=policy,source_baseline=initial,trials=trials,
        original_row_group_folds=4,condition_normalized_loss=True,label_prior_distribution_preserved=True,
        repeated_original_IDs_always_in_same_fold=True,query_labels_used=False,seed=0))
    return policy,best_oof


def predict_cell(c,policy,anchor,device,cache):
    if policy['chosen'] is None:return anchor
    task=str(c['meta']['task_id']);path=policy['models'][task]
    if path not in cache:cache[path]=torch.load(path,map_location='cpu',weights_only=True)
    p=infer(feature(c,policy['m']),anchor,cache[path],device);a=policy['chosen']['blend']
    return normalize((1-a)*anchor+a*p)


def run(budget,block,device):
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    cells=load('validation',budget,block);root=OUT/'policies'/str(budget)/'classification'/block
    shared,shared_oof=fit_policy(cells,10,root,'Shared',device)
    specific={}
    for family in sorted({c['meta']['family'] for c in cells}):
        group=[c for c in cells if c['meta']['family']==family]
        specific[family],_=fit_policy(group,21,root,f'Specific_{family}',device,shared)
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY_BANKS_LOADED',
        extra_clean_TRAIN_supervision=True,Specific_family_visible=True,actual_strength_visible_to_network=False,
        Shared_members=10,Specific_members=21,Base_is_additional_cached_prediction=True,
        frozen_TFM_checkpoints=True,checkpoint_finetuned=False,seed=0,development_only=True,original_paper_replication=False))
    query=load('query',budget,block);cache={};rows=[]
    for c in query:
        sh=predict_cell(c,shared,c['base'],device,cache)
        sp=predict_cell(c,specific[c['meta']['family']],sh,device,cache)
        out=OUT/'predictions'/str(budget)/c['key'];source.adaptation.raw.scaling.npwrite(out,Base=c['base'],Shared=sh,Specific=sp,y_target=c['y'])
        meta=c['meta'];summary={k:meta[k] for k in ['task','block','task_id','dataset','family','rho','requested_context','actual_context','scale']}
        summary.update(input_hashes=meta['source_input_hashes'],source_bank=str(c['path']),role_permutation=c['order'],
            methods={name:metrics(c['y'],p,'classification',1.) for name,p in [('Base',c['base']),('Shared',sh),('Specific',sp)]},
            provisional_Base_calibration=False,adaptation_version='v51',seed=0,extra_clean_TRAIN_supervision=True,
            query_labels_used_for_fit=False,Specific_family_visible=True,actual_strength_visible_to_network=False,
            query_feedback_used_for_benchmark_design=True,development_only=True,original_paper_replication=False,independent_test_claim=False)
        atomic_json(out.with_suffix('.json'),summary)
        for name,values in summary['methods'].items():rows.append(dict(context=budget,block=block,method=name,task_id=meta['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=len(query),no_new_TFM_calls=True,adaptation_marker='dagger'))
    print('PROBABILITY_HEAD_COMPLETE',budget,block,len(query),flush=True)


def plan():
    assert not (OUT/'plan.json').exists()
    specification=json.loads((RUN/'results/classification_residual_head_next_spec_v51.json').read_text())
    specification.update(status='DECLARED_BEFORE_HEAD_FITS',source_banks=str(source.OUT),profiles=PROFILES,blends=BLENDS,
        probability_only=True,no_new_TFM_calls=True,calibration_generator='existing matched pseudo-label v44 protocol; rejected v50 not used',
        folds='Group all occurrences of the same original TRAIN row in one of four folds; normalize loss per condition.',
        weighting_design_amendment='Use condition-normalized loss, not inverse original-row frequency: the latter would erase legitimate label-prior resampling. Count independent supervision by unique original TRAIN IDs.',
        regression_reuse='v47, preserving all48 goal checks',Markdown_written=False)
    atomic_json(OUT/'plan.json',specification)
    (OUT/'paper_targets.json').write_text((source.OUT/'paper_targets.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);p.add_argument('--block',choices=['clean','five','seven']);p.add_argument('--device',default='cuda:0');a=p.parse_args();plan() if a.phase=='plan' else run(a.budget,a.block,a.device)
