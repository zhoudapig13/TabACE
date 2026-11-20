"""Actual corruption-matched TabICL weight updates followed by TabACE.

The v22 development benchmark is immutable. Classification five corruptions
act on query FEATURES; regression Clean/five is semisynthetic. These facts are
reported rather than presenting this suite as the archived paper experiment.
Only official TRAIN rows train weights, choose checkpoints, or calibrate fusion.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tabicl import TabICLClassifier, TabICLRegressor

def _build_ensemble_generator(*args, **kwargs):
    # Inference/cached replay does not need the optional training dependencies.
    from tabicl._finetune.data import _build_ensemble_generator as implementation
    return implementation(*args, **kwargs)

import benchmark256_v22 as frozen
import tabicl_context_scaling_v23 as scaling
import tabace_context_reproduction_v28 as contexts
import tabace_validation_banks_v28 as calibration
import tabace_fusion_v28 as fusion
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import CHECKPOINTS, Predictor, metrics

OUT = RUN / 'results/finetune_tabace_main_v1'
STRENGTHS = [20, 50, 80, 100]
STEPS = 96
LEARNING_RATE = 1e-5
CLASS_QUERY_MODE = 'uniform'
RARE_CLASS_RETRIES = {3902,3918,9978,3021,146819,146820}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs():
    result = []
    # Interleave the four tracks so that partial progress covers both tasks.
    groups = [(task, track, frozen.cohort(task, track))
              for task in ('classification', 'regression') for track in ('five', 'seven')]
    for index in range(max(len(group) for _, _, group in groups)):
        for task, track, group in groups:
            if index < len(group):
                result.append(dict(task=task, track=track, meta=group[index]))
    return result


def families(track):
    from seven_shifts import FAMILIES
    return list(protocol.FAMILIES if track == 'five' else FAMILIES)


def folder(job):
    return OUT / 'jobs' / job['task'] / job['track'] / str(job['meta']['task_id'])


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'plan.json').exists():
        return
    atomic_json(OUT / 'plan.json', dict(
        status='DECLARED_BEFORE_NEW_WEIGHT_UPDATES', seed=0, context_cap=256,
        deadline_UTC='2026-11-12T00:00:00Z', deadline_Asia_Shanghai='2026-11-12T08:00:00+08:00',
        user_latest_override='Do not rerun unfinetuned Shared/Specific; apply both to the fine-tuned backbone',
        methods=['FT', 'FT+Shared', 'FT+Specific'], Base='reuse audited frozen v22 predictions',
        paper_Shared_Specific='historical reference only, not paired reruns',
        benchmark_sha256=sha(frozen.CONFIG), original_paper_replication=False,
        development_benchmark=True, benchmark_selected_previously_using_Base_scores=True,
        regression_clean_five_semisynthetic=True,
        classification_five_operator_location='query features, not source context; unchanged frozen v22 protocol',
        other_corruptions='same frozen source-context operators as v22',
        final_query_labels_for_weight_updates_checkpoint_selection_or_fusion=False,
        supervision='additional labeled official TRAIN rows and pre-corruption clean TRAIN labels',
        split='disjoint fine-tune train / checkpoint validation / fusion calibration; all disjoint final queries',
        steps=STEPS, learning_rate=LEARNING_RATE, weight_decay=.01, grad_clip=1.,
        train_time_cap_seconds_per_dataset=150, trainable_modules='last four ICL blocks plus ICL output/y encoders',
        checkpoint_selection='source validation only, clean 20% + corruption 80%',
        training_mixture='20% clean; 80% same registered mechanism and strengths .2,.5,.8,1',
        n_estimators_per_branch=1, shared_alternative_branches=5, specific_alternative_branches=10,
        specific_includes_shared=True, Shared_family_blind=True,
        Specific_family_visible=True, constructor_strength_blind=True,
        calibration_strengths=[.5,1.], fusion='original v28 confidence/agreement/Base gate, gain=1, fixed regularization=.01',
        cost='synchronized per-branch inference, construction, fusion fitting/apply, training, peak CUDA memory',
        worker_GPUs=list(range(8)), full_cohorts=dict(classification_clean_five=72,classification_seven=35,
            regression_clean_five=35,regression_seven=33),
        counts_of_task_track_checkpoints=len(jobs()), Markdown_written=False,
        script_sha256=sha(__file__)))
    atomic_json(OUT / 'job_manifest.json', jobs())
    reference = pd.read_csv(frozen.OUT / 'Base_averages.csv')
    reference.to_csv(OUT / 'frozen_Base_averages.csv', index=False)


def flat(pool):
    if 'h_ids' in pool:
        return np.vstack([pool['H'], pool['R']]), np.r_[pool['hy'], pool['ry']], np.r_[pool['h_ids'], pool['r_ids']]
    return pool['X'], pool['y'], pool['source_ids']


def query_sample(eligible,y,size,rng,preferred=None):
    """Keep one source support example per class and cover query classes."""
    eligible=np.asarray(eligible,dtype=int)
    protected=np.asarray([eligible[np.flatnonzero(y[eligible]==c)[0]] for c in np.unique(y[eligible])],dtype=int)
    candidates=np.setdiff1d(eligible,protected)
    mandatory=np.asarray([int(rng.choice(candidates[y[candidates]==c])) for c in np.unique(y[candidates])],dtype=int)
    favored=candidates if preferred is None else np.intersect1d(candidates,preferred)
    order=np.r_[mandatory,rng.permutation(np.setdiff1d(favored,mandatory)),rng.permutation(np.setdiff1d(candidates,np.r_[favored,mandatory]))]
    return order[:min(max(size,len(mandatory)),len(order))]


def prepare_split(job, pool):
    path = folder(job) / 'split.npz'
    if path.exists():
        with np.load(path) as z:
            result={key:z[key] for key in z.files}
        _,existing_y,_=flat(pool)
        if job['task']!='classification' or len(np.unique(existing_y[result['val']]))>=2 or (folder(job)/'training_result.json').exists():
            return result
        # Retry a failed imbalanced development split before any checkpoint
        # exists. Preserve its exact original split for the failure audit.
        backup=path.with_name('split_before_rare_class_fix.npz')
        if not backup.exists():backup.write_bytes(path.read_bytes())
    X, y, ids = flat(pool)
    original = calibration.validation_split(job['task'],job['track'],job['meta'],pool)
    held = original['held_positions']
    available = np.setdiff1d(np.arange(len(y)), held)
    # Protect one source example per label so that rare labels are not lost.
    protected = np.asarray([available[np.flatnonzero(y[available] == c)[0]] for c in np.unique(y[available])], dtype=int) if job['task']=='classification' else np.array([],dtype=int)
    candidates = np.setdiff1d(available, np.r_[np.arange(min(256,len(y))),protected])
    if len(candidates) < 16:
        candidates = np.setdiff1d(available,protected)
    rng = np.random.default_rng(4301)
    safe_class=(job['task']=='classification' and (CLASS_QUERY_MODE!='uniform' or
                (job['track']=='seven' and int(job['meta']['task_id']) in RARE_CLASS_RETRIES)))
    val = (query_sample(available,y,min(64,max(8,len(candidates)//4)),rng,candidates)
           if safe_class else rng.permutation(candidates)[:min(64,max(8,len(candidates)//4))])
    train = np.setdiff1d(available,val)
    # Retain canonical source order then draw up to 2048 other TRAIN rows.
    train = np.r_[train[train<256],rng.permutation(train[train>=256])][:2048]
    assert min(len(train),len(val),len(held)) > 0
    assert not np.intersect1d(ids[train],ids[val]).size
    assert not np.intersect1d(ids[train],ids[held]).size
    assert not np.intersect1d(ids[val],ids[held]).size
    assert not np.intersect1d(ids[np.r_[train,val,held]],pool['query_ids']).size
    result=dict(train=train,val=val,cal=held,ids=ids)
    scaling.npwrite(path,**result)
    atomic_json(path.with_suffix('.json'),dict(train_rows=len(train),checkpoint_validation_rows=len(val),fusion_calibration_rows=len(held),
        final_query_rows=len(pool['query_ids']),all_four_partitions_disjoint=True,
        source_original_ids_sha256=protocol.array_sha256(ids),additional_clean_TRAIN_labels=True))
    return result


def materialize(job,pool,positions,query_positions,family,rho):
    X,y,ids=flat(pool)
    assert not np.intersect1d(positions,query_positions).size
    positions=np.asarray(positions,dtype=int)
    Q=X[query_positions].copy();truth=y[query_positions].copy()
    task,track,meta=job['task'],job['track'],job['meta']
    data=dict(X_source=X[positions[:256]].copy(),y_source=y[positions[:256]].copy(),X_target=Q,y_target=truth)
    if not rho:
        return data
    if task=='classification' and track=='five':
        data['X_target']=calibration.class_five_shift(meta,family,rho,pool['X'][:256],pool['y'][:256],Q)
    elif task=='classification':
        keep=np.isin(np.arange(len(ids)),positions)
        nh=len(pool['H']);hk=keep[:nh];rk=keep[nh:]
        filtered=pool|dict(H=pool['H'][hk],hy=pool['hy'][hk],R=pool['R'][rk],ry=pool['ry'][rk],
            H0=pool['H0'][hk[:len(pool['H0'])]],hy0=pool['hy0'][hk[:len(pool['H0'])]],
            R0=pool['R0'][rk[:len(pool['R0'])]],ry0=pool['ry0'][rk[:len(pool['R0'])]])
        data['X_source'],data['y_source']=scaling.class_seven(filtered,family,rho,256)
        from internal_seven_fixed_strength import make_condition
        clean=(filtered['H0'],filtered['hy0'],filtered['R0'],filtered['ry0'],Q,truth)
        if family!='label' or len(np.unique(truth))>1:
            condition,_=make_condition(meta,family,rho,frozen.configuration()['classification_seven'],clean)
            data['X_target'],data['y_target']=condition['X_target'],condition['y_target']
    elif track=='five':
        from tabicl_semisynthetic_five_v22 import corrupt_data
        data=corrupt_data(data,family,rho,frozen.configuration()['regression_five']['families'][family]['parameters'])
    else:
        filtered=pool|dict(X=X[positions],y=y[positions])
        data['X_source'],data['y_source']=scaling.reg_seven(filtered,family,rho,256,Q)
    assert len(data['y_source'])<=256 and all(np.isfinite(a).all() for a in data.values())
    return data


def batch(data,task,seed,max_classes):
    X,y,Q,t=(data[k].copy() for k in ('X_source','y_source','X_target','y_target'))
    if task=='classification':
        labels=np.unique(y)
        if len(labels)>max_classes:
            present=np.intersect1d(labels,np.unique(t))
            rng=np.random.default_rng(seed)
            first=int(rng.choice(present))
            labels=np.sort(np.r_[first,rng.choice(labels[labels!=first],max_classes-1,replace=False)])
        cx=np.isin(y,labels);cq=np.isin(t,labels)
        if not cq.any() or len(labels)<2:return None
        X,y,Q,t=X[cx],np.searchsorted(labels,y[cx]),Q[cq],np.searchsorted(labels,t[cq])
        mean,sd,ratio=0.,1.,1.
    else:
        mean=float(y.mean());sd=max(float(y.std()),1e-8)
        target_sd=max(float(t.std()),1e-6);ratio=sd/target_sd
        y=(y-mean)/sd;t=(t-mean)/sd
    if X.shape[1]==0 or np.all(np.ptp(X,axis=0)==0):return None
    gen=_build_ensemble_generator(classification=task=='classification',n_estimators=1,norm_methods=None,
        feat_shuffle_method='latin',class_shuffle_method='shift',outlier_threshold=4.,random_state=0)
    gen.fit(X,y);variants=gen.transform(Q,mode='both');As=[];ys=[];ts=[]
    for norm,(A,b) in variants.items():
        As.append(A);ys.append(b)
        for _,pattern in gen.ensemble_configs_[norm]:
            ts.append(np.asarray(pattern)[t.astype(int)] if task=='classification' and pattern is not None else t)
    return dict(X=torch.from_numpy(np.concatenate(As)).float(),y=torch.from_numpy(np.concatenate(ys)).float(),
        t=torch.from_numpy(np.stack(ts)).long() if task=='classification' else torch.from_numpy(np.stack(ts)).float(),
        ratio=ratio,classes=len(np.unique(y)) if task=='classification' else None)


def train_mode(model):
    model.train();model.col_embedder.eval();model.row_interactor.eval()


def loss_for(model,b,task):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(b['X'].cuda(),b['y'].cuda()).float()
    t=b['t'].cuda()
    if task=='classification':
        logits=out[...,:b['classes']]
        return F.cross_entropy(logits.reshape(-1,b['classes']),t.reshape(-1))
    pred=out.mean(-1)
    # Normalized by clean query TRAIN scale; extreme corrupt labels cannot
    # make severely corrupted episodes appear artificially easy.
    ratio=min(b['ratio'],100.)
    alphas=torch.linspace(0.,1.,out.shape[-1]+2,device=out.device)[1:-1]
    diff=t.unsqueeze(-1)-out
    return (((pred-t)*ratio)**2).mean()+.05*(torch.maximum(alphas*diff,(alphas-1)*diff).mean()*ratio)


def validate(model,batches,task):
    train_mode(model)
    with torch.no_grad():
        values=[float(loss_for(model,b,task)) for b in batches if b is not None]
    return float(np.mean(values)) if values else float('inf')


def train(job,pool,split,steps=STEPS,lr=LEARNING_RATE):
    dest=folder(job);dest.mkdir(parents=True,exist_ok=True)
    path=dest/'training_result.json'
    if path.exists():return json.loads(path.read_text())
    task,track,meta=job['task'],job['track'],job['meta']
    torch.manual_seed(0);np.random.seed(0);torch.cuda.reset_peak_memory_stats()
    cls=TabICLClassifier if task=='classification' else TabICLRegressor
    checkpoint=CHECKPOINTS/f'tabicl-{"classifier" if task=="classification" else "regressor"}-v2-20260212.ckpt'
    loader=cls(n_estimators=1,device='cuda:0',model_path=str(checkpoint),allow_auto_download=False,n_jobs=1,random_state=0)
    loader._resolve_device();loader._load_model();model=loader.model_.cuda()
    model.row_interactor.register_forward_hook(lambda _module,_args,output:output.clone())
    for p in model.parameters():p.requires_grad=False
    for p in model.icl_predictor.parameters():p.requires_grad=True
    for block in model.icl_predictor.tf_icl.blocks[:-4]:
        for p in block.parameters():p.requires_grad=False
    initial={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
    limit=int(model.max_classes) if task=='classification' else 1
    start=time.monotonic();fs=families(track);rng=np.random.default_rng(417)
    val=[]
    for family,rho in [('original',0)]+[(f,80) for f in fs]:
        data=materialize(job,pool,split['train'],split['val'],family,rho)
        val.append(batch(data,task,1337,limit))
    base_loss=validate(model,val,task);history=[];best=None;updates=0;skipped=0;last=0
    X,y,_=flat(pool)
    log=(dest/'steps.jsonl').open('a',buffering=1)
    prepared={}
    for step in range(1,steps+1):
        # Every five-step group contains one clean episode; other episodes
        # cycle through all registered mechanisms and nominal severities.
        clean=(step-1)%5==0
        if clean:family,rho='original',0
        else:
            index=(step-1)-(step-1)//5-1
            family=fs[index%len(fs)];rho=STRENGTHS[(index//len(fs))%4]
        key=(family,rho,(step//(5*len(fs)*4))%2)
        if key not in prepared:
            preferred=split['train'][split['train']>=256]
            if len(preferred)<16:preferred=split['train']
            nq=min(64,max(8,len(preferred)//4))
            safe_class=(task=='classification' and (CLASS_QUERY_MODE!='uniform' or
                        (track=='seven' and int(meta['task_id']) in RARE_CLASS_RETRIES)))
            qry=(query_sample(split['train'],y,nq,rng,preferred) if safe_class
                 else rng.permutation(preferred)[:nq])
            pos=np.setdiff1d(split['train'],qry)
            pos=np.r_[pos[pos<256],rng.permutation(pos[pos>=256])]
            data=materialize(job,pool,pos,qry,family,rho)
            prepared[key]=batch(data,task,step,limit)
        b=prepared[key]
        if b is None:skipped+=1;continue
        train_mode(model);optimizer.zero_grad(set_to_none=True)
        loss=loss_for(model,b,task)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        if not torch.isfinite(grad):raise FloatingPointError('nonfinite training gradient')
        optimizer.step();updates+=1;last=step
        log.write(json.dumps(dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),family=family,rho=rho))+'\n')
        stop=time.monotonic()-start>=150
        if step%32==0 or step==steps or stop:
            value=validate(model,val,task);history.append(dict(step=step,source_validation_loss=value))
            if best is None or value<best['source_validation_loss']:
                best=history[-1].copy()
                # Store only the trainable tensor delta (~75 MB vs full model).
                state={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
                torch.save(dict(state_dict=state,step=step,pretrained_checkpoint=str(checkpoint)),dest/'weights.pt')
            print('FT_CHECKPOINT',task,track,meta['task_id'],step,value,flush=True)
            if stop:break
    log.close()
    if best is None:raise RuntimeError('No actual weight-update checkpoint saved')
    selected=torch.load(dest/'weights.pt',map_location='cpu',weights_only=False)['state_dict']
    delta=sum(float(torch.sum((selected[k]-p)**2)) for k,p in initial.items())**.5
    changed=sum(not torch.equal(selected[k],p) for k,p in initial.items())
    assert delta>0 and changed>0
    result=dict(status='COMPLETE',task=task,track=track,task_id=meta['task_id'],dataset=meta['dataset'],
        seed=0,learning_rate=lr,requested_steps=steps,optimizer_steps=updates,last_step=last,skipped_degenerate=skipped,
        classification_query_sampling=CLASS_QUERY_MODE,rare_class_retry=(track=='seven' and int(meta['task_id']) in RARE_CLASS_RETRIES),
        selected_checkpoint=best,pretrained_source_validation_loss=base_loss,history=history,
        actual_backbone_weight_updates=True,changed_parameter_tensors=changed,trainable_weight_delta_L2=delta,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.monotonic()-start,
        peak_training_CUDA_MiB=torch.cuda.max_memory_allocated()/2**20,
        final_query_labels_used=False,source_validation_queries_excluded_weight_training=True,
        fusion_calibration_queries_excluded_weight_training_and_checkpoint_selection=True,
        checkpoint_file_sha256=sha(dest/'weights.pt'))
    atomic_json(path,result)
    del model,loader,optimizer,initial,selected,prepared,val
    gc.collect();torch.cuda.empty_cache()
    print('FT_TRAIN_COMPLETE',task,track,meta['task_id'],result['training_seconds'],flush=True)
    return result


class UpdatedPredictor(Predictor):
    def __init__(self,job):
        super().__init__(job['task'],'cuda:0');self.job=job

    def __call__(self,X,y,Q,**kwargs):
        if not self.models:
            cls=TabICLClassifier if self.task=='classification' else TabICLRegressor
            name=f'tabicl-{"classifier" if self.task=="classification" else "regressor"}-v2-20260212.ckpt'
            model=cls(n_estimators=1,device='cuda:0',model_path=str(CHECKPOINTS/name),
                allow_auto_download=False,n_jobs=1,batch_size=1,random_state=0)
            model._resolve_device();model._load_model()
            state=torch.load(folder(self.job)/'weights.pt',map_location='cpu',weights_only=False)['state_dict']
            missing,unexpected=model.model_.load_state_dict(state,strict=False)
            assert not unexpected and all(not k.startswith('icl_predictor.') or '.tf_icl.blocks.' in k for k in missing)
            model._load_model=lambda:None;model._weights_cached_for_rebuttal=True
            self.models[1]=model
        return super().__call__(X,y,Q,**kwargs)


def make_bank(job,predictor,data,block,family,rho,role,scale,path,nc):
    if path.exists() and path.with_suffix('.json').exists():
        with np.load(path) as z:arrays={k:z[k] for k in z.files}
        return arrays,json.loads(path.with_suffix('.json').read_text())
    task=job['task'];X,y,Q=(data[k] for k in ('X_source','y_source','X_target'))
    cache={};timing={};rows={}
    def predict(name,A,b,V):
        key=tuple(protocol.array_sha256(a) for a in (A,b,V))
        rows[name]=len(b)
        if key not in cache:
            torch.cuda.synchronize();start=time.perf_counter()
            p,_=predictor(A,b,V,members=1,n_classes=nc,random_state=0)
            torch.cuda.synchronize();timing[name]=time.perf_counter()-start
            assert np.isfinite(p).all()
            cache[key]=(p,name)
        else:timing[name]=0.
        return cache[key][0]
    base=predict('FT',X,y,Q)
    torch.cuda.synchronize();start=time.perf_counter()
    shared=contexts.shared_contexts(X,y,Q,256,task,base)
    shared_construction=time.perf_counter()-start
    start=time.perf_counter()
    extra=contexts.specific_contexts(X,y,Q,256,task,block,family,base)
    specific_construction=time.perf_counter()-start
    names=[];members=[]
    for i,(name,A,b,V) in enumerate(shared+extra):
        unique=f'{i}:{name}';names.append(unique);members.append(predict(unique,A,b,V))
    assert len(shared)==5 and len(members)==10 and max(rows.values())<=256
    arrays=dict(base=base,bank=np.stack(members),y=data['y_target'],mu=np.array(float(y.mean())),
        sigma=np.array(max(float(y.std()),1e-8)))
    scaling.npwrite(path,**arrays)
    record=dict(task=task,track=job['track'],block=block,task_id=job['meta']['task_id'],
        dataset=job['meta']['dataset'],family=family,rho=rho,role=role,scale=scale,seed=0,
        source_rows=len(y),query_rows=len(Q),members=names,member_rows=rows,
        inference_seconds=timing,shared_construction_seconds=shared_construction,
        specific_extra_construction_seconds=specific_construction,unique_TFM_calls=len(cache),
        FT_inference_seconds=timing['FT'],FT_Shared_inference_seconds=sum(timing[n] for n in ['FT']+names[:5]),
        FT_Specific_inference_seconds=sum(timing.values()),
        input_hashes={k:protocol.array_sha256(a) for k,a in data.items()},
        original_query_labels_used_by_predictor_or_constructor=False)
    atomic_json(path.with_suffix('.json'),record)
    return arrays,record


def cell(arrays,record):
    return dict(task=record['task'],role=record['role'],base=arrays['base'],bank=arrays['bank'],y=arrays['y'],
        mu=float(arrays['mu']),sigma=float(arrays['sigma']),meta=record)


def evaluate(job,pool,split):
    dest=folder(job)
    if (dest/'completion.json').exists():return
    torch.cuda.reset_peak_memory_stats()
    task,track,meta=job['task'],job['track'],job['meta']
    predictor=UpdatedPredictor(job);X,y,_=flat(pool)
    nc=int(y.max())+1 if task=='classification' else None
    calibration_cells=[]
    for family,rho in [('original',0)]+[(f,r) for f in families(track) for r in (50,100)]:
        block='clean' if rho==0 else track
        path=dest/'calibration'/f'{family}_{rho}.npz'
        data=materialize(job,pool,np.r_[split['train'],split['val']],split['cal'],family,rho)
        scale=1. if task=='classification' else max(float(data['y_target'].std()),1e-8)
        arrays,record=make_bank(job,predictor,data,block,family,rho,'validation',scale,path,nc)
        calibration_cells.append(cell(arrays,record))
    start=time.perf_counter()
    # Fixed regularization and gate form; query scores do not select policies.
    shared_policy=fusion.fit(calibration_cells,task,5,1.,.01)
    specific_policies={}
    for family in ['original']+families(track):
        cases=[c for c in calibration_cells if c['meta']['family']==family]
        specific_policies[family]=fusion.fit(cases,task,10,1.,.01)
    fitting_seconds=time.perf_counter()-start
    atomic_json(dest/'policies.json',dict(status='FROZEN_BEFORE_QUERY_SCORING',Shared=shared_policy,Specific=specific_policies,
        final_query_labels_used=False,calibration_rows=len(split['cal']),calibration_conditions=len(calibration_cells),
        calibration_fitting_seconds=fitting_seconds,additional_clean_TRAIN_labels=True))
    blocks=['clean','five'] if track=='five' else ['seven']
    result=[]
    for block in blocks:
        for family,rho in contexts.case_grid(block):
            path=dest/'query'/block/f'{family}_{rho}.npz'
            data,info=scaling.bank(task,block,meta,family,rho)
            arrays,record=make_bank(job,predictor,data,block,family,rho,'query',info['scale'],path,nc)
            c=cell(arrays,record);p=arrays['base']
            start=time.perf_counter()
            predictions={'FT':p,'FT+Shared':fusion.predict(c,shared_policy),
                'FT+Specific':fusion.predict(c,specific_policies[family])}
            fusion_seconds=time.perf_counter()-start
            for name,pred in predictions.items():
                result.append(dict(task=task,block=block,task_id=meta['task_id'],data_id=meta['data_id'],dataset=meta['dataset'],
                    family=family,rho=rho,method=name,scale=info['scale'],query_rows=len(c['y']),
                    **metrics(c['y'],pred,task,info['scale']),
                    inference_seconds=record[{'FT':'FT_inference_seconds','FT+Shared':'FT_Shared_inference_seconds',
                        'FT+Specific':'FT_Specific_inference_seconds'}[name]],fusion_apply_seconds=fusion_seconds,
                    prediction_bank=str(path),semisynthetic=(task=='regression' and block in ['clean','five'])))
        print('FT_QUERY_BLOCK_COMPLETE',task,block,meta['task_id'],flush=True)
    pd.DataFrame(result).to_csv(dest/'query_detail.csv',index=False)
    atomic_json(dest/'completion.json',dict(status='COMPLETE',task=task,track=track,task_id=meta['task_id'],
        rows=len(result),calibration_fitting_seconds=fitting_seconds,peak_inference_CUDA_MiB=torch.cuda.max_memory_allocated()/2**20,
        raw_unfinetuned_Shared_Specific_not_rerun=True))
    del predictor;gc.collect();torch.cuda.empty_cache()


def worker(shard,shards,limit=0,only_task=None,steps=STEPS):
    torch.set_num_threads(1);warnings.filterwarnings('ignore')
    items=jobs()[shard::shards]
    if only_task:items=[j for j in items if j['task']==only_task]
    if limit:items=items[:limit]
    for job in items:
        try:
            print('JOB_START',job['task'],job['track'],job['meta']['task_id'],flush=True)
            pool=scaling.prepare(job['task'],job['track'],job['meta']);split=prepare_split(job,pool)
            train(job,pool,split,steps=steps)
            evaluate(job,pool,split)
            print('JOB_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(folder(job)/'error.json',dict(job=job,error=error,time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            gc.collect();torch.cuda.empty_cache()


def report():
    files=sorted((OUT/'jobs').glob('*/*/*/query_detail.csv'))
    if not files:return
    detail=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    # Pair historical Base with EXACTLY the completed dataset/case rows.
    # Never compare a partial FT average to the full cohort's Base average.
    baselines=[]
    for _,r in detail[detail['method']=='FT'].iterrows():
        path=frozen.OUT/'input_banks'/r['task']/r['block']/str(r['task_id'])/f"{r['family']}_{r['rho']}.json"
        record=json.loads(path.read_text())
        baselines.append({k:r[k] for k in ['task','block','task_id','data_id','dataset','family','rho','scale','query_rows','semisynthetic']}
                         |dict(method='Base (cached, paired)')|record['metrics'])
    detail=pd.concat([pd.DataFrame(baselines),detail],ignore_index=True)
    detail.to_csv(OUT/'query_detail.csv',index=False)
    rows=[]
    for keys,group in detail.groupby(['task','block','method']):
        task,block,method=keys
        expected=len(frozen.cohort(task,block))
        expected_conditions=expected*(1 if block=='clean' else (20 if block=='five' else 28))
        row=dict(task=task,block=block,method=method,datasets=group['task_id'].nunique(),expected_datasets=expected,
            conditions=len(group),complete=group['task_id'].nunique()==expected and len(group)==expected_conditions)
        fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        row.update({f:float(group[f].mean()) for f in fields});rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'averages.csv',index=False)
    cost=[]
    for p in sorted((OUT/'jobs').glob('*/*/*/training_result.json')):
        record=json.loads(p.read_text());cost.append({k:v for k,v in record.items() if k not in ['history','selected_checkpoint']})
    if cost:pd.DataFrame(cost).to_csv(OUT/'training_cost.csv',index=False)
    expected=len(jobs());done=len(list((OUT/'jobs').glob('*/*/*/completion.json')))
    errors=list((OUT/'jobs').glob('*/*/*/error.json'))
    unresolved=[p for p in errors if not (p.parent/'completion.json').exists()]
    atomic_json(OUT/'progress.json',dict(total_jobs=expected,complete_jobs=done,trained_jobs=len(cost),
        historical_error_attempts=len(errors),unresolved_errors=len(unresolved),full_complete=done==expected,
        report_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),Markdown_written=False))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('PROGRESS',done,'/',expected,'trained',len(cost),flush=True)


def launch():
    declare();children=[]
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    for gpu in range(8):
        logfile=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)}
        process=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
            env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(gpu=gpu,pid=process.pid,log=str(OUT/'logs'/f'gpu{gpu}.log')))
        logfile.close()
    atomic_json(OUT/'workers.json',children);print(json.dumps(children),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['declare','worker','report','launch'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--task',default=None)
    p.add_argument('--steps',type=int,default=STEPS);a=p.parse_args()
    if a.phase=='declare':declare()
    elif a.phase=='report':report()
    elif a.phase=='launch':launch()
    else:worker(a.shard,a.shards,a.limit,a.task,a.steps)
