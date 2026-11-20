"""Six-GPU clean-source supervised fine-tuning feasibility pilot.

This is a distinct, explicitly supervised development extension. Source training
data are used before synthetic corruption, so it cannot be described as frozen
Shared robustness. No final query labels, query scores, or test-adaptation labels
are opened. Final inference checkpoints will still be used with context<=256.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from tabicl import FinetunedTabICLClassifier,FinetunedTabICLRegressor
from tabicl._finetune.base import ValidationMetrics
import tabicl._finetune.data as episode_data
import unseen_shared_robustness_v117 as source
from prepare_data import RUN,ROOT,atomic_json
from run_tabicl import CHECKPOINTS

OUT=RUN/'results/shared_finetune_pilot_v298'
GPU_POLICY=RUN/'configs/rebuttal_gpu_policy.json'


_original_episode_split=episode_data._split_ctx_query


def safe_episode_split(y_chunk,query_size,seed,stratify):
    try:
        return _original_episode_split(y_chunk,query_size=query_size,seed=seed,stratify=stratify)
    except ValueError:
        if not stratify:raise
        rng=np.random.default_rng(seed)
        y=np.asarray(y_chunk)
        mandatory=np.asarray([rng.choice(np.flatnonzero(y==c)) for c in np.unique(y)],dtype=int)
        candidates=np.setdiff1d(np.arange(len(y)),mandatory)
        size=min(int(query_size),len(candidates))
        if size<1:raise
        query=rng.choice(candidates,size,replace=False)
        context=np.setdiff1d(np.arange(len(y)),query)
        assert set(y[query]).issubset(set(y[context]))
        return context,query


def clone_frozen_training_view(module,args):
    if module.training and torch.is_grad_enabled():
        return (args[0].clone(),)+args[1:]
    return args


episode_data._split_ctx_query=safe_episode_split


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def class_split(y,n):
    rng=np.random.default_rng(0);groups=[rng.permutation(np.flatnonzero(y==c)).tolist() for c in np.unique(y)]
    selected=[]
    while len(selected)<n:
        changed=False
        for rows in groups:
            if len(rows)>1 and len(selected)<n:selected.append(rows.pop());changed=True
        if not changed:break
    val=np.asarray(selected,dtype=int);train=np.setdiff1d(np.arange(len(y)),val)
    return train,val


def validation_context(y,classification):
    if len(y)<=256:return np.arange(len(y))
    if classification:
        # Balanced validation context with capacity constraints; never duplicate.
        rng=np.random.default_rng(0);groups=[rng.permutation(np.flatnonzero(y==c)).tolist() for c in np.unique(y)]
        rows=[]
        while len(rows)<256:
            for group in groups:
                if group and len(rows)<256:rows.append(group.pop())
        return np.asarray(rows,dtype=int)
    return np.random.default_rng(0).choice(len(y),256,replace=False)


class CaptureMixin:
    def _load_pretrained(self,device):
        super()._load_pretrained(device)
        self.initial_model_sha256_=model_digest(self.model_)
        self.loss_calls_=0;self.validation_trace_=[]
        self._training_view_hook_=self.model_.icl_predictor.register_forward_pre_hook(clone_frozen_training_view)

    def _compute_batch_loss(self,batch,model):
        self.loss_calls_+=1
        return super()._compute_batch_loss(batch,model)

    def _run_validation(self,inner,X_train,y_train,X_val,y_val):
        rows=validation_context(y_train,self._model_type=='classifier')
        root=getattr(getattr(inner,'model_',None),'icl_predictor',None)
        if root is not None and hasattr(root,'root'):root.root=None
        if self._model_type=='classifier':
            inner.fit(X_train[rows],y_train[rows]);p=inner.predict_proba(X_val)
            pred=inner.classes_[p.argmax(1)]
            result=ValidationMetrics(primary=float(balanced_accuracy_score(y_val,pred)),
                secondary={'accuracy':float(accuracy_score(y_val,pred))})
        else:
            result=super()._run_validation(inner,X_train[rows],y_train[rows],X_val,y_val)
        self.validation_trace_.append({'primary':result.primary,'secondary':result.secondary,'context_rows':len(rows)})
        return result


class PilotClassifier(CaptureMixin,FinetunedTabICLClassifier):
    @property
    def _metric_name(self):return 'balanced_accuracy'

    def _compute_batch_loss(self,batch,model):
        self.loss_calls_+=1
        # Training logits support max_classes; inference handles larger label
        # spaces hierarchically. Train class-subset episodes for those datasets.
        losses=[]
        limit=int(self.model_.max_classes)
        for i in range(len(batch.X)):
            yc=batch.y_train[i].long();yq=batch.y_query[i].long()
            labels=torch.unique(yc)
            if len(labels)>limit:
                present=torch.unique(yq);present=present[torch.isin(present,labels)]
                first=present[(self.loss_calls_-1)%len(present)]
                remaining=labels[labels!=first]
                order=torch.randperm(len(remaining),device=remaining.device)
                labels=torch.sort(torch.cat([first[None],remaining[order[:limit-1]]])).values
            mc=torch.isin(yc,labels);mq=torch.isin(yq,labels)
            assert mq.any()
            train=torch.searchsorted(labels,yc[mc]);target=torch.searchsorted(labels,yq[mq])
            X=torch.cat([batch.X[i,:batch.train_size][mc],batch.X[i,batch.train_size:][mq]])[None]
            logits=model(X,train.float()[None])[...,:len(labels)]
            counts=torch.bincount(train,minlength=len(labels)).float()
            weight=counts.sum()/counts.clamp_min(1);weight/=weight.mean()
            losses.append(F.cross_entropy(logits.reshape(-1,len(labels)),target,weight=weight))
        return torch.stack(losses).mean()


class PilotRegressor(CaptureMixin,FinetunedTabICLRegressor):pass


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    refs={(r['task'],r['task_id']):r for r in json.loads((RUN/'results/expanded_clean_reference_pool_v278/reference_manifest.json').read_text())}
    cases=[c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track']=='perturbation']
    selected=[]
    for task in ('classification','regression'):
        pool=sorted([c for c in cases if c['task']==task],key=lambda c:c['task_id'])
        for lower,upper in [(0,10),(11,50),(51,10**9)]:
            options=[c for c in pool if lower<=c['features']<=upper]
            assert options
            chosen=options[0]
            if task=='classification' and lower==51:
                many=[]
                for c in options:
                    with np.load(c['input_file']) as z:
                        if len(np.unique(z['y_source']))>10:many.append(c)
                if many:chosen=many[0]
            r=refs[(task,chosen['task_id'])]
            selected.append(dict(chosen,reference_file=r['reference_file'],reference_sha256=r['reference_sha256']))
    assert len(selected)==6
    atomic_json(OUT/'pilot_manifest.json',selected)
    atomic_json(OUT/'plan.json',{'status':'CLEAN_SOURCE_SUPERVISED_BACKBONE_FINE_TUNING_PILOT',
        'seed':0,'allowed_GPUs':[0,1,2,3,4,5],'reserved_GPUs':[6,7],
        'selection':'lowest task_id in three feature-count bins per task; high-dimensional classification exercises >10 classes when available',
        'pilot_datasets':6,'not_full_benchmark_evidence':True,
        'training_implementation_fixes':['clone frozen no_grad representation before label-embedding inplace update','keep singleton-label training rows in context when stratified episodic split cannot run'],
        'upstream_training_data_sha256':source.sha(episode_data.__file__),
        'previous_failed_pilot_preserved':'shared_finetune_pilot_v297',
        'training_data':'clean pre-corruption source256 plus independent clean labeled training references',
        'clean_pre_corruption_training_is_extra_supervision':True,
        'final_query_labels_or_scores_opened':False,'target_test_adaptation_labels_used':False,
        'validation':'128 source-training rows, class-balanced with one row per class retained for training',
        'validation_context_cap':256,'meta_batch_max_rows':320,'context_query_ratio':.2,
        'epochs':3,'learning_rate':1e-5,'freeze_column_and_row_encoders':True,
        'class_loss':'inverse-context-frequency CE; class-subset episodes if >max_classes',
        'checkpoint_selection':'independent source validation, best can remain pretrained',
        'script_sha256':source.sha(__file__),'Markdown_written':False})


def worker(index):
    policy=json.loads(GPU_POLICY.read_text());visible=int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert visible in policy['allowed_GPUs'] and visible not in policy['reserved_GPUs']
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    c=json.loads((OUT/'pilot_manifest.json').read_text())[index];task,tid=c['task'],c['task_id']
    assert source.sha(c['input_file'])==c['input_sha256']
    with np.load(c['input_file']) as z:
        X0,y0,sid,qid=(z[k] for k in ('X_source','y_source','source_indices','target_indices'))
    assert source.sha(c['reference_file'])==c['reference_sha256']
    with np.load(c['reference_file']) as z:
        TX,Ty,tids=z['X_train_reference'],z['y_train_reference'],z['train_reference_indices']
    assert set(map(int,np.r_[sid,tids])).isdisjoint(map(int,qid))
    X,y=np.r_[X0,TX],np.r_[y0,Ty]
    if task=='classification':train,val=class_split(y,128)
    else:
        p=np.random.default_rng(0).permutation(len(y));val,train=p[:128],p[128:]
    assert set(train).isdisjoint(val)
    cls=PilotClassifier if task=='classification' else PilotRegressor
    name=f'tabicl-{"classifier" if task=="classification" else "regressor"}-v2-20260212.ckpt'
    kwargs={'eval_metric':'accuracy'} if task=='classification' else {}
    est=cls(epochs=3,learning_rate=1e-5,freeze_col=True,freeze_row=True,
        n_estimators_finetune=1,n_estimators_validation=1,n_estimators_inference=1,
        max_data_size=320,finetune_ctx_query_ratio=.2,norm_methods=['none'],
        model_path=str(CHECKPOINTS/name),allow_auto_download=False,device='cuda:0',
        random_state=0,early_stopping=True,patience=2,time_limit=120,
        save_interval=10000,verbose=True,**kwargs)
    dest=OUT/'checkpoints'/task/str(tid)
    atomic_json(dest/'input_audit.json',{'training_rows':len(train),'validation_rows':len(val),
        'train_pool_positions':train.tolist(),'validation_pool_positions':val.tolist(),
        'all_train_and_validation_row_ids_disjoint_from_final_query':True,
        'source_clean_features_used_before_synthetic_corruption':True,'actual_GPU':visible})
    est.fit(X[train],y[train],X_val=X[val],y_val=y[val],output_dir=dest)
    final=model_digest(est.model_)
    atomic_json(dest/'result.json',{'status':'COMPLETE','task':task,'task_id':tid,'actual_GPU':visible,
        'training_loss_calls':est.loss_calls_,'validation_trace':est.validation_trace_,
        'initial_model_sha256':est.initial_model_sha256_,'selected_model_sha256':final,
        'selected_weights_differ_from_pretrained':final!=est.initial_model_sha256_,
        'checkpoint_file':str(dest/'best.ckpt'),'checkpoint_sha256':source.sha(dest/'best.ckpt'),
        'no_final_query_labels_opened':True,'pilot_only':True})
    print('FINETUNE_PILOT_COMPLETE',task,tid,'updates',est.loss_calls_,'selected_changed',final!=est.initial_model_sha256_,flush=True)


def launch():
    prepare();policy=json.loads(GPU_POLICY.read_text());assert policy['allowed_GPUs']==list(range(6))
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(policy['allowed_GPUs'][i]);log=OUT/f'worker_{i}.log'
        with log.open('a') as f:p=subprocess.run([sys.executable,__file__,'worker','--index',str(i)],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'index':i,'GPU':policy['allowed_GPUs'][i],'returncode':p.returncode,'log':str(log)}
    results=[];atomic_json(OUT/'execution.json',{'status':'RUNNING','allowed_GPUs':policy['allowed_GPUs'],'reserved_GPUs':policy['reserved_GPUs']})
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(execute,i) for i in range(6)]):
            results.append(future.result());atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':results})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if all(r['returncode']==0 for r in results) else 'FAILED',
        'workers':results,'allowed_GPUs':policy['allowed_GPUs'],'reserved_GPUs':policy['reserved_GPUs']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    launch() if a.phase=='launch' else worker(a.index)
