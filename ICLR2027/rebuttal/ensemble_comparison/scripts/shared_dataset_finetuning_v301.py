"""Dataset-supervised fine-tuning on source rows only, six GPUs maximum.

Separate models are trained per track/task so a different track's evaluation
rows cannot enter a reused checkpoint. Clean pre-corruption source features and
external training references are explicit additional supervised information.
No final-query labels or target-adaptation labels are read by this stage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gc
import json
import os
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import torch
import shared_finetune_pilot_v300 as implementation
from prepare_data import RUN,ROOT,atomic_json,protocol
from run_tabicl import CHECKPOINTS

source=implementation.source
OUT=RUN/'results/shared_dataset_finetuning_v301'


def prepare():
    policy=json.loads(implementation.GPU_POLICY.read_text())
    assert policy['allowed_GPUs']==list(range(6)) and policy['reserved_GPUs']==[6,7]
    OUT.mkdir(parents=True,exist_ok=True)
    refs={(r['task'],r['task_id']):r for r in json.loads((RUN/'results/expanded_clean_reference_pool_v278/reference_manifest.json').read_text())}
    shifts={(r['task'],r['task_id']):r for r in json.loads((RUN/'results/shift_target_adaptation_manifest_v285/case_manifest.json').read_text())}
    cases=[]
    for original in json.loads((source.OUT/'case_manifest.json').read_text()):
        c=dict(original)
        if c['track']=='perturbation':
            r=refs[(c['task'],c['task_id'])]
            c.update(reference_file=r['reference_file'],reference_sha256=r['reference_sha256'])
        else:
            r=shifts[(c['task'],c['task_id'])]
            c.update(reference_file=r['adaptation_input_file'],reference_sha256=r['adaptation_input_sha256'])
        cases.append(c)
    assert len(cases)==175
    p=OUT/'case_manifest.json'
    if p.exists():assert json.loads(p.read_text())==cases
    else:atomic_json(p,cases)
    plan={'status':'CLEAN_SOURCE_DATASET_SUPERVISED_BACKBONE_FINE_TUNING',
        'seed':0,'cases':175,'separate_models_by_track_and_task':True,
        'training_data':'pre-corruption clean source256 plus independent clean labeled references, up to1984',
        'source_clean_features_before_synthetic_corruption_explicitly_used':True,
        'new_supervision_not_original_frozen_Shared':True,
        'final_query_or_target_adaptation_labels_used_for_weights':False,
        'source_validation':'up to128 rows, rare-class retention fixed as pilot300',
        'epochs':20,'learning_rate':1e-5,'weight_decay':.01,'patience':5,'time_limit_per_dataset_seconds':120,
        'freeze_column_encoder':True,'train_row_and_ICL_modules':True,
        'classification_selection':'balanced accuracy then balanced log-loss',
        'regression_selection':'R2, min_delta1e-4',
        'meta_batch_rows_max':320,'context_query_ratio':.2,'validation_context_cap':256,
        'final_query_inference_context_cap_required':256,
        'training_estimators':1,'norm_methods':['none'],
        'allowed_GPUs':policy['allowed_GPUs'],'reserved_GPUs':policy['reserved_GPUs'],
        'case_manifest_sha256':source.sha(p),'script_sha256':source.sha(__file__),
        'implementation_sha256':source.sha(implementation.__file__),
        'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def train_one(c,gpu):
    torch.manual_seed(0);np.random.seed(0)
    track,task,tid=c['track'],c['task'],c['task_id']
    dest=OUT/'checkpoints'/track/task/str(tid)
    if (dest/'result.json').exists():
        old=json.loads((dest/'result.json').read_text())
        assert old['status']=='COMPLETE' and source.sha(old['checkpoint_file'])==old['checkpoint_sha256']
        return old
    assert source.sha(c['input_file'])==c['input_sha256']
    with np.load(c['input_file']) as z:
        assert 'y_target' not in z.files
        X0,y0,sid,qid=(z[k] for k in ('X_source','y_source','source_indices','target_indices'))
    assert source.sha(c['reference_file'])==c['reference_sha256']
    with np.load(c['reference_file']) as z:
        TX,Ty,tids=(z[k] for k in ('X_train_reference','y_train_reference','train_reference_indices'))
    ids=np.r_[sid,tids];assert len(set(map(int,ids)))==len(ids)
    assert set(map(int,ids)).isdisjoint(map(int,qid))
    X,y=np.r_[X0,TX],np.r_[y0,Ty]
    if task=='classification':train,val=implementation.class_split(y,128)
    else:
        order=np.random.default_rng(0).permutation(len(y));val,train=order[:128],order[128:]
    assert set(train).isdisjoint(val)
    cls=implementation.PilotClassifier if task=='classification' else implementation.PilotRegressor
    name=f'tabicl-{"classifier" if task=="classification" else "regressor"}-v2-20260212.ckpt'
    metric='accuracy' if task=='classification' else 'r2'
    model=cls(epochs=20,learning_rate=1e-5,weight_decay=.01,freeze_col=True,freeze_row=False,
        n_estimators_finetune=1,n_estimators_validation=1,n_estimators_inference=1,
        max_data_size=320,finetune_ctx_query_ratio=.2,norm_methods=['none'],
        model_path=str(CHECKPOINTS/name),allow_auto_download=False,device='cuda:0',
        random_state=0,early_stopping=True,patience=5,time_limit=120,
        save_interval=10000,verbose=True,eval_metric=metric)
    atomic_json(dest/'input_audit.json',{'track':track,'task':task,'task_id':tid,
        'original_input_sha256':c['input_sha256'],'reference_sha256':c['reference_sha256'],
        'training_rows':len(train),'validation_rows':len(val),'extra_labeled_source_references':len(Ty),
        'train_pool_positions':train.tolist(),'validation_pool_positions':val.tolist(),
        'training_row_ids_sha256':protocol.array_sha256(ids[train]),
        'validation_row_ids_sha256':protocol.array_sha256(ids[val]),
        'final_query_row_ids_sha256':protocol.array_sha256(qid),
        'source_train_validation_disjoint_from_entire_original_query':True,
        'target_adaptation_labels_used_for_weights':False,'source_clean_features_pre_corruption_used':True,'actual_GPU':gpu})
    model.fit(X[train],y[train],X_val=X[val],y_val=y[val],output_dir=dest)
    selected=implementation.model_digest(model.model_)
    result={'status':'COMPLETE','track':track,'task':task,'task_id':tid,'actual_GPU':gpu,
        'training_loss_calls':model.loss_calls_,'validation_trace':model.validation_trace_,
        'initial_model_sha256':model.initial_model_sha256_,'selected_model_sha256':selected,
        'selected_weights_changed':selected!=model.initial_model_sha256_,
        'selected_validation_primary':float(model._best_metric_),
        'selected_balanced_nll':getattr(model,'_best_balanced_nll_',None),
        'checkpoint_file':str(dest/'best.ckpt'),'checkpoint_sha256':source.sha(dest/'best.ckpt'),
        'no_final_query_labels_or_scores_opened':True,'target_adaptation_labels_used_for_weights':False,
        'extra_supervision_type':'CLEAN_PRE_CORRUPTION_SOURCE_DATASET_LABELS_AND_INDEPENDENT_REFERENCES'}
    atomic_json(dest/'result.json',result)
    print('DATASET_FINETUNE_COMPLETE',track,task,tid,'steps',model.loss_calls_,'changed',result['selected_weights_changed'],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def worker(shard,shards):
    torch.set_num_threads(1)
    policy=json.loads(implementation.GPU_POLICY.read_text());gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert gpu in policy['allowed_GPUs'] and gpu not in policy['reserved_GPUs']
    plan=json.loads((OUT/'plan.json').read_text())
    assert source.sha(__file__)==plan['script_sha256'] and source.sha(implementation.__file__)==plan['implementation_sha256']
    cases=json.loads((OUT/'case_manifest.json').read_text())
    for i,pos in enumerate(np.random.default_rng(301).permutation(len(cases))):
        if i%shards==shard:train_one(cases[int(pos)],gpu)


def launch():
    prepare();policy=json.loads(implementation.GPU_POLICY.read_text());allowed=policy['allowed_GPUs']
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(allowed[i]);log=OUT/f'worker_{i}.log'
        with log.open('a') as f:p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i),'--shards',str(len(allowed))],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'GPU':allowed[i],'returncode':p.returncode,'log':str(log)}
    done=[];atomic_json(OUT/'execution.json',{'status':'RUNNING','allowed_GPUs':allowed,'reserved_GPUs':policy['reserved_GPUs']})
    with ThreadPoolExecutor(max_workers=len(allowed)) as pool:
        for future in as_completed([pool.submit(execute,i) for i in range(len(allowed))]):
            done.append(future.result());atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':done})
    count=len(list((OUT/'checkpoints').glob('*/*/*/result.json')))
    success=all(r['returncode']==0 for r in done) and count==175
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if success else 'FAILED','workers':done,
        'completed_datasets':count,'allowed_GPUs':allowed,'reserved_GPUs':policy['reserved_GPUs']})
    assert success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6)
    a=p.parse_args();launch() if a.phase=='launch' else worker(a.shard,a.shards)
