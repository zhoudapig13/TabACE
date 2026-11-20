"""Accuracy-aligned gradient adaptation, restricted to the active endpoint."""
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
import serial_heavy_classification_v331 as endpoint
import accuracy_aligned_classifier_v333 as implementation
from prepare_data import RUN,ROOT,atomic_json,protocol

source=endpoint.source
OUT=RUN/'results/heavy_accuracy_finetuning_v334'


def verify_frozen():
    registry=RUN/'results/frozen_endpoints_v330/registry.json'
    data=json.loads(registry.read_text())
    for item in data['endpoints']:
        for row in item['conditions']:
            assert source.sha(row['prediction_file'])==row['prediction_sha256']
            assert source.sha(row['metadata_file'])==row['metadata_sha256']
    return source.sha(registry)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);cases=endpoint.endpoint_cases();assert len(cases)==64
    diagnostic=json.loads((RUN/'results/reference_membership_diagnostic_v320/diagnostic.json').read_text())
    values={r['task_id']:r['source_reference_prior_TV'] for r in diagnostic['Results'] if r['track']=='perturbation' and r['task']=='classification'}
    ordered=sorted(cases,key=lambda c:(values[c['task_id']],c['task_id']))
    pilot_ids=[ordered[i]['task_id'] for i in [0,len(ordered)//2,len(ordered)-1]]
    frozen_sha=verify_frozen()
    p=OUT/'case_manifest.json'
    if p.exists():assert json.loads(p.read_text())==cases
    else:atomic_json(p,cases)
    plan={'active_endpoint':list(endpoint.ENDPOINT),'seed':0,'cases':64,'pilot_task_ids':pilot_ids,
        'pilot_selection':'minimum, median and maximum source/reference prior discrepancy; no evaluation scores used',
        'training':'clean source256, existing independent clean source references, first75percent of v311 target-adaptation labels repeated4times',
        'validation':'reserved64 clean source reference rows plus last25percent v311 target-adaptation labels',
        'validation_primary':'mean of source ordinary accuracy and target ordinary accuracy; equal-domain NLL tie-break',
        'validation_context':'source heavy-tail severity0.8 plus gradient-training target rows, total<=256',
        'loss':'ordinary unweighted cross entropy; class-subset episodes for >10 classes',
        'epochs':20,'learning_rate':5e-5,'weight_decay':.01,'patience':5,'time_limit_per_dataset_seconds':120,
        'warm_start':'source-only v307 checkpoint, disjoint from entire original query',
        'freeze_column_encoder':True,'train_row_and_ICL':True,'meta_batch_rows_max':320,
        'context_query_ratio':.2,'norm_methods':['none'],'estimators':1,
        'target_validation_labels_not_used_in_gradients_or_validation_context':True,
        'heldout_evaluation_labels_never_opened':True,'extra_source_and_test_subset_supervision_explicit':True,
        'not_original_zero_shot_Shared':True,'previously_scored_development_cohort':True,
        'frozen_registry_sha256':frozen_sha,'GPU_policy':json.loads(endpoint.oldbank.training.implementation.GPU_POLICY.read_text()),
        'script_sha256':source.sha(__file__),'implementation_sha256':source.sha(implementation.__file__),
        'compatibility_sha256':source.sha(implementation.compatible.__file__),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def train_one(c,gpu):
    assert (c['track'],c['task'])==endpoint.ENDPOINT[:2]
    tid=c['task_id'];dest=OUT/'checkpoints'/str(tid)
    if (dest/'result.json').exists():
        old=json.loads((dest/'result.json').read_text());assert source.sha(old['checkpoint_file'])==old['checkpoint_sha256'];return
    assert source.sha(c['input_file'])==c['input_sha256']
    with np.load(c['input_file']) as z:X0,y0,Q,num,sid,qid=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
    assert source.sha(c['reference_file'])==c['reference_sha256']
    with np.load(c['reference_file']) as z:
        RX,Ry,rid,CX,Cy,cid=(z[k] for k in ('X_train_reference','y_train_reference','train_reference_indices','X_calibration','y_calibration','calibration_indices'))
    assert set(np.r_[sid,rid,cid]).isdisjoint(qid) and set(cid).isdisjoint(np.r_[sid,rid]) and set(rid).isdisjoint(sid)
    assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
    with np.load(c['adaptation_file']) as z:apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
    nval=max(4,len(apos)//4);train_a,val_a=np.arange(len(apos)-nval),np.arange(len(apos)-nval,len(apos))
    assert set(apos[train_a]).isdisjoint(apos[val_a]) and set(apos).isdisjoint(epos)
    trainX=np.r_[X0,RX,np.tile(Q[apos[train_a]],(4,1))]
    trainy=np.r_[y0,Ry,np.tile(ay[train_a],4)]
    trainids=np.r_[sid,rid,np.tile(qid[apos[train_a]],4)]
    valX,valy=np.r_[CX,Q[apos[val_a]]],np.r_[Cy,ay[val_a]]
    valid=np.r_[cid,qid[apos[val_a]]]
    assert set(trainids).isdisjoint(valid) and set(np.r_[trainids,valid]).isdisjoint(qid[epos])
    observed,labels,_,_=source.novel.generate(X0,y0,Q,num,'classification','cell_heavy_tail',80,tid)
    contextX,contexty,contextmeta=endpoint.reserve_target('canonical',observed,labels,Q,Q[apos[train_a]],ay[train_a],'classification')
    warm=json.loads((endpoint.oldbank.training.OUT/'checkpoints'/'perturbation'/'classification'/str(tid)/'result.json').read_text())
    assert source.sha(warm['checkpoint_file'])==warm['checkpoint_sha256']
    model=implementation.AccuracyClassifier(epochs=20,learning_rate=5e-5,weight_decay=.01,freeze_col=True,freeze_row=False,
        n_estimators_finetune=1,n_estimators_validation=1,n_estimators_inference=1,
        max_data_size=320,finetune_ctx_query_ratio=.2,norm_methods=['none'],
        model_path=warm['checkpoint_file'],allow_auto_download=False,device='cuda:0',random_state=0,
        early_stopping=True,patience=5,time_limit=120,save_interval=10000,verbose=True,eval_metric='accuracy')
    model.context_X_,model.context_y_=contextX,contexty;model.source_validation_count_=len(Cy)
    atomic_json(dest/'input_audit.json',{'task_id':tid,'active_endpoint':list(endpoint.ENDPOINT),'actual_GPU':gpu,
        'input_sha256':c['input_sha256'],'reference_sha256':c['reference_sha256'],'adaptation_sha256':c['adaptation_sha256'],
        'warm_checkpoint_sha256':warm['checkpoint_sha256'],'source_training_rows':len(y0)+len(Ry),
        'source_reserved_validation_rows':len(Cy),'unique_target_gradient_training_rows':len(train_a),
        'target_training_repetition_factor':4,'target_validation_rows':len(val_a),
        'target_training_positions':apos[train_a].tolist(),'target_validation_positions':apos[val_a].tolist(),
        'heldout_evaluation_positions':epos.tolist(),'gradient_training_row_ids_sha256':protocol.array_sha256(trainids),
        'validation_row_ids_sha256':protocol.array_sha256(valid),'validation_context':contextmeta,
        'validation_context_X_sha256':protocol.array_sha256(contextX),'validation_context_y_sha256':protocol.array_sha256(contexty),
        'all_gradient_and_validation_rows_disjoint_from_evaluation':True,'heldout_eval_labels_opened':False})
    torch.manual_seed(0);np.random.seed(0)
    model.fit(trainX,trainy,X_val=valX,y_val=valy,output_dir=dest)
    selected=implementation.compatible.model_digest(model.model_)
    atomic_json(dest/'result.json',{'status':'COMPLETE','task_id':tid,'active_endpoint':list(endpoint.ENDPOINT),'actual_GPU':gpu,
        'checkpoint_file':str(dest/'best.ckpt'),'checkpoint_sha256':source.sha(dest/'best.ckpt'),
        'initial_model_sha256':model.initial_model_sha256_,'selected_model_sha256':selected,
        'selected_weights_changed':selected!=model.initial_model_sha256_,'training_loss_calls':model.loss_calls_,
        'validation_trace':model.validation_trace_,'selected_validation_score':float(model._best_metric_),
        'selected_validation_NLL':model._best_nll_,'test_subset_used_for_supervised_training_and_model_selection':True,
        'heldout_eval_labels_opened':False})
    print('HEAVY_ACCURACY_FT_CASE',tid,'changed',selected!=model.initial_model_sha256_,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


def worker(shard,shards,pilot):
    torch.set_num_threads(1);gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    plan=json.loads((OUT/'plan.json').read_text());assert gpu in plan['GPU_policy']['allowed_GPUs']
    assert source.sha(__file__)==plan['script_sha256'] and source.sha(implementation.__file__)==plan['implementation_sha256']
    cases=json.loads((OUT/'case_manifest.json').read_text())
    if pilot:cases=[c for c in cases if c['task_id'] in plan['pilot_task_ids']]
    for i,c in enumerate(cases):
        if i%shards==shard:train_one(c,gpu)


def launch(pilot):
    prepare();plan=json.loads((OUT/'plan.json').read_text());workers=3 if pilot else 6;prefix='pilot' if pilot else 'full'
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i);log=OUT/f'{prefix}_worker_{i}.log'
        command=[sys.executable,__file__,'worker','--shard',str(i),'--shards',str(workers)]+(['--pilot'] if pilot else [])
        with log.open('a') as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'GPU':i,'returncode':p.returncode,'log':str(log)}
    atomic_json(OUT/f'{prefix}_execution.json',{'status':'RUNNING','workers':workers,'active_endpoint':list(endpoint.ENDPOINT)})
    with ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(execute,range(workers)))
    complete=all(r['returncode']==0 for r in results)
    atomic_json(OUT/f'{prefix}_execution.json',{'status':'COMPLETE' if complete else 'FAILED','workers':results,'active_endpoint':list(endpoint.ENDPOINT)})
    verify_frozen();assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    assert complete


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','launch','worker']);p.add_argument('--pilot',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='launch':launch(a.pilot)
    else:worker(a.shard,a.shards,a.pilot)
