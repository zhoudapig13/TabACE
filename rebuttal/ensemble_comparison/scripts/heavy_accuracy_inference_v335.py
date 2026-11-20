"""Paired warm/new weights, identical contexts, frozen Shared fusion, heavy only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import torch
from tabicl import TabICLClassifier
import heavy_accuracy_finetuning_v334 as training
from prepare_data import RUN,ROOT,atomic_json,protocol
from run_tabicl import Predictor

source=training.source
OUT=RUN/'results/heavy_accuracy_inference_v335'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'seed':0,'active_endpoint':list(training.endpoint.ENDPOINT),'cases':64,'conditions':256,
        'weight_versions':['source-only warm307','accuracy-aligned target-subset gradient adapted334'],
        'inference_membership':'original observed source, plus all32 v311 target-adaptation labels, context<=256',
        'registry':'unchanged original five Shared contexts, no enlarged inference reference pool',
        'fusion':'unchanged dataset-OOF v77 Shared gate','final_query_labels_or_scores_used_for_selection':False,
        'test_subset_gradient_and_context_supervision_explicit':True,'extra_source_reserved64_validation_labels_explicit':True,
        'same_unique_test_label_budget_for_Warm_and_Adapted_and_Native_and_Shared':True,
        'no_crossfit_or_meta_fit_on_gradient_training_rows':True,'not_original_zero_shot_Shared':True,
        'previously_scored_development_cohort':True,'frozen_registry_sha256':training.verify_frozen(),
        'training_plan_sha256':source.sha(training.OUT/'plan.json'),'script_sha256':source.sha(__file__),
        'generator_sha256':source.sha(source.novel.__file__),'registry_sha256':source.sha(source.constructors.__file__),
        'fusion_sha256':source.sha(source.fusion.__file__),'GPU_policy':json.loads((training.OUT/'plan.json').read_text())['GPU_policy'],
        'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def worker(shard,shards,pilot):
    torch.set_num_threads(1);torch.manual_seed(0);gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in range(6)
    tp=json.loads((training.OUT/'plan.json').read_text())
    cases=json.loads((training.OUT/'case_manifest.json').read_text())
    if pilot:cases=[c for c in cases if c['task_id'] in tp['pilot_task_ids']]
    for i,c in enumerate(cases):
        if i%shards!=shard:continue
        tid=c['task_id'];assert c['task']=='classification' and c['track']=='perturbation'
        fitted=json.loads((training.OUT/'checkpoints'/str(tid)/'result.json').read_text())
        warm=json.loads((training.endpoint.oldbank.training.OUT/'checkpoints'/'perturbation'/'classification'/str(tid)/'result.json').read_text())
        versions={'Warm':warm,'Adapted':fitted};predictors={}
        for name,cp in versions.items():
            assert source.sha(cp['checkpoint_file'])==cp['checkpoint_sha256']
            predictor=Predictor('classification','cuda:0')
            predictor.models[1]=TabICLClassifier(n_estimators=1,model_path=cp['checkpoint_file'],allow_auto_download=False,
                device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
            predictors[name]=predictor
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:X0,y0,Q0,num=(z[k] for k in ('X_source','y_source','X_target','numeric_columns'))
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        pfile=source.OUT/'policies'/'classification'/f'{c["fold"]}.json';gate=json.loads(pfile.read_text())
        assert c['data_id'] not in gate['training_data_ids']
        for rho in source.novel.STRENGTHS:
            family='cell_heavy_tail';dest=OUT/'predictions'/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];continue
            X,y,Qfull,_=source.novel.generate(X0,y0,Q0,num,'classification',family,rho,tid)
            Q,AX=Qfull[epos],Qfull[apos]
            views=[('canonical',X,y,Q)]+source.constructors.shared_contexts(X,y,Q,len(y),'classification',None)
            order=source.constructors.specific.importance(X,y.astype(float))
            columns=np.array([j for j in range(X.shape[1]) if j!=int(order[0])])
            if not len(columns):columns=np.array([0])
            contexts=[];contextmeta=[]
            for name,A,b,T in views:
                V,labels,record=training.endpoint.reserve_target(name,A,b,T,AX,ay,'classification',columns)
                contexts.append((V,labels,T))
                contextmeta.append(record|{'name':name,'X_sha256':protocol.array_sha256(V),'y_sha256':protocol.array_sha256(labels),'query_X_sha256':protocol.array_sha256(T)})
            mu,sigma=float(contexts[0][1].mean()),max(float(contexts[0][1].std()),1e-12)
            original=source.OUT/'predictions'/'perturbation'/'classification'/str(tid)/dest.name
            originalmeta=json.loads(original.with_suffix('.json').read_text());assert source.sha(original)==originalmeta['output_sha256']
            with np.load(original) as z:arrays={'Base':z['Base'][epos]}
            for name,predictor in predictors.items():
                predictions=[]
                for A,b,T in contexts:
                    for estimator in predictor.models.values():
                        root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                        if root is not None and hasattr(root,'root'):root.root=None
                    torch.cuda.empty_cache()
                    predictions.append(predictor(A,b,T,members=1,random_state=0,n_classes=len(np.unique(y0)))[0])
                core,bank=predictions[0],np.stack(predictions[1:])
                arrays[name+'_Native']=core;arrays[name+'_members']=bank
                arrays[name+'_Shared']=source.fusion.predict({'task':'classification','base':core,'bank':bank,'mu':mu,'sigma':sigma},gate['Shared'])
            source.npwrite(dest,**arrays)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('task','track','task_id','data_id','fold')}|
                {'family':family,'rho':rho,'output_sha256':source.sha(dest),'actual_GPU':gpu,
                 'input_sha256':c['input_sha256'],'adaptation_sha256':c['adaptation_sha256'],
                 'evaluation_positions':epos.tolist(),'target_adaptation_positions':apos.tolist(),
                 'checkpoint_sha256':{name:cp['checkpoint_sha256'] for name,cp in versions.items()},
                 'policy_sha256':source.sha(pfile),'original_prediction_sha256':originalmeta['output_sha256'],
                 'source_mean':mu,'source_scale':sigma,'contexts':contextmeta,'eval_labels_opened':False})
        print('HEAVY_ACCURACY_INFERENCE_CASE',tid,flush=True)
        del predictors;gc.collect();torch.cuda.empty_cache()


def launch(pilot):
    prepare();workers=3 if pilot else 6;prefix='pilot' if pilot else 'full'
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i);log=OUT/f'{prefix}_worker_{i}.log'
        command=[sys.executable,__file__,'worker','--shard',str(i),'--shards',str(workers)]+(['--pilot'] if pilot else [])
        with log.open('a') as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'GPU':i,'returncode':p.returncode,'log':str(log)}
    atomic_json(OUT/f'{prefix}_execution.json',{'status':'RUNNING','active_endpoint':list(training.endpoint.ENDPOINT)})
    with ThreadPoolExecutor(max_workers=workers) as pool:results=list(pool.map(execute,range(workers)))
    ok=all(r['returncode']==0 for r in results)
    atomic_json(OUT/f'{prefix}_execution.json',{'status':'COMPLETE' if ok else 'FAILED','workers':results})
    training.verify_frozen();assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    assert ok


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','launch','worker']);p.add_argument('--pilot',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='launch':launch(a.pilot)
    else:worker(a.shard,a.shards,a.pilot)
