"""Source-finetuned TFM bank and label-free latent features for task calibration.

Neither target adaptation nor evaluation labels are loaded by this process.
The fixed five Shared constructors and fusion gate are preserved. Six physical
GPUs at most; every TFM context remains at most 256 rows.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import os
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import torch
from tabicl import TabICLClassifier,TabICLRegressor
import task_adaptation_split_v311 as split
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,normalize

source=split.source
training=split.training
OUT=RUN/'results/task_adaptation_bank_v312'


class LatentPredictor(Predictor):
    def core(self,X,y,Q,n_classes):
        # Match the original Predictor's constant-context fallbacks.
        if (X.shape[1]==0 or np.all(np.ptp(np.asarray(X,dtype=float),axis=0)==0)
            or (self.task=='classification' and len(np.unique(y))==1)
            or (self.task=='regression' and np.std(y)<1e-12)):
            return self(X,y,Q,members=1,random_state=0,n_classes=n_classes)[0],np.zeros((len(Q),1),dtype=np.float32)
        model=self.models[1]
        root=getattr(getattr(model,'model_',None),'icl_predictor',None)
        if root is not None and hasattr(root,'root'):root.root=None
        model.fit(np.asarray(X,dtype=np.float32),y)
        if not getattr(model,'_weights_cached_for_rebuttal',False):
            model._load_model=lambda:None
            model._weights_cached_for_rebuttal=True
        module=model.model_.icl_predictor
        blocks=[];state={}
        def before_transformer(_module,args,kwargs):
            assert 'train_size' in kwargs
            state['train_size']=int(kwargs['train_size'])
        def before_decoder(_module,args):
            z=args[0];n=state['train_size']
            assert z.ndim==3 and z.shape[0]==1 and z.shape[1]-n==len(Q),(z.shape,n,len(Q))
            blocks.append(z[0,n:].detach().float().cpu().numpy())
        hooks=[module.tf_icl.register_forward_pre_hook(before_transformer,with_kwargs=True),
               module.decoder.register_forward_pre_hook(before_decoder)]
        try:
            if self.task=='classification':
                raw=model.predict_proba(np.asarray(Q,dtype=np.float32))
                pred=np.zeros((len(Q),n_classes),dtype=float)
                pred[:,np.asarray(model.classes_,dtype=int)]=raw;pred=normalize(pred)
            else:pred=np.asarray(model.predict(np.asarray(Q,dtype=np.float32)),dtype=float)
        finally:
            for hook in hooks:hook.remove()
        assert blocks and all(np.isfinite(b).all() for b in blocks)
        return pred,np.concatenate(blocks,axis=1)


def prepare():
    assert json.loads((split.OUT/'audit.json').read_text())['status']=='PASS'
    assert json.loads((training.OUT/'independent_training_audit.json').read_text())['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    policy=json.loads(training.implementation.GPU_POLICY.read_text())
    assert policy['allowed_GPUs']==list(range(6)) and policy['reserved_GPUs']==[6,7]
    plan={'seed':0,'cases':175,'conditions':2175,'TFM_context_cap':256,
        'target_adaptation_labels_in_TFM_contexts':False,'final_eval_labels_opened':False,
        'features':'core decoder-input query representations, all hierarchical nodes concatenated',
        'registry':'unchanged original v28 five contexts','gate':'unchanged dataset-OOF v77 Shared gate',
        'source_supervised_finetuning':True,'backbone_training':str(training.OUT),
        'supervised_extension_not_frozen_Shared':True,'previously_scored_development_cohort':True,
        'manifest_sha256':source.sha(split.OUT/'case_manifest.json'),
        'generator_sha256':source.sha(source.novel.__file__),'registry_sha256':source.sha(source.constructors.__file__),
        'fusion_sha256':source.sha(source.fusion.__file__),'script_sha256':source.sha(__file__),
        'GPU_policy':policy,'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def worker(shard,shards,case_index=None):
    torch.set_num_threads(1);torch.manual_seed(0)
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    policy=json.loads(training.implementation.GPU_POLICY.read_text())
    assert gpu in policy['allowed_GPUs'] and gpu not in policy['reserved_GPUs']
    plan=json.loads((OUT/'plan.json').read_text());assert source.sha(__file__)==plan['script_sha256']
    cases=json.loads((split.OUT/'case_manifest.json').read_text())
    positions=[case_index] if case_index is not None else np.random.default_rng(312).permutation(len(cases))
    for counter,pos in enumerate(positions):
        if counter%shards!=shard:continue
        c=cases[int(pos)];track,task,tid=c['track'],c['task'],c['task_id']
        checkpoint_dir=training.OUT/'checkpoints'/track/task/str(tid)
        checkpoint=json.loads((checkpoint_dir/'result.json').read_text())
        assert source.sha(checkpoint['checkpoint_file'])==checkpoint['checkpoint_sha256']
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0,y0,Q0,numeric=(z[k] for k in ('X_source','y_source','X_target','numeric_columns'))
        K=len(np.unique(y0)) if task=='classification' else None
        predictor=LatentPredictor(task,'cuda:0');cls=TabICLClassifier if task=='classification' else TabICLRegressor
        predictor.models[1]=cls(n_estimators=1,model_path=checkpoint['checkpoint_file'],allow_auto_download=False,
            device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
        pfile=source.OUT/'policies'/task/f'{c["fold"]}.json';policy_snapshot=json.loads(pfile.read_text())
        assert c['data_id'] not in policy_snapshot['training_data_ids']
        for family,rho in [('clean',0)]+[(f,r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest=OUT/'predictions'/track/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];continue
            if rho:X,y,Q,_=source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:X,y,Q=X0,y0,Q0
            assert len(y)<=256
            core,latent=predictor.core(X,y,Q,K)
            cache={tuple(protocol.array_sha256(v) for v in (X,y,Q)):core}
            def infer(A,b,T):
                assert len(b)<=256
                key=tuple(protocol.array_sha256(v) for v in (A,b,T))
                if key not in cache:
                    for estimator in predictor.models.values():
                        root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                        if root is not None and hasattr(root,'root'):root.root=None
                    torch.cuda.empty_cache()
                    cache[key]=predictor(A,b,T,members=1,random_state=0,n_classes=K)[0]
                return cache[key]
            views=source.constructors.shared_contexts(X,y,Q,len(y),task,core)
            assert len(views)==5
            predictions=[];records=[]
            for name,A,b,T in views:
                predictions.append(infer(A,b,T));records.append({'name':name,'rows':len(b),
                    'X_sha256':protocol.array_sha256(A),'y_sha256':protocol.array_sha256(b),'query_sha256':protocol.array_sha256(T)})
            bank=np.stack(predictions);mu,sigma=float(y.mean()),max(float(y.std()),1e-12)
            shared=source.fusion.predict({'task':task,'base':core,'bank':bank,'mu':mu,'sigma':sigma},policy_snapshot['Shared'])
            source.npwrite(dest,Native_finetuned=core,Shared_finetuned=shared,members=bank,
                latent=latent.astype(np.float32),source_mean=mu,source_scale=sigma)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('track','task','task_id','data_id','fold')}|
                {'family':family,'rho':rho,'output_sha256':source.sha(dest),'original_input_sha256':c['input_sha256'],
                 'checkpoint_sha256':checkpoint['checkpoint_sha256'],'policy_sha256':source.sha(pfile),
                 'observed_source_X_sha256':protocol.array_sha256(X),'observed_source_y_sha256':protocol.array_sha256(y),
                 'observed_query_X_sha256':protocol.array_sha256(Q),'actual_GPU':gpu,
                 'target_labels_used_in_context':0,'target_labels_opened':False,'contexts':records,
                 'latent_shape':list(latent.shape),'actual_TFM_calls':len(cache)})
        print('TASK_BANK_CASE_COMPLETE',track,task,tid,flush=True)
        del predictor;gc.collect();torch.cuda.empty_cache()


def launch():
    prepare();policy=json.loads(training.implementation.GPU_POLICY.read_text());allowed=policy['allowed_GPUs']
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(allowed[i]);log=OUT/f'worker_{i}.log'
        with log.open('a') as f:p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i),'--shards','6'],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'GPU':allowed[i],'returncode':p.returncode,'log':str(log)}
    done=[];atomic_json(OUT/'execution.json',{'status':'RUNNING','allowed_GPUs':allowed,'reserved_GPUs':policy['reserved_GPUs']})
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(execute,i) for i in range(6)]):
            done.append(future.result());atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':done})
    count=len(list((OUT/'predictions').glob('*/*/*/*.json')))
    success=all(r['returncode']==0 for r in done) and count==2175
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if success else 'FAILED','workers':done,'conditions':count,
        'allowed_GPUs':allowed,'reserved_GPUs':policy['reserved_GPUs']})
    assert success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','launch','worker'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);p.add_argument('--case-index',type=int)
    a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='launch':launch()
    else:worker(a.shard,a.shards,a.case_index)
