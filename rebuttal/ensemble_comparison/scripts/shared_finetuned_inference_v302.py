"""Original five Shared roles with source-supervised fine-tuned TabICL weights.

Extra source references enter through gradient training, not query-local pool
selection. Shift-only target labels are reserved in every context and excluded
from final evaluation. Original Base forecasts are immutable comparisons.
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
from tabicl import TabICLClassifier,TabICLRegressor
import shared_dataset_finetuning_v301 as training
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor

source=training.source
OUT=RUN/'results/shared_finetuned_inference_v302'


def reserve_target(name,A,b,T,AX,ay,task,columns=None):
    if len(ay)==0:return A,b,{'source_rows':len(b),'target_rows':0,'retained_candidate_positions':list(range(len(b)))}
    n=min(len(b),256-len(ay));rng=np.random.default_rng(0)
    if task=='classification':
        mandatory=np.asarray([rng.choice(np.flatnonzero(b==label)) for label in np.unique(b)],dtype=int)
        assert len(mandatory)<=n
        remaining=np.setdiff1d(np.arange(len(b)),mandatory)
        rows=np.r_[mandatory,rng.choice(remaining,n-len(mandatory),replace=False)];rng.shuffle(rows)
    else:rows=rng.choice(len(b),n,replace=False)
    adapted_X=AX[:,columns] if task=='classification' and name=='feature_subspace' else AX
    X,y=np.r_[A[rows],adapted_X],np.r_[b[rows],ay]
    assert X.shape[1]==T.shape[1] and len(y)<=256
    return X,y,{'source_rows':len(rows),'target_rows':len(ay),'retained_candidate_positions':rows.tolist()}


def prepare():
    assert json.loads((training.OUT/'execution.json').read_text())['status']=='COMPLETE'
    assert json.loads((training.OUT/'independent_training_audit.json').read_text())['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'status':'SOURCE_SUPERVISED_BACKBONE_FINE_TUNING_WITH_ORIGINAL_SHARED_REGISTRY',
        'seed':0,'TFM_context_cap':256,'registry':'unchanged original v28 five contexts',
        'gate':'unchanged v77 dataset-OOF Shared snapshot','query_cluster_or_enlarged_inference_pool_used':False,
        'clean_pre_corruption_source_supervision_used_for_weights':True,
        'external_clean_source_references_used_for_weights_only':True,
        'target_test_subset_context_adaptation':'shift only, fixed25percent cap64; excluded from final evaluation',
        'final_eval_labels_or_scores_used_for_fit_or_selection':False,
        'methods':['original Base','fine-tuned Native1','fine-tuned Native1 same target labels','fine-tuned Shared','fine-tuned Mean6 descriptive'],
        'fine_tuned_inference_norm_methods':['none'],'n_estimators_each_context':1,
        'all_conditions':2175,'training_cases':175,'expected_perturbation_conditions':1327,'expected_shift_conditions':848,
        'generator_sha256':source.sha(source.novel.__file__),'script_sha256':source.sha(__file__),
        'registry_sha256':source.sha(source.constructors.__file__),'fusion_sha256':source.sha(source.fusion.__file__),
        'training_manifest_sha256':source.sha(training.OUT/'case_manifest.json'),
        'training_audit_sha256':source.sha(training.OUT/'independent_training_audit.json'),
        'maximum_TFM_calls_per_condition':7,'compute_matched_to_Base':False,
        'supervised_adaptation_not_original_zero_shot_Shared':True,'previously_scored_development_cohort':True,
        'GPU_policy':json.loads(training.implementation.GPU_POLICY.read_text()),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def worker(shard,shards):
    torch.set_num_threads(1);torch.manual_seed(0)
    policy=json.loads(training.implementation.GPU_POLICY.read_text());gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert gpu in policy['allowed_GPUs'] and gpu not in policy['reserved_GPUs']
    plan=json.loads((OUT/'plan.json').read_text());assert source.sha(__file__)==plan['script_sha256']
    cases=json.loads((training.OUT/'case_manifest.json').read_text())
    shifts={(c['task'],c['task_id']):c for c in json.loads((RUN/'results/shift_target_adaptation_manifest_v285/case_manifest.json').read_text())}
    for counter,pos in enumerate(np.random.default_rng(302).permutation(len(cases))):
        if counter%shards!=shard:continue
        c=cases[int(pos)];track,task,tid=c['track'],c['task'],c['task_id']
        checkpoint_dir=training.OUT/'checkpoints'/track/task/str(tid)
        checkpoint=json.loads((checkpoint_dir/'result.json').read_text())
        assert source.sha(checkpoint['checkpoint_file'])==checkpoint['checkpoint_sha256']
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0,y0,Q0,numeric=(z[k] for k in ('X_source','y_source','X_target','numeric_columns'))
        if track=='shift':
            s=shifts[(task,tid)];assert source.sha(s['adaptation_input_file'])==s['adaptation_input_sha256']
            with np.load(s['adaptation_input_file']) as z:
                apos,epos,ay=(z[k] for k in ('target_adaptation_positions','evaluation_positions','target_adaptation_y'))
            assert set(apos).isdisjoint(epos)
        else:apos,epos,ay=np.array([],dtype=int),np.arange(len(Q0)),y0[:0]
        K=len(np.unique(y0)) if task=='classification' else None
        predictor=Predictor(task,'cuda:0');cls=TabICLClassifier if task=='classification' else TabICLRegressor
        predictor.models[1]=cls(n_estimators=1,model_path=checkpoint['checkpoint_file'],allow_auto_download=False,
            device='cuda:0',norm_methods=['none'],random_state=0,n_jobs=1,batch_size=1)
        pfile=source.OUT/'policies'/task/f'{c["fold"]}.json'
        snapshot=json.loads(pfile.read_text());assert c['data_id'] not in snapshot['training_data_ids']
        for family,rho in [('clean',0)]+[(f,r) for f in c['families'] for r in source.novel.STRENGTHS]:
            dest=OUT/'predictions'/track/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];continue
            if rho:X,y,Qfull,_=source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:X,y,Qfull=X0,y0,Q0
            Q,AX=Qfull[epos],Qfull[apos]
            old=source.OUT/'predictions'/track/task/str(tid)/dest.name
            om=json.loads(old.with_suffix('.json').read_text());assert source.sha(old)==om['output_sha256']
            assert protocol.array_sha256(X)==om['observed_source_X_sha256']
            assert protocol.array_sha256(y)==om['observed_source_y_sha256']
            assert protocol.array_sha256(Qfull)==om['observed_target_X_sha256']
            with np.load(old) as z:base=z['Base'][epos]
            cache={}
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
            native=infer(X,y,Q)
            coreX,corey,coremeta=reserve_target('canonical',X,y,Q,AX,ay,task)
            core=infer(coreX,corey,Q)
            coremeta.update(X_sha256=protocol.array_sha256(coreX),y_sha256=protocol.array_sha256(corey))
            views=source.constructors.shared_contexts(X,y,Q,len(y),task,core)
            assert len(views)==5
            predictions=[];records=[]
            columns=None
            if task=='classification':
                order=source.constructors.specific.importance(X,y.astype(float))
                columns=np.asarray([j for j in range(X.shape[1]) if j!=int(order[0])])
                if not len(columns):columns=np.asarray([0])
            for name,A,b,T in views:
                V,labels,record=reserve_target(name,A,b,T,AX,ay,task,columns)
                predictions.append(infer(V,labels,T))
                record.update(name=name,X_sha256=protocol.array_sha256(V),y_sha256=protocol.array_sha256(labels),
                    prediction_X_sha256=protocol.array_sha256(T));records.append(record)
            bank=np.stack(predictions);mu,sigma=float(corey.mean()),max(float(corey.std()),1e-12)
            shared=source.fusion.predict({'task':task,'base':core,'bank':bank,'mu':mu,'sigma':sigma},snapshot['Shared'])
            mean=np.concatenate([core[None],bank],axis=0).mean(0)
            assert np.isfinite(shared).all()
            source.npwrite(dest,Base=base,Base_finetuned=native,Base_finetuned_same_target_labels=core,
                Shared_finetuned=shared,Mean6_finetuned=mean,members=bank,source_mean=mu,source_scale=sigma)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('track','task','task_id','data_id','fold')}|
                {'family':family,'rho':rho,'output_sha256':source.sha(dest),'original_input_sha256':c['input_sha256'],
                 'original_prediction_sha256':om['output_sha256'],'checkpoint_sha256':checkpoint['checkpoint_sha256'],
                 'checkpoint_weights_changed':checkpoint['selected_weights_changed'],
                 'training_result_file':str(checkpoint_dir/'result.json'),'policy_sha256':source.sha(pfile),
                 'target_labels_used_in_context':len(ay),'target_adaptation_positions':apos.tolist(),
                 'evaluation_positions_in_original_query':epos.tolist(),'final_eval_labels_opened':False,
                 'source_supervised_finetuning':True,'core_context':coremeta,'contexts':records,
                 'actual_TFM_calls':len(cache)})
        print('FINETUNED_SHARED_CASE_COMPLETE',track,task,tid,'changed',checkpoint['selected_weights_changed'],flush=True)
        del predictor;gc.collect();torch.cuda.empty_cache()


def launch():
    prepare();policy=json.loads(training.implementation.GPU_POLICY.read_text());allowed=policy['allowed_GPUs']
    assert allowed==list(range(6))
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(allowed[i]);log=OUT/f'worker_{i}.log'
        with log.open('a') as f:p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i),'--shards',str(len(allowed))],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'GPU':allowed[i],'returncode':p.returncode,'log':str(log)}
    done=[];atomic_json(OUT/'execution.json',{'status':'RUNNING','allowed_GPUs':allowed,'reserved_GPUs':policy['reserved_GPUs']})
    with ThreadPoolExecutor(max_workers=len(allowed)) as pool:
        for future in as_completed([pool.submit(execute,i) for i in range(len(allowed))]):
            done.append(future.result());atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':done})
    count=len(list((OUT/'predictions').glob('*/*/*/*.json')))
    success=all(r['returncode']==0 for r in done) and count==2175
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if success else 'FAILED','workers':done,'conditions':count,
        'allowed_GPUs':allowed,'reserved_GPUs':policy['reserved_GPUs']})
    assert success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6)
    a=p.parse_args();launch() if a.phase=='launch' else worker(a.shard,a.shards)
