"""Explicit larger labeled-pool and query-local adaptation of five Shared roles."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import torch
import expanded_clean_reference_pool_v278 as references
import valid_cluster_shared_contexts_v282 as constructors
import posterior_and_ridge_calibration_v276 as posterior
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor

source=references.source
OUT=RUN/'results/shared_expanded_pool_v283'


def freeze():
    assert json.loads((references.OUT/'audit.json').read_text())['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'status':'EXTRA_LABELED_POOL_AND_QUERY_LOCAL_SHARED_ADAPTATION',
        'seed':0,'context_cap':256,'maximum_extra_clean_training_reference_rows':1984,
        'reserved_calibration_rows':64,'reserved_calibration_labels_used':False,
        'correction':'auxiliary calibration assignments restricted to nonempty query clusters; no scoring-based changes',
        'underlying_constructor_sha256':source.sha(constructors.parent.__file__),
        'predecessor_script_sha256':source.sha(RUN/'scripts/shared_expanded_pool_v280.py'),
        'query_clustering':'unlabeled four-cluster KMeans, fixed seed0, source-normalized geometry',
        'classification':'same five score roles; within-class source-prior quotas; drop-top-feature on canonical256',
        'regression':'same five reconstructed original roles with target-local canonical256',
        'source_pool':'observed source256 plus disjoint clean training reference rows',
        'classification_posterior':'source-class-prior marginal alignment, as v276',
        'fusion':'unchanged frozen dataset-OOF v77 Shared snapshot',
        'new_method_specific_training_on_query_labels':False,
        'method_receives_generator_or_strength':False,'backbone_weights_updated':False,
        'maximum_TFM_calls_per_condition':24,'compute_matched_to_Base':False,
        'same_training_pool_Native1_control':True,'Base_and_benchmark_unchanged':True,
        'extra_supervision_count_reported_per_dataset':True,
        'original_Shared_unseen_generalization_claimed':False,
        'previously_scored_development_cohort':True,'expected_conditions':1327,
        'script_sha256':source.sha(__file__),'constructors_sha256':source.sha(constructors.__file__),
        'posterior_sha256':source.sha(posterior.__file__),
        'reference_manifest_sha256':source.sha(references.OUT/'reference_manifest.json'),
        'numerical_threshold_each_family':{'accuracy_gain':.05,'r2_gain':.05},'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def worker(shard,shards):
    torch.set_num_threads(1);torch.manual_seed(0)
    plan=json.loads((OUT/'plan.json').read_text());assert source.sha(__file__)==plan['script_sha256']
    refs={(r['task'],r['task_id']):r for r in json.loads((references.OUT/'reference_manifest.json').read_text())}
    cases=[c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track']=='perturbation']
    predictors={}
    for counter,pos in enumerate(np.random.default_rng(280).permutation(len(cases))):
        if counter%shards!=shard:continue
        c=cases[int(pos)];task,tid=c['task'],c['task_id']
        if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
        predictor=predictors[task]
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0,y0,Q0,numeric,sid,qid=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        r=refs[(task,tid)];assert source.sha(r['reference_file'])==r['reference_sha256']
        with np.load(r['reference_file']) as z:
            TX,Ty,tids,C,cids=(z[k] for k in ('X_train_reference','y_train_reference','train_reference_indices','X_calibration','calibration_indices'))
        assert set(map(int,tids)).isdisjoint(map(int,np.r_[sid,qid,cids]))
        K=len(np.unique(y0)) if task=='classification' else None
        policy_path=source.OUT/'policies'/task/f'{c["fold"]}.json'
        saved=json.loads(policy_path.read_text());assert c['data_id'] not in saved['training_data_ids']
        for family,rho in [('clean',0)]+[(f,s) for f in c['families'] for s in source.novel.STRENGTHS]:
            dest=OUT/'predictions'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];continue
            if rho:X,y,Q,_=source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:X,y,Q=X0,y0,Q0
            original=source.OUT/'predictions/perturbation'/task/str(tid)/dest.name
            om=json.loads(original.with_suffix('.json').read_text());assert source.sha(original)==om['output_sha256']
            assert protocol.array_sha256(X)==om['observed_source_X_sha256']
            assert protocol.array_sha256(Q)==om['observed_target_X_sha256']
            with np.load(original) as z:base=z['Base']
            poolX,pooly=np.concatenate([X,TX]),np.concatenate([y,Ty])
            geometry=constructors.geometry(poolX,pooly,Q,C,task)
            core=np.empty_like(base);members=np.empty((5,)+base.shape)
            rshape=(len(C),K) if task=='classification' else (len(C),)
            rcore=np.empty(rshape);rmembers=np.empty((5,)+rshape)
            groups=[];calls=0
            def infer(A,b,T):
                nonlocal calls
                assert len(b)<=256
                for estimator in predictor.models.values():
                    root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                    if root is not None and hasattr(root,'root'):root.root=None
                torch.cuda.empty_cache();calls+=1
                return predictor(A,b,T,members=1,random_state=0,n_classes=K)[0]
            for group in np.unique(geometry['query_groups']):
                qi,ci,canonical,T,views=constructors.contexts(poolX,pooly,Q,C,y,task,geometry,group)
                out=infer(poolX[canonical],pooly[canonical],T)
                core[qi]=out[:len(qi)];rcore[ci]=out[len(qi):]
                context_records=[]
                for j,(name,A,b,V) in enumerate(views):
                    out=infer(A,b,V);members[j,qi]=out[:len(qi)];rmembers[j,ci]=out[len(qi):]
                    context_records.append({'name':name,'rows':len(b),'X_sha256':protocol.array_sha256(A),
                        'y_sha256':protocol.array_sha256(b),'prediction_X_sha256':protocol.array_sha256(V)})
                groups.append({'group':int(group),'query_positions':qi.tolist(),'calibration_positions':ci.tolist(),
                    'canonical_pool_rows':canonical.tolist(),'contexts':context_records})
            mu,sigma=float(pooly.mean()),max(float(pooly.std()),1e-12)
            raw=source.fusion.predict({'task':task,'base':core,'bank':members,'mu':mu,'sigma':sigma},saved['Shared'])
            rraw=source.fusion.predict({'task':task,'base':rcore,'bank':rmembers,'mu':mu,'sigma':sigma},saved['Shared'])
            if task=='classification':
                adapted,pm=posterior.align_probability_marginal(raw,y)
                control,cm=posterior.align_probability_marginal(core,y)
            else:adapted,control,pm,cm=raw,core,None,None
            assert np.isfinite(adapted).all() and np.isfinite(control).all()
            source.npwrite(dest,Base=base,Base_same_pool=control,Shared_adapted=adapted,
                Local_core=core,Shared_raw=raw,members=members,reference_core=rcore,
                reference_members=rmembers,reference_Shared=rraw,source_mean=mu,source_scale=sigma)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('task','task_id','data_id','fold')}|
                {'family':family,'rho':rho,'original_input_sha256':c['input_sha256'],
                 'reference_sha256':r['reference_sha256'],'additional_training_labels':len(Ty),
                 'reserved_calibration_labels_used':False,'query_labels_opened':False,
                 'original_prediction_sha256':om['output_sha256'],'policy_sha256':source.sha(policy_path),
                 'pool_X_sha256':protocol.array_sha256(poolX),'pool_y_sha256':protocol.array_sha256(pooly),
                 'groups':groups,'actual_TFM_calls':calls,'posterior_models':{'Shared':pm,'control':cm},
                 'output_sha256':source.sha(dest)})
        print('EXPANDED_POOL_CASE_COMPLETE',task,tid,len(Ty),flush=True)


def reuse_valid_predecessor():
    import shutil
    previous=RUN/'results/shared_expanded_pool_v280'
    failed=json.loads((previous/'coverage_failure_inventory.json').read_text())
    bad={(m['task'],m['task_id'],m['family'],m['rho']) for m in failed['affected_conditions']}
    count=0
    for src in sorted((previous/'predictions').glob('*/*/*.json')):
        meta=json.loads(src.read_text())
        if (meta['task'],meta['task_id'],meta['family'],meta['rho']) in bad:continue
        original=src.with_suffix('.npz')
        assert source.sha(original)==meta['output_sha256']
        dst=OUT/'predictions'/src.relative_to(previous/'predictions')
        if dst.exists():continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,dst.with_suffix('.npz'))
        meta['reused_frozen_predecessor_file']=str(original)
        atomic_json(dst,meta);count+=1
    atomic_json(OUT/'reuse_audit.json',{'copied_conditions':count,'excluded_buggy_conditions':len(bad),'outcome_based_selection':False})


def launch(shards):
    freeze()
    reuse_valid_predecessor()
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i);log=OUT/f'worker_{i}.log'
        with log.open('a') as f:
            p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i),'--shards',str(shards)],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'returncode':p.returncode,'log':str(log)}
    results=[];atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':[],'shards':shards})
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute,i) for i in range(shards)]):
            results.append(future.result());atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':results,'shards':shards})
    count=len(list((OUT/'predictions').glob('*/*/*.json')))
    success=all(r['returncode']==0 for r in results) and count==1327
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if success else 'FAILED','workers':results,'conditions':count})
    assert success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8)
    a=p.parse_args();launch(a.shards) if a.phase=='launch' else worker(a.shard,a.shards)
