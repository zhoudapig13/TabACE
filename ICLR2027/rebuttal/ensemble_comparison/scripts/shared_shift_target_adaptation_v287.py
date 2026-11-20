"""Three shifts with explicit target-label context adaptation and held-out evaluation."""
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
import shift_target_adaptation_manifest_v285 as references
import target_reserved_shared_contexts_v286 as constructor
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor

source=references.source
OUT=RUN/'results/shared_shift_target_adaptation_v287'


def freeze():
    assert json.loads((references.OUT/'audit.json').read_text())['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'status':'SUPERVISED_TARGET_LABEL_AND_EXTRA_SOURCE_REFERENCE_ADAPTATION',
        'seed':0,'TFM_context_cap':256,'external_labeled_source_training_rows':'per-case count, up to1984',
        'labeled_target_adaptation':'fixed25percent of original query, cap64, removed from final evaluation',
        'every_context_contains_all_target_adaptation_rows':True,
        'source_context_selection':'same five pooled Shared roles and four query groups as v282',
        'classification_quotas':'observed source plus target adaptation label proportions, with class-coverage pseudocount',
        'class_prior_stability_assumption_not_imposed':True,'posterior_marginal_alignment_applied':False,
        'fusion':'original frozen dataset-OOF v77 Shared policy',
        'controls':['original Base on same remaining evaluation rows','Native1 same source pool and same target labels','target adaptation only'],
        'source_calibration64_labels_used':False,'TFM_weights_updated':False,
        'maximum_TFM_calls_per_condition':25,'compute_matched_to_original_Base':False,
        'adaptation_or_eval_split_selected_using_metrics':False,
        'final_evaluation_labels_opened':False,'expected_conditions':848,
        'previously_scored_development_dataset_collection':True,
        'original_Shared_unseen_unsupervised_generalization_claimed':False,
        'threshold_each_family':{'accuracy_gain_strictly_greater_than':.02,'r2_gain_strictly_greater_than':.02},
        'script_sha256':source.sha(__file__),'target_mix_sha256':source.sha(constructor.__file__),
        'cluster_geometry_sha256':source.sha(constructor.parent.__file__),
        'underlying_constructor_sha256':source.sha(constructor.parent.parent.__file__),
        'case_manifest_sha256':source.sha(references.OUT/'case_manifest.json'),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)


def worker(shard,shards):
    torch.set_num_threads(1);torch.manual_seed(0)
    plan=json.loads((OUT/'plan.json').read_text());assert source.sha(__file__)==plan['script_sha256']
    cases=json.loads((references.OUT/'case_manifest.json').read_text());predictors={}
    for counter,pos in enumerate(np.random.default_rng(287).permutation(len(cases))):
        if counter%shards!=shard:continue
        c=cases[int(pos)];task,tid=c['task'],c['task_id']
        if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
        predictor=predictors[task]
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0,y0,Q0,numeric,sid,qid=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        assert source.sha(c['adaptation_input_file'])==c['adaptation_input_sha256']
        with np.load(c['adaptation_input_file']) as z:
            TX,Ty,tids,C,cids,apos,ay,epos,aids,eids=(z[k] for k in
                ('X_train_reference','y_train_reference','train_reference_indices','X_calibration','calibration_indices',
                 'target_adaptation_positions','target_adaptation_y','evaluation_positions','target_adaptation_indices','evaluation_indices'))
        np.testing.assert_array_equal(qid[apos],aids);np.testing.assert_array_equal(qid[epos],eids)
        assert set(map(int,aids)).isdisjoint(map(int,eids))
        assert set(map(int,tids)).isdisjoint(map(int,np.r_[sid,qid,cids]))
        K=len(np.unique(y0)) if task=='classification' else None
        pfile=source.OUT/'policies'/task/f'{c["fold"]}.json'
        policy=json.loads(pfile.read_text());assert c['data_id'] not in policy['training_data_ids']
        for family,rho in [('clean',0)]+[(f,s) for f in c['families'] for s in source.novel.STRENGTHS]:
            dest=OUT/'predictions'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256'];continue
            if rho:X,y,Qfull,_=source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:X,y,Qfull=X0,y0,Q0
            original=source.OUT/'predictions/shift'/task/str(tid)/dest.name
            om=json.loads(original.with_suffix('.json').read_text());assert source.sha(original)==om['output_sha256']
            assert protocol.array_sha256(X)==om['observed_source_X_sha256']
            assert protocol.array_sha256(y)==om['observed_source_y_sha256']
            assert protocol.array_sha256(Qfull)==om['observed_target_X_sha256']
            with np.load(original) as z:base=z['Base'][epos]
            Q,AX=Qfull[epos],Qfull[apos]
            poolX,pooly=np.concatenate([X,TX]),np.concatenate([y,Ty])
            quota=np.r_[y,ay,np.unique(pooly)] if task=='classification' else y
            geometry=constructor.parent.geometry(poolX,pooly,Q,C,task)
            core=np.empty_like(base);members=np.empty((5,)+base.shape)
            rshape=(len(C),K) if task=='classification' else (len(C),)
            rcore=np.empty(rshape);rmembers=np.empty((5,)+rshape)
            calls=0;groups=[]
            def infer(A,b,T):
                nonlocal calls
                assert len(b)<=256
                for estimator in predictor.models.values():
                    root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                    if root is not None and hasattr(root,'root'):root.root=None
                torch.cuda.empty_cache();calls+=1
                return predictor(A,b,T,members=1,random_state=0,n_classes=K)[0]
            target_only=infer(AX,ay,np.concatenate([Q,C]))
            for group in np.unique(geometry['query_groups']):
                qi,ci,canonical,T,views=constructor.parent.contexts(poolX,pooly,Q,C,quota,task,geometry,group)
                A,b,cm=constructor.mix('canonical',poolX[canonical],pooly[canonical],T,AX,ay,task,quota)
                p=infer(A,b,T);core[qi]=p[:len(qi)];rcore[ci]=p[len(qi):]
                cm.update(X_sha256=protocol.array_sha256(A),y_sha256=protocol.array_sha256(b))
                records=[]
                for j,(name,V,labels,S) in enumerate(views):
                    A,b,record=constructor.mix(name,V,labels,S,AX,ay,task,quota,geometry.get('columns'))
                    p=infer(A,b,S);members[j,qi]=p[:len(qi)];rmembers[j,ci]=p[len(qi):]
                    record.update(name=name,X_sha256=protocol.array_sha256(A),y_sha256=protocol.array_sha256(b),
                        prediction_X_sha256=protocol.array_sha256(S));records.append(record)
                groups.append({'group':int(group),'evaluation_positions':qi.tolist(),'calibration_positions':ci.tolist(),
                    'canonical_pool_rows':canonical.tolist(),'canonical_context':cm,'contexts':records})
            mu,sigma=float(np.r_[pooly,ay].mean()),max(float(np.r_[pooly,ay].std()),1e-12)
            shared=source.fusion.predict({'task':task,'base':core,'bank':members,'mu':mu,'sigma':sigma},policy['Shared'])
            rshared=source.fusion.predict({'task':task,'base':rcore,'bank':rmembers,'mu':mu,'sigma':sigma},policy['Shared'])
            assert np.isfinite(shared).all()
            source.npwrite(dest,Base=base,Base_same_pool_and_target_labels=core,Shared_adapted=shared,
                Target_only=target_only[:len(Q)],members=members,reference_core=rcore,
                reference_members=rmembers,reference_Shared=rshared,source_mean=mu,source_scale=sigma)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('task','task_id','data_id','fold')}|
                {'family':family,'rho':rho,'original_input_sha256':c['input_sha256'],
                 'adaptation_input_sha256':c['adaptation_input_sha256'],
                 'original_prediction_sha256':om['output_sha256'],'policy_sha256':source.sha(pfile),
                 'external_source_labels':len(Ty),'original_test_labels_used_for_adaptation':len(ay),
                 'held_out_evaluation_rows':len(Q),'source_calibration_labels_used':False,
                 'final_eval_labels_opened':False,'actual_TFM_calls':calls,'groups':groups,
                 'pool_X_sha256':protocol.array_sha256(poolX),'pool_y_sha256':protocol.array_sha256(pooly),
                 'target_adaptation_X_sha256':protocol.array_sha256(AX),
                 'target_adaptation_y_sha256':protocol.array_sha256(ay),
                 'output_sha256':source.sha(dest)})
        print('SHIFT_SUPERVISED_CASE_COMPLETE',task,tid,'source_ref',len(Ty),'target_labels',len(ay),'eval',len(epos),flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i);log=OUT/f'worker_{i}.log'
        with log.open('a') as f:
            p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i),'--shards',str(shards)],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'returncode':p.returncode,'log':str(log)}
    results=[];atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':[],'shards':shards})
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for f in as_completed([pool.submit(execute,i) for i in range(shards)]):
            results.append(f.result());atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':results,'shards':shards})
    count=len(list((OUT/'predictions').glob('*/*/*.json')))
    success=all(r['returncode']==0 for r in results) and count==848
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if success else 'FAILED','workers':results,'conditions':count})
    assert success


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8)
    a=p.parse_args();launch(a.shards) if a.phase=='launch' else worker(a.shard,a.shards)
