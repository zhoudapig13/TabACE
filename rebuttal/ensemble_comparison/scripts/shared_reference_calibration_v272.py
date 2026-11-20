"""Calibrate existing Shared and clean-reference experts without evaluation labels.

The 64 reference labels train a local convex fusion. Any expert that also uses
those labels for context is cross-fitted over four reference folds first.
Original Base/Shared forecasts and v269 mixed-context control are preserved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import torch
from sklearn.model_selection import KFold

import reference_convex_fusion_v271 as fusion
import shared_clean_reference_contexts_v269 as parent
from prepare_data import RUN, ROOT, atomic_json, protocol
from run_tabicl import Predictor

source, reference = parent.source, parent.reference
OUT = RUN/'results/shared_reference_calibration_v272'


def source_priority(y, task):
    first = parent.source_rows(y, task)
    rest = np.setdiff1d(np.arange(len(y)), first)
    return np.r_[first, rest]


def freeze():
    assert json.loads((parent.OUT/'independent_audit.json').read_text())['status']=='PASS'
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'EXPLICIT_REFERENCE_LABEL_FUSION_ADAPTATION_NOT_ORIGINAL_SHARED',
        'seed': 0, 'reference_labels_per_dataset': 64, 'context_cap': 256,
        'experts': ['original_Base', 'original_Shared', 'source192_reference64', 'reference64_only'],
        'calibration': 'convex CE or normalized MSE; fixedL2=.02 toward(.5,.5,0,0)',
        'reference_context_experts': 'four-fold reference OOF forecasts for fitting fusion; all64 only for deployment',
        'final_evaluation_labels_used_for_fit_or_selection': False,
        'original_Base_Shared_and_benchmark_unchanged': True,
        'final_reference_mix_reuses_v269_forecast': True,
        'backbone_weights_updated': False, 'additional_supervision_explicit': True,
        'same_reference_Base_control': 'source192_reference64 Native1, unchanged v269',
        'TFM_compute_matched_to_Base': False,
        'additional_TFM_calls_per_condition_maximum': 10,
        'additional_clean_reference_TFM_calls_per_dataset': 5,
        'reference_features_held_out_from_original_constructor_geometry': True,
        'reference_classification_feature_subspace_uses_same_source_selected_columns': True,
        'new_family_or_strength_identity_received_by_fusion': False,
        'development_cohort_previously_evaluated': True,
        'independent_unseen_generalization_claimed': False,
        'expected_conditions': 1327,
        'numerical_acceptance': {'each_family_Acc_gain': .05, 'each_family_R2_gain': .05},
        'script_sha256': source.sha(__file__), 'fusion_sha256': source.sha(fusion.__file__),
        'constructor_sha256': source.sha(source.constructors.__file__),
        'original_fusion_sha256': source.sha(source.fusion.__file__),
        'reference_manifest_sha256': source.sha(reference.OUT/'reference_manifest.json'),
        'Markdown_written': False}
    p=OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text())==plan
    else:
        atomic_json(p, plan)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    plan = json.loads((OUT/'plan.json').read_text())
    assert source.sha(__file__)==plan['script_sha256']
    refs={(r['task'], r['task_id']):r for r in json.loads((reference.OUT/'reference_manifest.json').read_text())}
    cases=[c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track']=='perturbation']
    predictors={}
    for count, pos in enumerate(np.random.default_rng(272).permutation(len(cases))):
        if count%shards!=shard:
            continue
        c=cases[int(pos)]
        task, tid=c['task'], c['task_id']
        if task not in predictors:
            predictors[task]=Predictor(task, 'cuda:0')
        predictor=predictors[task]
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X0,y0,Q0,numeric,sid,qid=(z[k] for k in ('X_source','y_source','X_target','numeric_columns','source_indices','target_indices'))
        r=refs[(task,tid)]
        assert source.sha(r['reference_file'])==r['reference_sha256']
        with np.load(r['reference_file']) as z:
            RX,Ry,rid=(z[k] for k in ('X_reference','y_reference','reference_indices'))
        assert set(map(int,rid)).isdisjoint(map(int,np.r_[sid,qid]))
        K=len(np.unique(y0)) if task=='classification' else None
        folds=list(KFold(4,shuffle=True,random_state=0).split(RX))
        policyfile=source.OUT/'policies'/task/f'{c["fold"]}.json'
        policy=json.loads(policyfile.read_text())
        assert c['data_id'] not in policy['training_data_ids']
        priority=source_priority(y0,task)
        calls=0
        def infer(A,b,T):
            nonlocal calls
            assert len(b)<=256
            for estimator in predictor.models.values():
                root=getattr(getattr(estimator,'model_',None),'icl_predictor',None)
                if root is not None and hasattr(root,'root'):
                    root.root=None
            torch.cuda.empty_cache()
            calls+=1
            return predictor(A,b,T,members=1,random_state=0,n_classes=K)[0]
        cfpath=OUT/'reference_forecasts'/task/f'{tid}.npz'
        if cfpath.with_suffix('.json').exists():
            cm=json.loads(cfpath.with_suffix('.json').read_text())
            assert source.sha(cfpath)==cm['output_sha256']
            with np.load(cfpath) as z:
                clean_ref_oof,clean_ref_query=z['reference_OOF'],z['query']
        else:
            shape=(64,K) if task=='classification' else (64,)
            clean_ref_oof=np.empty(shape)
            for train,valid in folds:
                clean_ref_oof[valid]=infer(RX[train],Ry[train],RX[valid])
            clean_ref_query=infer(RX,Ry,Q0)
            source.npwrite(cfpath,reference_OOF=clean_ref_oof,query=clean_ref_query)
            atomic_json(cfpath.with_suffix('.json'),{'output_sha256':source.sha(cfpath),
                'reference_sha256':r['reference_sha256'], 'original_input_sha256':c['input_sha256'],
                'folds':[{'train':tr.tolist(),'validation':va.tolist()} for tr,va in folds],
                'query_labels_opened':False,'actual_TFM_calls':5})
        for family,rho in [('clean',0)]+[(f,s) for f in c['families'] for s in source.novel.STRENGTHS]:
            dest=OUT/'predictions'/task/str(tid)/f'{family}_{rho}.npz'
            if dest.with_suffix('.json').exists():
                assert source.sha(dest)==json.loads(dest.with_suffix('.json').read_text())['output_sha256']
                continue
            if rho:
                X,y,Q,_=source.novel.generate(X0,y0,Q0,numeric,task,family,rho,tid)
            else:
                X,y,Q=X0,y0,Q0
            old=source.OUT/'predictions/perturbation'/task/str(tid)/dest.name
            om=json.loads(old.with_suffix('.json').read_text())
            assert source.sha(old)==om['output_sha256']
            assert protocol.array_sha256(X)==om['observed_source_X_sha256']
            mixed=parent.OUT/'predictions'/task/str(tid)/dest.name
            mm=json.loads(mixed.with_suffix('.json').read_text())
            assert source.sha(mixed)==mm['output_sha256']
            with np.load(old) as z:
                base,shared=z['Base'],z['Shared']
            with np.load(mixed) as z:
                control=z['Base_plus_same_reference']
            before=calls
            base_ref=infer(X,y,RX)
            contexts=source.constructors.shared_contexts(X,y,Q,256,task,base)
            banks=[]
            for name,A,b,T in contexts:
                R=RX
                if task=='classification' and name=='feature_subspace':
                    order=source.constructors.specific.importance(X,y.astype(float))
                    columns=np.asarray([j for j in range(X.shape[1]) if j!=int(order[0])])
                    if not len(columns):
                        columns=np.asarray([0])
                    np.testing.assert_array_equal(Q[:,columns],T)
                    R=RX[:,columns]
                else:
                    np.testing.assert_array_equal(T,Q)
                banks.append(infer(A,b,R))
            bank_ref=np.stack(banks)
            mu,sigma=float(y.mean()),max(float(y.std()),1e-12)
            shared_ref=source.fusion.predict({'task':task,'base':base_ref,'bank':bank_ref,'mu':mu,'sigma':sigma},policy['Shared'])
            mixed_ref=np.empty_like(base_ref)
            crossfit=[]
            for train,valid in folds:
                source_part=priority[:256-len(train)]
                A=np.concatenate([X[source_part],RX[train]])
                b=np.concatenate([y[source_part],Ry[train]])
                mixed_ref[valid]=infer(A,b,RX[valid])
                crossfit.append({'reference_train':train.tolist(),'reference_validation':valid.tolist(),
                    'source_positions':source_part.tolist(),'context_X_sha256':protocol.array_sha256(A),
                    'context_y_sha256':protocol.array_sha256(b)})
            P=np.stack([base_ref,shared_ref,mixed_ref,clean_ref_oof])
            Pquery=np.stack([base,shared,control,clean_ref_query])
            local=fusion.fit(P,Ry,task,max(float(np.std(Ry)),1e-12))
            pred=fusion.predict(Pquery,local)
            assert np.isfinite(pred).all()
            source.npwrite(dest,Base=base,OriginalShared=shared,Base_plus_same_reference=control,
                Shared_reference_calibrated=pred,reference_expert_predictions=P,
                query_expert_predictions=Pquery,reference_members=bank_ref)
            atomic_json(dest.with_suffix('.json'),{k:c[k] for k in ('task','task_id','data_id','fold')} |
                {'family':family,'rho':rho,'original_input_sha256':c['input_sha256'],
                 'reference_sha256':r['reference_sha256'],'original_prediction_sha256':om['output_sha256'],
                 'mixed_parent_sha256':mm['output_sha256'],'clean_reference_forecast_sha256':source.sha(cfpath),
                 'local_model':local,'reference_folds':crossfit,'extra_reference_labels':64,
                 'final_query_labels_opened':False,'local_fitting_labels_sha256':protocol.array_sha256(Ry),
                 'new_TFM_calls':calls-before,'output_sha256':source.sha(dest)})
        print('REFERENCE_CALIBRATION_CASE_COMPLETE',task,tid,flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=str(i)
        log=OUT/f'worker_{i}.log'
        with log.open('a') as f:
            p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i),'--shards',str(shards)],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'returncode':p.returncode,'log':str(log)}
    results=[]
    atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':[],'shards':shards})
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute,i) for i in range(shards)]):
            results.append(future.result())
            atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':results,'shards':shards})
    count=len(list((OUT/'predictions').glob('*/*/*.json')))
    success=all(r['returncode']==0 for r in results) and count==1327
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if success else 'FAILED','workers':results,'conditions':count})
    assert success


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['launch','worker'])
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=8)
    args=p.parse_args()
    launch(args.shards) if args.phase=='launch' else worker(args.shard,args.shards)
