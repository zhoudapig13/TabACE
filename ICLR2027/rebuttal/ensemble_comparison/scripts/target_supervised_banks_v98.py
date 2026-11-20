"""Target-labeled adaptation banks; evaluation truth is never opened here.

Shared and a native1 comparator access the same original256+target64 label
pool; each inference context stays <=256. Target calibration predictions are
nested out-of-fold, with disjoint adaptation/selection halves. A/C may use
Shortcut calibration; B/D may use only Clean and auxiliary label-independent
Domain/Covariate calibration. All constructors retain every column in C/D.
"""
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

import target_calibration_split_v97 as split
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,preserving_bootstrap
from shared_reference import contexts,class_subset

OUT=RUN/'results/target_supervised_banks_v98'
CONDITIONS=[('clean',0)]+[('shortcut',r) for r in (20,50,80,100)]+[
    ('domain',r) for r in (50,100)]+[('covariate',r) for r in (50,100)]


def condition(data,family,rho):
    y=data['y_source'].copy()
    X=data[f'X_source_{rho}' if family=='shortcut' else 'X_source_0'].copy()
    if family=='domain':
        s=rho/100.;center=X.mean(0);sd=np.maximum(X.std(0),1e-6)
        X=center+(X-center)*(1+.75*s)+.5*s*sd
    if family=='covariate':
        score=(X[:,0]-X[:,0].mean())/max(float(X[:,0].std()),1e-6)
        w=np.exp(np.clip(3.*rho/100.*score,-15,15));w/=w.sum()
        rng=np.random.default_rng(0)
        mandatory=[int(rng.choice(np.flatnonzero(y==label))) for label in np.unique(y)]
        ids=np.r_[mandatory,rng.choice(len(y),len(y)-len(mandatory),replace=True,p=w)].astype(int)
        rng.shuffle(ids);X,y=X[ids],y[ids]
    return X.astype(np.float32),y


def native_context(X,y,T,t,budget=256):
    count=budget-len(t)
    # Source identities stay fixed across severities; no query labels enter.
    score=np.random.default_rng(0).random(len(y))
    ids=class_subset(y,score,count/len(y),count)
    assert len(ids)==count
    return np.vstack([X[ids],T]),np.r_[y[ids],t]


def target_context(X,y,T,t):
    missing=[label for label in np.unique(y) if not np.any(t==label)]
    rng=np.random.default_rng(0)
    ids=np.array([rng.choice(np.flatnonzero(y==label)) for label in missing],dtype=int)
    return (np.vstack([T,X[ids]]),np.r_[t,y[ids]]) if len(ids) else (T,t)


def stage_specs(data,family):
    cal=np.arange(len(data['y_calibration']))
    for prefix in ('adapt','all'):
        for fold in range(2):
            yield f'{prefix}_oof_{fold}',data[f'{prefix}_fit_{fold}'],data[f'{prefix}_query_{fold}']
    yield 'validation',data['adapt_positions'],data['validation_positions']
    if family in ('clean','shortcut'):
        yield 'evaluation',cal,np.arange(len(data['X_evaluation']))+len(cal)


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text())['code_sha256']==split.parent.original.sha(__file__)
        return
    complete=json.loads((split.OUT/'completion.json').read_text());assert complete['datasets']==72
    atomic_json(path,{'status':'FROZEN_BEFORE_MODEL_INFERENCE','seed':0,'datasets':72,
        'original_source_labels':256,'extra_target_labels_per_dataset':64,'context_cap':256,
        'alternative_contexts':5,'estimators_per_branch':1,
        'base_controls':['zero-target native1 on original256','target-aware native1 on source192+target64'],
        'label_pool_for_all_target_aware_methods':'original256+same target64',
        'branch1':'target-labeled context, original source anchors only for missing classes',
        'branch3_A_B':'observed-source leading-correlation feature deletion',
        'branch3_C_D':'class-preserving full-feature bootstrap of the canonical target-aware context',
        'other_branches':'registered target-neighbor, locally stable, low conflict from common accessible pool',
        'calibration_protocol':'32 adaptation/32 selection; internal2-fold predictions; full64 OOF for final refit',
        'evaluation_truth_opened_by_preparation':False,'evaluation_features_unlabeled_transductive':True,
        'constructor_query_support':'same union of unlabeled calibration+evaluation features for every stage',
        'B_D_calibration_allowed':['clean','domain','covariate'],
        'B_D_calibration_excludes':['shortcut','other equivalent label-conditional feature corruption'],
        'auxiliary_conditions':'source-only label-independent affine and covariate-resampling calibration; not extra evaluation conditions',
        'conditions':CONDITIONS,'semisynthetic_generator_parameters_changed':False,
        'partial_original_test_labels_reassigned_to_training':True,'original_paper_replication':False,
        'scope':'few-shot target-supervised adaptation; not zero-target-label robustness',
        'code_sha256':split.parent.original.sha(__file__),'Markdown_written':False})


def worker(shard,shards):
    torch.set_num_threads(1);torch.manual_seed(0)
    paths=sorted((split.OUT/'inputs').glob('*.npz'))
    predictor=Predictor('classification','cuda:0')
    def infer(A,b,Q,k):
        for model in predictor.models.values():
            root=getattr(getattr(model,'model_',None),'icl_predictor',None)
            if root is not None and hasattr(root,'root'):root.root=None
        torch.cuda.empty_cache()
        return predictor(A,b,Q,n_classes=k)[0]
    for i,path in enumerate(paths):
        if i%shards!=shard:continue
        with np.load(path) as z:data={k:z[k] for k in z.files}
        meta=json.loads(path.with_suffix('.json').read_text());task_id=meta['task_id']
        assert split.parent.original.sha(path)==meta['model_inputs_sha256']
        union=np.vstack([data['X_calibration'],data['X_evaluation']]);K=len(np.unique(data['y_source']))
        for family,rho in CONDITIONS:
            X,y=condition(data,family,rho)
            for stage,fit_ids,query_ids in stage_specs(data,family):
                out=OUT/'banks'/str(task_id)/f'{family}_{rho}'/f'{stage}.npz'
                if out.with_suffix('.json').exists():continue
                T,t=data['X_calibration'][fit_ids],data['y_calibration'][fit_ids]
                assert set(fit_ids).isdisjoint(query_ids) or stage=='evaluation'
                A,b=native_context(X,y,T,t)
                base=infer(A,b,union,K)
                pool_X,pool_y=np.vstack([X,T]),np.r_[y,t]
                views=contexts(pool_X,pool_y,union,A,b,256,'classification')
                C,d=target_context(X,y,T,t)
                views[1]=('target_labeled',C,d,union)
                bank=np.stack([infer(U,v,Q,K) for _,U,v,Q in views])
                ids=preserving_bootstrap(b,np.random.default_rng(0))
                bootstrap=infer(A[ids],b[ids],union,K)
                values={'base_target':base[query_ids],'members':bank[:,query_ids],
                    'bootstrap':bootstrap[query_ids],'fit_cal_positions':fit_ids,'query_positions':query_ids,
                    'calibration_rows_in_context':len(fit_ids)}
                if stage=='evaluation':values['base_zero_target']=infer(X,y,union,K)[query_ids]
                out.parent.mkdir(parents=True,exist_ok=True)
                temp=out.with_suffix('.tmp.npz');np.savez_compressed(temp,**values);temp.replace(out)
                atomic_json(out.with_suffix('.json'),{'status':'COMPLETE','task_id':task_id,'data_id':meta['data_id'],
                    'family':family,'rho':rho,'stage':stage,'query_rows':len(query_ids),
                    'fit_target_calibration_rows':len(fit_ids),'source_rows':len(y),'base_context_rows':len(b),
                    'branch_context_rows':[len(v[2]) for v in views],
                    'target_only_source_anchors':len(d)-len(t),'context_cap':256,
                    'uses_evaluation_labels':False,'unlabeled_evaluation_features_accessible':True,
                    'constructor_receives_family_or_severity':False,
                    'source_input_sha256':protocol.array_sha256(X),'source_label_sha256':protocol.array_sha256(y),
                    'fit_cal_positions_sha256':protocol.array_sha256(fit_ids),
                    'query_positions_sha256':protocol.array_sha256(query_ids),
                    'input_file_sha256':meta['model_inputs_sha256'],'matched_column_identity_oracle':False})
        print('TARGET_BANK_TASK',task_id,flush=True)


def launch(shards):
    freeze()
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i);log=OUT/f'worker_{i}.log'
        with log.open('a') as handle:
            p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i),'--shards',str(shards)],env=env,stdout=handle,stderr=subprocess.STDOUT)
        return {'shard':i,'returncode':p.returncode,'log':str(log)}
    results=[]
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for f in as_completed([pool.submit(execute,i) for i in range(shards)]):
            r=f.result();results.append(r);print(r,flush=True)
            atomic_json(OUT/'execution.json',{'status':'RUNNING','workers':results})
    atomic_json(OUT/'execution.json',{'status':'COMPLETE' if all(r['returncode']==0 for r in results) else 'FAILED','workers':results})
    if any(r['returncode'] for r in results):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8)
    a=p.parse_args()
    if a.phase=='worker':worker(a.shard,a.shards)
    else:launch(a.shards)
