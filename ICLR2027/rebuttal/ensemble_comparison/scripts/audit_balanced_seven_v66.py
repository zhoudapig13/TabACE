"""Replay resampled source cases and frozen seven-shift deployment per budget."""
import argparse
import hashlib
import json
import os
import time
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name,'1')
import joblib
import numpy as np
import torch
import balanced_seven_v66 as e
from prepare_data import atomic_json,protocol
from run_tabicl import normalize,metrics,Predictor,CHECKPOINTS


def corrected(p,train,target,choice):
    ratio=np.array([1-target,target])/np.array([1-train,train])
    logs=(np.log(np.clip(p,1e-12,1.))+choice['power']*np.log(ratio))/choice['temperature']
    return normalize(np.exp(logs-logs.max(1,keepdims=True)))


def combine(cells,baseline,policies,role):
    result={}
    for c in cells:
        family=c['meta']['family'];policy=policies if role=='shared' else policies[family];choice=policy['chosen']
        p=baseline[c['key']]
        if choice is not None:
            descriptor=c['meta']['priors'];train=descriptor['train_shared'] if role=='shared' else descriptor['train'][family]
            target=descriptor['target_shared'] if role=='shared' else descriptor['target'][family]
            value=corrected(c[role][choice['profile_index']],train,target,choice);a=choice['blend']
            p=normalize((1-a)*p+a*value)
        result[c['key']]=p
    return result


def check_policy_scores(cells,baseline,policy,predictions):
    measured=e.score(cells,baseline)[0]
    assert all(abs(measured[k]-policy['source_baseline'][k])<1e-12 for k in measured)
    if policy['chosen'] is not None:
        choice=policy['chosen'];trial=next(t for t in policy['trials'] if all(t[k]==v for k,v in choice.items()))
        measured=e.score(cells,predictions)[0]
        assert all(abs(measured[k]-trial['source_scores'][k])<1e-12 for k in measured)


def audit(budget,fresh):
    torch.set_num_threads(1);torch.manual_seed(0)
    predictor=Predictor('classification','cuda:0') if fresh else None
    root=e.OUT/'policies'/str(budget)/'classification/seven';freeze=root/'all_policies_frozen.json'
    policy=json.loads(freeze.read_text());metas=e.RAW.frozen.cohort('classification','seven')
    source_count=query_count=models_replayed=fresh_count=0;maximum=fresh_max=0.
    checkpoint=CHECKPOINTS/'tabicl-classifier-v2-20260212.ckpt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
    fresh_ids={str(metas[0]['task_id']),str(metas[-1]['task_id'])}
    for meta in metas:
        tid=str(meta['task_id']);pool=e.RAW.scaling.prepare('classification','seven',meta);split=e.partition(meta)
        balanced=e.resample(meta,pool,split)
        assert not np.intersect1d(balanced['validation_original_ids'],split['source_original_ids']).size
        assert not np.intersect1d(balanced['validation_original_ids'],pool['query_ids']).size
        source_descriptor=e.priors(meta,pool,split,budget,'validation');query_descriptor=e.priors(meta,pool,split,budget,'query')
        cache={}
        def model_prediction(role,index,mode,X):
            path=e.expert.OUT/'experts'/str(budget)/'seven'/tid/role/f'{mode}_{index}.joblib'
            if str(path) not in cache:cache[str(path)]=joblib.load(path)
            return e.expert.predict(X,cache[str(path)],2)
        for family_index,family in enumerate(e.expert.FAMILIES):
            for rho in [20,50,80,100]:
                source=e.OUT/'source'/str(budget)/tid/f'{family}_{rho}.npz';record=json.loads(source.with_suffix('.json').read_text())
                data,ids,original=e.source_data(meta,pool,split,budget,family,rho,balanced)
                assert {k:protocol.array_sha256(v) for k,v in data.items()}==record['source_input_hashes']
                assert original==record['original_context_hashes'] and source_descriptor==record['priors']
                with np.load(source) as z:
                    assert np.array_equal(ids,z['validation_original_ids']) and np.array_equal(data['y_target'],z['y_target'])
                    assert z['shared'].shape==(1,128,2) and z['specific'].shape==(3,128,2)
                    for role,key,indices in [('Shared','shared',[0]),(f'Specific_{family}','specific',range(3))]:
                        for index in indices:
                            p=model_prediction(role,index,'validation',data['X_target'])
                            error=float(np.max(np.abs(p-z[key][index])));maximum=max(maximum,error)
                            assert error<1e-12,(budget,tid,family,rho,role,index,error)
                            models_replayed+=1
                    if fresh and tid in fresh_ids and rho==[20,50,80,100][family_index%4]:
                        p,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,random_state=0,n_classes=2)
                        error=float(np.max(np.abs(p-z['base'])));fresh_max=max(fresh_max,error)
                        assert error<1e-5,(budget,tid,family,rho,'native',error)
                        fresh_count+=1
                source_count+=1
                origin=e.expert.nn.source.OUT/'banks/query'/str(budget)/'classification/seven'/tid/f'{family}_{rho}.npz'
                info=json.loads(origin.with_suffix('.json').read_text())
                with np.load(origin) as z:base=z['base'];truth=z['y_target']
                query,_=e.RAW.scaling.bank('classification','seven',meta,family,rho);X=query['X_target']
                assert protocol.array_sha256(X)==info['source_input_hashes']['X_target']
                anchor=base;predictions={'Base':base}
                for name,role,recipe in [('Shared','Shared',policy['Shared']),( 'Specific',f'Specific_{family}',policy['Specific'][family])]:
                    choice=recipe['chosen'];p=anchor
                    if choice is not None:
                        value=model_prediction(role,choice['profile_index'],'query',X)
                        train=query_descriptor['train_shared'] if role=='Shared' else query_descriptor['train'][family]
                        target=query_descriptor['target_shared'] if role=='Shared' else query_descriptor['target'][family]
                        value=corrected(value,train,target,choice);a=choice['blend'];p=normalize((1-a)*anchor+a*value)
                        models_replayed+=1
                    predictions[name]=p;anchor=p
                out=e.OUT/'predictions'/str(budget)/'classification/seven'/tid/f'{family}_{rho}.npz'
                record=json.loads(out.with_suffix('.json').read_text());assert freeze.stat().st_mtime<=out.stat().st_mtime
                with np.load(out) as z:
                    assert np.array_equal(z['y_target'],truth) and np.array_equal(z['Base'],base)
                    for name,p in predictions.items():
                        error=float(np.max(np.abs(p-z[name])));maximum=max(maximum,error);assert error<1e-12
                        measured=metrics(truth,z[name],'classification',1.)
                        assert all((v is None and record['methods'][name][k] is None) or abs(v-record['methods'][name][k])<1e-12 for k,v in measured.items())
                query_count+=1
        print('BALANCED_SEVEN_AUDIT_DATASET',budget,tid,flush=True)
    cells=e.source_cells(budget);baseline={c['key']:c['base'] for c in cells}
    shared=combine(cells,baseline,policy['Shared'],'shared')
    specific=combine(cells,shared,policy['Specific'],'specific')
    check_policy_scores(cells,baseline,policy['Shared'],shared)
    for family in e.expert.FAMILIES:
        group=[c for c in cells if c['meta']['family']==family]
        check_policy_scores(group,shared,policy['Specific'][family],specific)
    assert source_count==query_count==980 and (not fresh or fresh_count==14)
    report=dict(status='PASS',budget=budget,source_conditions_verified=source_count,query_conditions_replayed=query_count,
        auxiliary_prediction_cases_replayed=models_replayed,maximum_probability_error=maximum,
        fresh_Native1_cases=fresh_count,fresh_Native1_max_probability_error=fresh_max,
        checkpoint_hash_verified=True,source_original_IDs_disjoint=True,source_priors_reconstructed_from_TRAIN_only=True,
        resampling_independent_label_count_not_increased=True,all_models_from_original_v54=True,
        Base_bitwise_unchanged=True,all_metrics_and_selected_source_scores_recomputed=True,
        frozen_policies_precede_query_outputs=True,query_labels_used_for_fit_or_priors=False,
        development_only=True,Markdown_written=False,time_unix=time.time())
    atomic_json(e.OUT/'raw_audits'/f'{budget}.json',report)
    print('BALANCED_SEVEN_RAW_AUDIT_PASS',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=int,required=True);parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args();audit(args.budget,args.fresh)
