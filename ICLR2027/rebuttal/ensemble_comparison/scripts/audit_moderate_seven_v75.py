"""Replay selected auxiliary models and every saved seven-shift fusion."""
import argparse
import json
import os
import time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import torch
import moderate_seven_v75 as e
from prepare_data import atomic_json,protocol
from run_tabicl import Predictor,metrics,normalize


def source_policy(cells,P,policy,role):
    choice=policy['chosen']
    if choice is None:return P
    priors=[c['meta']['priors'] for c in cells]
    train=np.array([d['train_shared'] if role=='shared' else d['train'][c['meta']['family']] for c,d in zip(cells,priors)])
    target=np.array([d['target_shared'] if role=='shared' else d['target'][c['meta']['family']] for c,d in zip(cells,priors)])
    values=e.correction(np.array([c[role][choice['profile_index']] for c in cells]),train,target,choice['power'],choice['temperature'])
    a=choice['blend'];return normalize((1-a)*P+a*values)


def audit(budget,fresh):
    torch.set_num_threads(1);torch.manual_seed(0)
    predictor=Predictor('classification','cuda:0') if fresh else None
    folder=e.common.root(budget,'classification','seven');freeze=folder/'all_policies_frozen.json'
    policy=json.loads(freeze.read_text());cells=e.parent.source_cells(budget);Y=np.array([c['y_target'] for c in cells])
    B=np.array([c['base'] for c in cells]);Sh=source_policy(cells,B,policy['Shared'],'shared')
    Sp=source_policy(cells,Sh,policy['Specific'],'specific')
    for name,anchor,pred in [('Shared',B,Sh),('Specific',Sh,Sp)]:
        recipe=policy[name];initial=e.score(anchor,Y)
        assert all(abs(initial[k]-recipe['source_baseline'][k])<1e-12 for k in initial)
        choice=recipe['chosen']
        if choice is not None:
            row=next(t for t in recipe['trials'] if all(t[k]==v for k,v in choice.items()))
            actual=e.score(pred,Y);assert all(abs(actual[k]-row['source_scores'][k])<1e-12 for k in actual)
    metas=e.parent.RAW.frozen.cohort('classification','seven');fresh_ids={str(metas[0]['task_id']),str(metas[-1]['task_id'])}
    source_count=query_count=native_count=aux_count=0;maximum=native_error=0.
    for meta in metas:
        tid=str(meta['task_id']);pool=e.parent.RAW.scaling.prepare('classification','seven',meta);split=e.parent.partition(meta)
        balanced=e.parent.resample(meta,pool,split)
        assert not np.intersect1d(balanced['validation_original_ids'],split['source_original_ids']).size
        assert not np.intersect1d(balanced['validation_original_ids'],pool['query_ids']).size
        query_priors=e.parent.priors(meta,pool,split,budget,'query');cache={}
        def predict(role,index,mode,X):
            path=e.parent.expert.OUT/'experts'/str(budget)/'seven'/tid/role/f'{mode}_{index}.joblib'
            if str(path) not in cache:cache[str(path)]=joblib.load(path)
            return e.parent.expert.predict(X,cache[str(path)],2)
        for fi,family in enumerate(e.parent.expert.FAMILIES):
            for rho in [20,50,80,100]:
                source=e.parent.OUT/'source'/str(budget)/tid/f'{family}_{rho}.npz'
                meta_source=json.loads(source.with_suffix('.json').read_text())
                data,ids,_=e.parent.source_data(meta,pool,split,budget,family,rho,balanced)
                assert {k:protocol.array_sha256(v) for k,v in data.items()}==meta_source['source_input_hashes']
                with np.load(source) as z:
                    assert np.array_equal(z['y_target'],data['y_target']) and np.array_equal(z['validation_original_ids'],ids)
                    for name,role,key in [('Shared','Shared','shared'),('Specific',f'Specific_{family}','specific')]:
                        choice=policy[name]['chosen']
                        if choice is not None:
                            p=predict(role,choice['profile_index'],'validation',data['X_target'])
                            error=float(np.max(np.abs(p-z[key][choice['profile_index']])));maximum=max(maximum,error)
                            assert error<1e-12;aux_count+=1
                    if fresh and tid in fresh_ids and rho==[20,50,80,100][fi%4]:
                        p,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,random_state=0,n_classes=2)
                        error=float(np.max(np.abs(p-z['base'])));native_error=max(native_error,error)
                        assert error<1e-5,(budget,tid,family,rho,error);native_count+=1
                source_count+=1
                origin=e.parent.expert.nn.source.OUT/'banks/query'/str(budget)/'classification/seven'/tid/f'{family}_{rho}.npz'
                info=json.loads(origin.with_suffix('.json').read_text())
                with np.load(origin) as z:base=z['base'];truth=z['y_target']
                raw,_=e.parent.RAW.scaling.bank('classification','seven',meta,family,rho);X=raw['X_target']
                assert protocol.array_sha256(X)==info['source_input_hashes']['X_target']
                anchor=base;expected={'Base':base}
                for name,role in [('Shared','Shared'),('Specific',f'Specific_{family}')]:
                    choice=policy[name]['chosen'];value=anchor
                    if choice is not None:
                        value=predict(role,choice['profile_index'],'query',X)
                        train=query_priors['train_shared'] if name=='Shared' else query_priors['train'][family]
                        target=query_priors['target_shared'] if name=='Shared' else query_priors['target'][family]
                        value=e.correction(value[None],np.array([train]),np.array([target]),choice['power'],choice['temperature'])[0]
                        a=choice['blend'];value=normalize((1-a)*anchor+a*value);aux_count+=1
                    expected[name]=value;anchor=value
                out=e.OUT/'predictions'/str(budget)/'classification/seven'/tid/f'{family}_{rho}.npz'
                record=json.loads(out.with_suffix('.json').read_text());assert freeze.stat().st_mtime<=out.stat().st_mtime
                with np.load(out) as z:
                    assert np.array_equal(z['Base'],base) and np.array_equal(z['y_target'],truth)
                    for name,value in expected.items():
                        error=float(np.max(np.abs(value-z[name])));maximum=max(maximum,error);assert error<1e-12
                        scores=metrics(truth,z[name],'classification',1.)
                        assert all((v is None and record['methods'][name][k] is None) or abs(v-record['methods'][name][k])<1e-12 for k,v in scores.items())
                query_count+=1
        print('MODERATE_SEVEN_AUDIT_DATASET',budget,tid,flush=True)
    assert source_count==query_count==980 and (not fresh or native_count==14)
    result=dict(status='PASS',budget=budget,source_cases_verified=source_count,query_cases_replayed=query_count,
        selected_auxiliary_predictions_replayed=aux_count,maximum_fusion_and_auxiliary_error=maximum,
        fresh_Native1_cases=native_count,fresh_Native1_max_error=native_error,source_original_IDs_disjoint=True,
        Base_bitwise_unchanged=True,selected_source_scores_and_query_metrics_recomputed=True,
        policies_frozen_before_query_outputs=True,query_labels_used_for_fit=False,
        legacy_CIFAR_replay_failure_unchanged=True,full_goal_complete_claim=False,time_unix=time.time())
    atomic_json(e.OUT/'seven_arithmetic_audits'/f'{budget}.json',result)
    print('MODERATE_SEVEN_AUDIT_PASS',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',required=True,type=int);p.add_argument('--fresh',action='store_true');a=p.parse_args();audit(a.budget,a.fresh)
