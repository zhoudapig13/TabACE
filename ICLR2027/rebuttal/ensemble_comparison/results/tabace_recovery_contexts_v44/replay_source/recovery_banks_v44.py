"""Generate matched v43-observation Shared/Specific regression banks."""
import argparse
import json
import os
from collections import OrderedDict
import time
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import numpy as np
import recovery_contexts_v44 as provider
from prepare_data import atomic_json,protocol
from run_tabicl import Predictor


def case(predictor,payload,block,meta,family,rho,budget,role,out,cache):
    before=payload['canonical_input_hashes'];pool_before=payload['pool_input_hashes']
    old=provider.full.OUT/'banks'/role/str(budget)/'regression'/block/str(meta['task_id'])/out.name
    if old.with_suffix('.json').exists():
        oldmeta=json.loads(old.with_suffix('.json').read_text())
        with np.load(old) as z:
            for key,pred in zip(oldmeta['member_input_hashes'],z['members']):
                cache[tuple(key)]=pred
    c=payload['canonical'];source_key=tuple(protocol.array_sha256(c[k]) for k in ['X_source','y_source','X_target'])
    cache[source_key]=payload['base']
    start=time.monotonic();views=provider.contexts(payload,block,family,meta);construction=time.monotonic()-start
    predictions=[];names=[];rows=[];hashes=[];aliases=[]
    for name,A,b,Q in views:
        A,Q=np.asarray(A,dtype=np.float32),np.asarray(Q,dtype=np.float32)
        b=np.asarray(b);key=tuple(protocol.array_sha256(v) for v in [A,b,Q]);alias=key in cache
        if alias:p=cache[key]
        else:p,_=predictor(A,b,Q,members=1,random_state=0)
        assert len(p)==len(payload['base']) and np.isfinite(p).all()
        cache[key]=p;cache.move_to_end(key)
        while len(cache)>4096:cache.popitem(last=False)
        predictions.append(p);names.append(name);rows.append(len(b));hashes.append(key);aliases.append(alias)
    assert before=={k:protocol.array_sha256(v) for k,v in c.items()},'Mutated immutable Base observations'
    assert pool_before==dict(X_pool=protocol.array_sha256(payload['X_pool']),y_pool=protocol.array_sha256(payload['y_pool']))
    provider.adaptation.raw.scaling.npwrite(out,base=payload['base'],members=np.asarray(predictions),
        y_target=c['y_target'],source_mean=np.asarray(float(payload['y_pool'].mean())),source_scale=np.asarray(max(float(payload['y_pool'].std()),1e-8)))
    atomic_json(out.with_suffix('.json'),dict(status='BANK_COMPLETE_FUSION_PENDING',task='regression',block=block,
        task_id=meta['task_id'],dataset=meta['dataset'],family=family,rho=rho,requested_context=budget,
        actual_context=payload['actual_Base_context'],scale=payload['scale'],seed=0,
        shared_roles=names[:10],specific_additional_roles=names[10:],context_rows=rows,member_input_hashes=hashes,
        aliases=aliases,new_TFM_calls=sum(not a for a in aliases),source_input_hashes=before,
        pool_input_hashes=pool_before,available_observed_pool_rows=payload['available_observed_pool'],
        Base_consumed_rows=payload['actual_Base_context'],Base_prediction_archive=payload['Base_origin'],
        validation_rows=payload['validation_rows'],selection_scope=payload['selection_scope'],
        constructor_seconds=construction,provisional_Base_calibration=False,
        all_context_members_respect_row_cap=True,Specific_contains_all_Shared=True,
        same_available_TRAIN_dataset=True,same_selected_rows_as_Base=False,
        observed_source_only_adaptation=True,extra_clean_TRAIN_information_in_benchmark=True,
        extra_pre_corruption_labels_used_by_constructors=False,clean_TRAIN_calibration_supervision=True,
        query_labels_used_by_constructors=False,actual_severity_received_by_adapted_constructors=False,
        query_feedback_used_for_benchmark_design=True,development_only=True,original_paper_replication=False,
        independent_test_claim=False,checkpoint_finetuned=False,adaptation_version='v44'))


def worker(shard,shards,limit):
    import torch,warnings
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    provider.verify_registry();model=Predictor('regression','cuda:0');failures=[];completed=0
    jobs=provider.adaptation.raw.repaired.prior.jobs()[shard::shards]
    for block,meta in jobs:
        cache=OrderedDict()
        try:
            pool=provider.adaptation.raw.scaling.prepare('regression',block,meta)
            for role in ['validation','query']:
                partition=provider.split(block,meta) if role=='validation' else None
                for budget in [256,1024,4096,8192]:
                    for family,rho in provider.adaptation.raw.case_grid(block):
                        out=provider.OUT/'banks'/role/str(budget)/'regression'/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                        if out.exists() and out.with_suffix('.json').exists():continue
                        payload=provider.materialize(role,block,meta,family,rho,budget,pool,partition)
                        case(model,payload,block,meta,family,rho,budget,role,out,cache);completed+=1
                        if limit and completed>=limit:
                            print('RECOVERY_BANK_SMOKE_COMPLETE',str(out),flush=True);return
                    print('RECOVERY_BANK_BLOCK_COMPLETE',role,budget,block,meta['dataset'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True);failures.append(dict(block=block,task_id=meta['task_id'],error=error))
            atomic_json(provider.OUT/'errors'/f'{block}_{meta["task_id"]}.json',failures[-1])
    atomic_json(provider.OUT/'workers'/f'{shard}.json',dict(status='COMPLETE' if not failures else 'FAILED',new_cases=completed,failures=failures))
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8);p.add_argument('--limit',type=int,default=0);a=p.parse_args();worker(a.shard,a.shards,a.limit)
