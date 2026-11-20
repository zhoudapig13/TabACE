"""Infer restored pool-based contexts and reuse exact canonical v29 branches."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'1')
import numpy as np
import full_pool_contexts_v34 as experiment
import tabace_context_adaptation_v29 as archive
from prepare_data import atomic_json,protocol
from run_tabicl import Predictor


def case(predictor,payload,task,block,meta,family,rho,budget,role,old,out):
    previous=json.loads(old.with_suffix('.json').read_text())
    assert previous['source_input_hashes']==payload['canonical_input_hashes'],'Cannot reuse changed canonical inputs'
    with np.load(old) as bank:base=bank['base'];members=bank['members'];truth=bank['y_target']
    assert np.array_equal(base,payload['base']) and np.array_equal(truth,payload['canonical']['y_target'])
    safe=experiment.constructor_input(payload);before=(protocol.array_sha256(safe['X_pool']),protocol.array_sha256(safe['y_pool']))
    start=time.monotonic();shared,specific=experiment.contexts(safe,task,block,family);construction=time.monotonic()-start
    cached={tuple(h):p for h,p in zip(previous['member_input_hashes'],members)}
    canonical=payload['canonical'];source_key=tuple(protocol.array_sha256(canonical[k]) for k in ['X_source','y_source','X_target']);cached[source_key]=base
    raw=[];names=[];rows=[];hashes=[];aliases=[];seconds=[]
    for name,A,b,Q in shared+specific:
        key=tuple(protocol.array_sha256(v) for v in [A,b,Q]);started=time.monotonic();alias=key in cached
        if alias:p=cached[key]
        else:p,_=predictor(A,b,Q,members=1,random_state=0,n_classes=base.shape[1] if task=='classification' else None)
        assert np.isfinite(p).all();cached[key]=p;raw.append(p);names.append(name);rows.append(len(b));hashes.append(key);aliases.append(alias);seconds.append(time.monotonic()-started)
    assert before==(protocol.array_sha256(safe['X_pool']),protocol.array_sha256(safe['y_pool'])),'Constructor/inference mutated source pool'
    assert payload['canonical_input_hashes']=={k:protocol.array_sha256(v) for k,v in payload['canonical'].items()},'Constructor/inference mutated canonical Base inputs'
    result=np.concatenate([np.asarray(raw[:5]),members[5:10],np.asarray(raw[5:]),members[15:21]])
    join=lambda fresh,prior:fresh[:5]+prior[5:10]+fresh[5:]+prior[15:21]
    means=float(payload['y_pool'].mean());scale=max(float(payload['y_pool'].std()),1e-8)
    experiment.reference.scaling.npwrite(out,base=base,members=result,y_target=truth,source_mean=np.asarray(means),source_scale=np.asarray(scale))
    role_names=names[:5]+previous['shared_roles'][5:]+names[5:]+previous['specific_additional_roles'][5:]
    assert result.shape[0]==21 and len(role_names)==21
    atomic_json(out.with_suffix('.json'),previous|dict(status='FULL_POOL_BANK_COMPLETE_FUSION_PENDING',
        shared_roles=role_names[:10],specific_additional_roles=role_names[10:],context_rows=join(rows,previous['context_rows']),
        member_input_hashes=join(hashes,previous['member_input_hashes']),
        aliases=join(aliases,[True]*21),member_seconds=join(seconds,[0.]*21),construction_seconds=construction,
        available_observed_pool_rows=len(payload['y_pool']),Base_consumed_rows=payload['actual_Base_context'],
        pool_input_hashes=payload['pool_input_hashes'],canonical_prediction_archive=str(old),
        canonical_generic_members_reused=11,new_TFM_calls=sum(not x for x in aliases),
        all_context_members_respect_row_cap=True,Specific_contains_all_Shared=True,
        same_selected_rows_as_Base=False,same_available_TRAIN_dataset=True,
        extra_pre_corruption_labels_used_by_constructors=False,
        selection_scope='full observed TRAIN pool; Base prediction unchanged',adaptation_version='v34',
        observed_source_information_matched=False,
        query_labels_used_by_constructors=False,development_only=True,original_paper_replication=False))


def worker(shard,shards):
    import warnings,torch
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictors={};failures=0;pending=[];completed=0
    for task,track,meta in experiment.reference.jobs()[shard::shards]:
        try:
            pool=experiment.reference.scaling.prepare(task,track,meta)
            for role in ['validation','query']:
                split=experiment.validation.validation_split(task,track,meta,pool) if role=='validation' else None
                for budget in [256,1024,4096,8192]:
                    for block in ['clean','five'] if track=='five' else ['seven']:
                        for family,rho in experiment.reference.case_grid(block):
                            relative=Path(role)/str(budget)/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                            old=archive.OUT/'banks'/relative;out=experiment.OUT/'banks'/relative
                            if out.exists() and out.with_suffix('.json').exists():continue
                            if not old.exists() or not old.with_suffix('.json').exists():pending.append(str(old));continue
                            payload=experiment.materialize(role,task,block,meta,family,rho,budget,pool,split)
                            if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
                            case(predictors[task],payload,task,block,meta,family,rho,budget,role,old,out);completed+=1
                        print('FULL_POOL_BLOCK_VISITED',role,budget,task,block,meta['dataset'],flush=True)
        except Exception:
            failures+=1;error=traceback.format_exc();print(error,flush=True)
            atomic_json(experiment.OUT/'errors'/f'{task}_{track}_{meta["task_id"]}.json',dict(error=error))
    atomic_json(experiment.OUT/'worker_completion'/f'{shard}.json',dict(
        status='FAILED' if failures else 'PENDING_ARCHIVE' if pending else 'COMPLETE',
        shard=shard,new_completed_conditions=completed,pending_archive=pending,failures=failures))
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=8);a=p.parse_args()
    worker(a.shard,a.shards)
