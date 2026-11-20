"""Observed contexts for comparisons under the frozen v43 development Base.

Regression five/seven use the exact restored observations and enlarged TRAIN
holdout of the accepted Base curves. Clean/classification banks remain v34.
Query truth is excluded from every context-constructor interface.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import full_pool_contexts_v34 as full
import tabace_context_adaptation_v29 as adaptation
import offset_response_recovery_v41 as five
import half_coupled_recovery_v42 as seven
from prepare_data import RUN,protocol,atomic_json

OUT=RUN/'results/tabace_recovery_contexts_v44'
REGISTRY=RUN/'results/tabicl_base_context_recovery_frozen_v43/parameters.json'


def verify_registry():
    registry=json.loads(REGISTRY.read_text())
    for path,value in registry['frozen_artifact_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,('Changed frozen Base artifact',path)
    return registry


def split_path(block,meta):
    root=five.OUT if block=='five' else seven.experiment.prior.OUT
    return root/'validation_splits'/block/f'{meta["task_id"]}.npz'


def split(block,meta):
    with np.load(split_path(block,meta)) as z:return {k:z[k] for k in z.files}


def policy(block,family):
    root=five.OUT if block=='five' else seven.OUT
    return json.loads((root/f'parameters_{block}.json').read_text())['policies'][family]


def restored(env,block,family,budget):
    return five.context(env,block,family,policy(block,family),budget) if block=='five' else seven.experiment.context(env,block,family,policy(block,family),budget)


def validation_base(block,meta,family,rho,budget):
    tid=str(meta['task_id']);key=f'{family}_{rho}'
    if budget==256:
        root=five.OUT if block=='five' else seven.experiment.prior.OUT
        path=root/'base_validation'/block/f'{tid}.npz'
    elif block=='five':
        path=five.OUT/'validation/five'/f'{tid}.npz'
        key+=f'_{adaptation.raw.repaired.tag(policy(block,family))}_{budget}'
    elif family in ['domain','temporal']:
        path=seven.OUT/'validation'/f'{tid}.npz'
        key+=f'_{policy(block,family)}_{budget}'
    else:
        path=seven.experiment.prior.OUT/'validation/seven'/f'{tid}.npz'
        key+=f'_0_{budget}'
    with np.load(path) as z:return z[key],z['y_validation'],path,key


def materialize(role,block,meta,family,rho,budget,pool=None,partition=None):
    assert role in ['validation','query'] and block in ['five','seven']
    source=adaptation.raw.scaling
    if pool is None:pool=source.prepare('regression',block,meta)
    if role=='validation':
        partition=partition or split(block,meta)
        positions=partition['train_positions'];Q,t=partition['X_validation'],partition['y_validation']
        base,truth,origin,key=validation_base(block,meta,family,rho,budget)
        assert np.array_equal(t,truth)
        scale=max(float(t.std()),1e-12)
        assert not np.intersect1d(partition['source_original_ids'],partition['validation_original_ids']).size
    else:
        positions=np.arange(len(pool['y']))
        canonical,info=source.bank('regression',block,meta,family,rho)
        Q,t=canonical['X_target'],canonical['y_target'];scale=info['scale']
        root=five.OUT if block=='five' else seven.OUT
        origin=root/'query'/str(budget)/block/str(meta['task_id'])/f'{family}_{rho}.npz'
        record=json.loads(origin.with_suffix('.json').read_text())
        with np.load(origin) as z:base=z['prediction'];assert np.array_equal(t,z['y_target'])
        key='prediction'
    env=adaptation.raw.repaired.environment(pool,meta,block,family,rho,positions)
    X,y=restored(env,block,family,len(positions));n=min(budget,len(y))
    data=dict(X_source=X[:n].copy(),y_source=y[:n].copy(),X_target=Q.copy(),y_target=t.copy())
    assert np.array_equal(data['X_source'][:256],env[2][:256]) and np.array_equal(data['y_source'][:256],env[3][:256])
    assert len(base)==len(t) and all(np.isfinite(v).all() for v in data.values())
    hashes={k:protocol.array_sha256(v) for k,v in data.items()}
    if role=='query':assert hashes==record['input_hashes']
    elif budget>256:
        records=json.loads(origin.with_suffix('.json').read_text())['records']
        row=next(r for r in records if r['prediction_key']==key)
        assert hashes==row['input_hashes']
    assert not np.intersect1d(pool['source_ids'],pool['query_ids']).size
    h=min(max(1,3*len(y)//4),len(y)-1)
    return dict(canonical=data,base=base,scale=scale,Base_origin=str(origin),Base_prediction_key=key,
        provisional=False,X_pool=X,y_pool=y,history=(X[:h],y[:h]),reference=(X[h:],y[h:]),
        canonical_input_hashes=hashes,pool_input_hashes=dict(X_pool=protocol.array_sha256(X),y_pool=protocol.array_sha256(y)),
        requested_context=budget,actual_Base_context=n,available_observed_pool=len(y),
        selection_scope='full observed restored TRAIN pool; same v43 observations as paired Base',
        extra_pre_corruption_labels_given_to_constructors=False,
        benchmark_uses_extra_clean_TRAIN_information=True,
        validation_rows=len(t) if role=='validation' else None,
        query_feedback_used_for_benchmark_design=True)


def contexts(payload,block,family,meta):
    safe=full.constructor_input(payload)
    assert 'y_target' not in safe['canonical']
    shared,specific=full.contexts(safe,'regression',block,family)
    c=safe['canonical'];X,y,Q=[c[k] for k in ['X_source','y_source','X_target']]
    generic=adaptation.new_views(X,y,Q,'regression',block,family,meta)
    views=shared+generic[:5]+specific+generic[5:]
    assert len(views)==21
    for name,A,b,T in views:
        assert 0<len(b)<=payload['actual_Base_context']
        assert A.shape[1]==T.shape[1]>0 and len(A)==len(b) and len(T)==len(payload['base'])
        assert all(np.isfinite(z).all() for z in [A,b,T])
    return views


def plan():
    verify_registry();assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_NEW_INFERENCE',Base_registry=str(REGISTRY),
        budgets=[256,1024,4096,8192],seed=0,shared_members=10,specific_total_members=21,
        per_member_row_cap=True,Specific_contains_all_Shared=True,
        regenerated_blocks=['regression five','regression seven'],
        reused_immutable_blocks=['classification clean/five/seven','regression clean'],
        full_goal_metric_checks=48,extra_clean_TRAIN_information_in_benchmark=True,
        extra_clean_TRAIN_calibration_supervision=True,checkpoint_finetuned=False,
        original_query_labels_given_to_constructors=False,actual_severity_given_to_constructors=False,
        query_feedback_used_for_benchmark_design=True,development_only=True,
        original_paper_replication=False,independent_test_claim=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((adaptation.raw.OUT/'paper_targets.json').read_text())
    for role in ['validation','query']:
        for budget in [256,1024,4096,8192]:
            for task,blocks in [('classification',['clean','five','seven']),('regression',['clean'])]:
                for block in blocks:
                    dest=OUT/'banks'/role/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
                    assert not dest.exists();dest.symlink_to(full.OUT/'banks'/role/str(budget)/task/block,target_is_directory=True)


if __name__=='__main__':plan()
