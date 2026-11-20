"""Verify all new TRAIN banks and replay the frozen Specific deployment."""
import argparse
import hashlib
import json
import time
import numpy as np
import balanced_clean_calibration_v71 as e
from prepare_data import atomic_json, protocol
from run_tabicl import normalize, metrics, Predictor, CHECKPOINTS


def transform(p, ratio, power, temperature):
    logs = (np.log(np.clip(p,1e-12,1.))+power*np.log(ratio))/temperature
    return normalize(np.exp(logs-logs.max(-1,keepdims=True)))


def value(c, previous, choice):
    if choice['source']=='member_mean':
        p = np.mean([transform(p,c['core_prior']/prior,choice['prior_power'],1.)
                     for p,prior in zip(c['bank'],c['member_priors'])],axis=0)
        return transform(p,np.ones(p.shape[1]),0.,choice['temperature'])
    p = previous if choice['source']=='previous_specific' else c['base']
    return transform(p,c['core_prior']/c['context_prior'],choice['prior_power'],choice['temperature'])


def audit(fresh):
    e.parent.configure()
    model = Predictor('classification','cuda:0') if fresh else None
    source_count=query_count=source_triples=fresh_count=0
    maximum=fresh_max=0.
    checkpoint=CHECKPOINTS/'tabicl-classifier-v2-20260212.ckpt'
    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_sha=='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
    import member_priors_v63 as exact_priors
    for budget in e.BUDGETS:
        cells=e.source_cells(budget)
        for c in cells:
            meta=c['meta'];tid=str(meta['task_id'])
            pool=e.full.reference.scaling.prepare('classification','five',meta)
            split=e.partition(meta,pool)
            assert np.array_equal(c['ids'],split['validation_original_ids'])
            assert np.array_equal(c['y'],split['y_validation'])
            assert not np.intersect1d(c['ids'],split['source_original_ids']).size
            assert not np.intersect1d(c['ids'],pool['query_ids']).size
            held_labels=pool['y'][split['unheld_core_positions']].astype(int)
            counts=np.bincount(held_labels,minlength=c['base'].shape[1])
            expected=(counts+.5)/(counts.sum()+.5*len(counts))
            assert np.array_equal(c['core_prior'],expected)
            views,canonical=e.construct(meta,pool,split,budget,c['base'])
            assert {k:protocol.array_sha256(v) for k,v in canonical.items()}==meta['source_input_hashes']
            for i,(name,A,b,Q) in enumerate(views):
                A,b,Q=np.asarray(A,dtype=np.float32),np.asarray(b),np.asarray(Q,dtype=np.float32)
                assert [protocol.array_sha256(v) for v in [A,b,Q]]==meta['member_input_hashes'][i]
                assert len(b)==meta['context_rows'][i]<=budget
                source_triples+=1
            assert np.isfinite(c['bank']).all() and np.allclose(c['bank'].sum(2),1.,atol=1e-12)
            if fresh and int(tid) in [167121,167124]:
                X,y,Q=[canonical[k] for k in ['X_source','y_source','X_target']]
                p,_=model(X,y,Q,members=1,random_state=0,n_classes=c['base'].shape[1])
                error=float(np.max(np.abs(p-c['base'])));fresh_max=max(fresh_max,error)
                assert error<1e-5,(tid,budget,'base',error)
                fresh_count+=1
                # The class-balanced view is chosen by registered role, not its score.
                i=next((i for i,(name,_,_,_) in enumerate(views) if 'balanced' in name),0)
                _,A,b,Q=views[i]
                p,_=model(A,b,Q,members=1,random_state=0,n_classes=c['base'].shape[1])
                with np.load(c['path']) as z:cached=z['members'][i]
                error=float(np.max(np.abs(p-cached)));fresh_max=max(fresh_max,error)
                assert error<1e-5,(tid,budget,'member',i,error)
                fresh_count+=1
            source_count+=1
        anchors,previous=e.parent_source(cells,budget)
        root=e.OUT/'policies'/str(budget)/'classification/clean'
        freeze=root/'all_policies_frozen.json';policy=json.loads(freeze.read_text());choice=policy['Specific']
        measured=e.parent.prior.score(cells,anchors)[0]
        assert all(abs(measured[k]-policy['source_baseline'][k])<1e-12 for k in measured)
        if choice is not None:
            mixed={}
            for c in cells:
                p=value(c,previous[c['key']],choice);a=choice['blend']
                mixed[c['key']]=normalize((1-a)*anchors[c['key']]+a*p)
            measured=e.parent.prior.score(cells,mixed)[0]
            selected=next(t for t in policy['trials'] if all(t[k]==v for k,v in choice.items()))
            assert all(abs(measured[k]-selected['source_scores'][k])<1e-12 for k in measured)
        for c in e.parent.load('query',budget):
            old=e.parent.OUT/'predictions'/str(budget)/c['key']
            with np.load(old) as z:base=z['Base'];sh=z['Shared'];previous=z['Specific'];truth=z['y_target']
            info=json.loads((exact_priors.OUT/'query'/str(budget)/f'{c["meta"]["task_id"]}.json').read_text())
            hist=np.asarray(info['member_label_counts'],dtype=float)[c['order']]
            c['member_priors']=(hist+.5)/(hist.sum(1,keepdims=True)+.5*hist.shape[1])
            p=sh if choice is None else normalize((1-choice['blend'])*sh+choice['blend']*value(c,previous,choice))
            out=e.OUT/'predictions'/str(budget)/c['key'];record=json.loads(out.with_suffix('.json').read_text())
            assert freeze.stat().st_mtime<=out.stat().st_mtime
            with np.load(out) as z:
                assert np.array_equal(z['Base'],base) and np.array_equal(z['Shared'],sh)
                assert np.array_equal(z['y_target'],truth)
                error=float(np.max(np.abs(z['Specific']-p)));maximum=max(maximum,error);assert error<1e-12
                for name in ['Base','Shared','Specific']:
                    scores=metrics(truth,z[name],'classification',1.)
                    assert all((v is None and record['methods'][name][k] is None) or abs(v-record['methods'][name][k])<1e-12 for k,v in scores.items())
            query_count+=1
    assert source_count==216 and query_count==216 and source_triples==4536
    assert not fresh or fresh_count==12
    report=dict(status='PASS',source_banks_verified=source_count,source_member_input_triples_verified=source_triples,
                query_conditions_replayed=query_count,maximum_fusion_error=maximum,
                fresh_Native1_cases=fresh_count,fresh_Native1_maximum_error=fresh_max,
                checkpoint_hash=checkpoint_sha,all_source_original_IDs_disjoint=True,
                core_priors_exclude_all_held_rows=True,source_selected_scores_recomputed=True,
                Base_and_Shared_bitwise_unchanged=True,all_query_metrics_recomputed=True,
                frozen_policies_precede_query_outputs=True,query_labels_used_for_fit_or_selection=False,
                development_only=True,independent_test_claim=False,Markdown_written=False,time_unix=time.time())
    atomic_json(e.OUT/('raw_audit.json' if fresh else 'structural_audit.json'),report)
    print('BALANCED_CLEAN_AUDIT_PASS',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');args=parser.parse_args()
    audit(args.fresh)
