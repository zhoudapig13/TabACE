"""Reconstruct exact observed label priors for every existing Clean context.

Every reconstructed X/y/query-feature input triple must match the archived
member hash before its label histogram is accepted. No TFM calls or query
labels are used to estimate priors. This tests whether a canonical-prior proxy
miscalibrates branches selected with class balancing or label repair.
"""
import argparse
import json
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import full_pool_contexts_v34 as full
import tabace_context_adaptation_v29 as generic
import prior_fusion_v61 as prior
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabace_member_priors_v63'


def one(meta,budget,role):
    tid=str(meta['task_id']);out=OUT/role/str(budget)/f'{tid}.json'
    if out.exists():return
    pool=full.reference.scaling.prepare('classification','five',meta)
    split=full.validation.validation_split('classification','five',meta,pool) if role=='validation' else None
    payload=full.materialize(role,'classification','clean',meta,'original',0,budget,pool,split)
    safe=full.constructor_input(payload);assert 'y_target' not in safe['canonical']
    common,specific=full.contexts(safe,'classification','clean','original')
    canonical=safe['canonical'];X,y,Q=[canonical[k] for k in ['X_source','y_source','X_target']]
    extra=generic.new_views(X,y,Q,'classification','clean','original',meta)
    views=common+extra[:5]+specific+extra[5:];assert len(views)==21
    archive=prior.risk.head.source.OUT/'banks'/role/str(budget)/'classification/clean'/tid/'original_0.npz'
    info=json.loads(archive.with_suffix('.json').read_text())
    with np.load(archive) as z:k=z['base'].shape[1]
    assert payload['canonical_input_hashes']==info['source_input_hashes']
    histograms=[];rows=[]
    for i,(name,A,b,T) in enumerate(views):
        A,T=np.asarray(A,dtype=np.float32),np.asarray(T,dtype=np.float32);b=np.asarray(b)
        hashes=[protocol.array_sha256(v) for v in [A,b,T]]
        assert hashes==info['member_input_hashes'][i],(role,budget,tid,i,name,hashes,info['member_input_hashes'][i])
        assert len(b)<=budget and len(b)==info['context_rows'][i]
        histograms.append(np.bincount(b.astype(int),minlength=k).tolist());rows.append(len(b))
    native=np.bincount(y.astype(int),minlength=k).astype(float)
    core=np.bincount(pool['y'][:256].astype(int),minlength=k).astype(float)
    raw=np.asarray(histograms,dtype=float);branch_p=raw/raw.sum(1,keepdims=True);native_p=native/native.sum()
    atomic_json(out,dict(status='PASS',task_id=meta['task_id'],dataset=meta['dataset'],budget=budget,role=role,
        member_label_counts=histograms,canonical_label_counts=native.tolist(),core_label_counts=core.tolist(),
        context_rows=rows,member_names=info['shared_roles']+info['specific_additional_roles'],
        member_input_hashes=info['member_input_hashes'],canonical_input_hashes=info['source_input_hashes'],
        all21_input_triples_exact=True,per_member_row_caps_verified=True,
        branch_canonical_prior_total_variation=(.5*np.abs(branch_p-native_p[None]).sum(1)).tolist(),
        query_target_labels_used_for_priors=False,TFM_calls=0,seed=0))
    print('EXACT_MEMBER_PRIORS',role,budget,tid,flush=True)


def worker(index,count,limit):
    verified=0
    for meta in full.reference.frozen.cohort('classification','clean')[index::count]:
        for role in ['validation','query']:
            for budget in [256,1024,4096,8192]:
                one(meta,budget,role);verified+=1
                if limit and verified>=limit:return


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_PRIOR_RECONSTRUCTION',seed=0,source_banks=str(prior.risk.head.source.OUT),
        tasks='all72 classification Clean datasets',budgets=[256,1024,4096,8192],roles=['validation','query'],
        required_cases=576,required_member_input_triples=12096,query_labels_used_for_priors=False,extra_TFM_calls=0,
        hypothesis='Canonical TRAIN class-prior proxy may differ from class-balanced/repaired individual context labels',
        rejection_rule='any input hash or row-count mismatch fails the case; never estimate from a nonmatching view',
        downstream='prior calibration or hierarchical Specific fitting only after reconstruction passes',Markdown_written=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker']);p.add_argument('--index',type=int,default=0);p.add_argument('--count',type=int,default=8);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    plan() if a.phase=='plan' else worker(a.index,a.count,a.limit)
