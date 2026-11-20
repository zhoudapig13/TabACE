"""Larger disjoint TRAIN calibration for the unresolved Clean8192 Specific.

Shared and all query conditions stay immutable. Only source calibration is
expanded (up to512, protecting the original core), and every validation expert
is refit excluding the expanded partition. Final query recipes reuse existing
models fitted on the capped original TRAIN pool. No query score selects a
recipe. This is a development adapter with explicit extra TRAIN supervision.
"""
import argparse
import json
import os
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import pandas as pd
import bagged_specific_v56 as bag
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,metrics

OUT=RUN/'results/tabace_clean_calibration_v59'
BUDGET=8192


def partition(meta):
    p=OUT/'validation_splits'/f'{meta["task_id"]}.npz'
    if p.exists():
        with np.load(p) as z:return {k:z[k] for k in z.files}
    parent=bag.parent;pool=parent.h.source.adaptation.raw.scaling.prepare('regression','five',meta)
    old=parent.h.calibration(meta,'clean');lookup={int(i):j for j,i in enumerate(pool['source_ids'])}
    previous=np.asarray([lookup[int(i)] for i in old['validation_original_ids']]);assert np.all(previous>=256)
    assert np.array_equal(pool['X'][previous],old['X_validation']) and np.array_equal(pool['y'][previous],old['y_validation'])
    desired=min(512,max(len(previous),(len(pool['y'])-256)//2))
    candidates=np.setdiff1d(np.arange(256,len(pool['y'])),previous)
    extra=np.random.default_rng(np.random.SeedSequence([0,int(meta['task_id']),5900])).permutation(candidates)[:desired-len(previous)]
    held=np.r_[previous,extra];train=np.setdiff1d(np.arange(len(pool['y'])),held)
    result=dict(X_validation=pool['X'][held],y_validation=pool['y'][held],validation_original_ids=pool['source_ids'][held],
        train_positions=train,held_positions=held,source_original_ids=pool['source_ids'][train])
    assert not np.intersect1d(result['validation_original_ids'],result['source_original_ids']).size
    assert not np.intersect1d(result['validation_original_ids'],pool['query_ids']).size
    parent.h.source.adaptation.raw.scaling.npwrite(p,**result)
    atomic_json(p.with_suffix('.json'),dict(status='FROZEN_SOURCE_PARTITION',task_id=meta['task_id'],old_rows=len(previous),
        calibration_rows=len(held),source_rows=len(train),protected_original_core256=True,query_overlap=0,
        hashes={k:protocol.array_sha256(v) for k,v in result.items()},seed=0))
    return result


def native_worker():
    import torch,warnings
    torch.set_num_threads(1);torch.manual_seed(0);warnings.filterwarnings('ignore')
    torch.cuda.set_per_process_memory_fraction(.2)
    predictor=Predictor('regression','cuda:0');parent=bag.parent
    for meta in parent.h.source.adaptation.raw.frozen.cohort('regression','clean'):
        tid=str(meta['task_id']);p=OUT/'source_native'/f'{tid}.npz'
        if p.exists() and p.with_suffix('.json').exists():continue
        cal=partition(meta);pool=parent.h.source.adaptation.raw.scaling.prepare('regression','five',meta)
        pos=cal['train_positions'][:BUDGET];X,y=pool['X'][pos],pool['y'][pos]
        assert len(pos)<=BUDGET and np.array_equal(X[:256],pool['X'][:256]) and np.array_equal(y[:256],pool['y'][:256])
        value,_=predictor(X,y,cal['X_validation'],members=1,random_state=0)
        assert np.isfinite(value).all()
        parent.h.source.adaptation.raw.scaling.npwrite(p,prediction=value,y_validation=cal['y_validation'])
        atomic_json(p.with_suffix('.json'),dict(status='SOURCE_NATIVE_COMPLETE',task_id=meta['task_id'],source_rows=len(pos),
            calibration_rows=len(value),source_validation_overlap=0,hashes={k:protocol.array_sha256(v) for k,v in [('X_source',X),('y_source',y),('X_validation',cal['X_validation'])]},
            n_estimators=1,seed=0,checkpoint_finetuned=False,query_data_loaded=False))
        print('EXPANDED_CALIBRATION_NATIVE_COMPLETE',tid,flush=True)


def expert_worker(index,count):
    parent=bag.parent
    for meta in parent.h.source.adaptation.raw.frozen.cohort('regression','clean')[index::count]:
        tid=str(meta['task_id']);p=OUT/'source_experts'/f'{tid}.npz'
        if p.exists() and p.with_suffix('.json').exists():continue
        pool=parent.h.source.adaptation.raw.scaling.prepare('regression','five',meta);cal=partition(meta);pos=cal['train_positions'][:BUDGET]
        values=[];bagged=[];records=[];folder=p.parent/'models'/tid;folder.mkdir(parents=True,exist_ok=True)
        for i,profile in enumerate(parent.PROFILES):
            model=parent.fit(pool['X'][pos],pool['y'][pos],profile);values.append(parent.h.head_predict(cal['X_validation'],model))
            joblib.dump(model,folder/f'single_{i}.joblib',compress=3)
        for i,profile in enumerate(bag.PROFILES):
            members=[]
            for k in range(bag.MEMBERS):
                selected=bag.subset(pool,cal,BUDGET,k,tid,'validation')
                model=bag.fit(pool['X'][selected],pool['y'][selected],profile)
                members.append(parent.h.head_predict(cal['X_validation'],model));joblib.dump(model,folder/f'bag_{i}_{k}.joblib',compress=3)
                records.append(dict(profile_index=i,member=k,source_original_ids=pool['source_ids'][selected].tolist(),
                    source_X_sha256=protocol.array_sha256(pool['X'][selected]),source_y_sha256=protocol.array_sha256(pool['y'][selected])))
            bagged.append(members)
        parent.h.source.adaptation.raw.scaling.npwrite(p,single=np.asarray(values),bagged=np.asarray(bagged),y_validation=cal['y_validation'])
        atomic_json(p.with_suffix('.json'),dict(status='SOURCE_EXPERTS_COMPLETE',task_id=meta['task_id'],single_source_rows=len(pos),
            single_source_ids=pool['source_ids'][pos].tolist(),bagged_records=records,calibration_rows=len(cal['y_validation']),
            source_validation_overlap=0,query_data_loaded=False))
        print('EXPANDED_CALIBRATION_EXPERTS_COMPLETE',tid,flush=True)


def score(cells,values):
    scores=[metrics(c['y'],v,'regression',max(float(np.std(c['y'])),1e-12)) for c,v in zip(cells,values)]
    means={k:float(np.mean([s[k] for s in scores])) for k in ['r2','nrmse']}
    return means,.5*(means['r2']-means['nrmse'])


def select():
    parent=bag.parent
    original=json.loads((parent.OUT/'policies/8192/regression/clean/all_policies_frozen.json').read_text())
    shared=original['Shared']['chosen'];cells=[];baseline=[]
    for meta in parent.h.source.adaptation.raw.frozen.cohort('regression','clean'):
        tid=str(meta['task_id']);cal=partition(meta)
        with np.load(OUT/'source_native'/f'{tid}.npz') as z:
            assert np.array_equal(z['y_validation'],cal['y_validation']);base=z['prediction']
        with np.load(OUT/'source_experts'/f'{tid}.npz') as z:
            assert np.array_equal(z['y_validation'],cal['y_validation']);single=z['single'];bags=z['bagged']
        sh=base if shared is None else (1-shared['blend'])*base+shared['blend']*single[shared['profile_index']]
        cells.append(dict(meta=meta,y=cal['y_validation'],single=single,bagged=bags));baseline.append(sh)
    initial,best_value=score(cells,baseline);best=None;trials=[]
    recipes=[dict(kind='single',profile_index=i,members=1) for i in range(len(parent.PROFILES))]+[
        dict(kind='bagged',profile_index=i,members=n) for i in range(len(bag.PROFILES)) for n in [4,8]]
    for recipe in recipes:
        values=[c['single'][recipe['profile_index']] if recipe['kind']=='single' else c['bagged'][recipe['profile_index'],:recipe['members']].mean(0) for c in cells]
        for blend in parent.BLENDS:
            pred=[(1-blend)*sh+blend*v for sh,v in zip(baseline,values)];scores,utility=score(cells,pred)
            eligible=scores['r2']>initial['r2']+1e-10 and scores['nrmse']<initial['nrmse']-1e-10
            trials.append(dict(recipe=recipe,blend=blend,source_scores=scores,utility=utility,eligible=eligible))
            if eligible and utility>best_value+1e-10:best_value=utility;best=dict(recipe=recipe,blend=blend)
    root=OUT/'policies/8192/regression/clean'
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Specific=best,Shared_unchanged=True,
        source_Shared_recipe=shared,source_Shared_scores=initial,trials=trials,seed=0,calibration_rows={str(c['meta']['task_id']):len(c['y']) for c in cells},
        additional_clean_TRAIN_supervision=True,query_labels_used_for_selection=False,existing_query_models_reused=True))
    print('EXPANDED_CALIBRATION_POLICY_FROZEN',best,flush=True)


def query():
    parent=bag.parent;root=OUT/'policies/8192/regression/clean';policy=json.loads((root/'all_policies_frozen.json').read_text());choice=policy['Specific'];rows=[]
    for meta in parent.h.source.adaptation.raw.frozen.cohort('regression','clean'):
        tid=str(meta['task_id']);old=bag.OUT/'predictions/8192/regression/clean'/tid/'original_0.npz'
        with np.load(old) as z:base=z['Base'];sh=z['Shared'];truth=z['y_target']
        sp=sh
        if choice is not None:
            recipe=choice['recipe'];data,_=parent.h.source.adaptation.raw.scaling.bank('regression','clean',meta,'original',0);X=data['X_target'];values=[]
            if recipe['kind']=='single':
                path=parent.OUT/'experts/8192/clean/models'/tid/f'query_{recipe["profile_index"]}.joblib';values.append(parent.h.head_predict(X,joblib.load(path)))
            else:
                for k in range(recipe['members']):
                    path=bag.OUT/'experts/8192/clean/models'/tid/f'query_{recipe["profile_index"]}_{k}.joblib';values.append(parent.h.head_predict(X,joblib.load(path)))
            a=choice['blend'];sp=(1-a)*sh+a*np.mean(values,axis=0)
        out=OUT/'predictions/8192/regression/clean'/tid/'original_0.npz';parent.h.source.adaptation.raw.scaling.npwrite(out,Base=base,Shared=sh,Specific=sp,y_target=truth)
        record=json.loads(old.with_suffix('.json').read_text());record.update(adaptation_version='v59',Shared_bitwise_unchanged_from_v53=True,
            larger_disjoint_TRAIN_calibration=True,Specific_recipe=choice,
            methods={n:metrics(truth,p,'regression',record['scale']) for n,p in [('Base',base),('Shared',sh),('Specific',sp)]})
        atomic_json(out.with_suffix('.json'),record)
        for name,values in record['methods'].items():rows.append(dict(context=8192,block='clean',method=name,task_id=meta['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=35,Shared_unchanged=True,adaptation_marker='dagger'))
    print('EXPANDED_CALIBRATION_QUERY_COMPLETE',flush=True)


def plan():
    assert not (OUT/'plan.json').exists();bag.parent.h.source.verify_registry()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_LARGER_CALIBRATION',seed=0,refit='Specific regression Clean8192 only',
        full_goal_checks=48,Shared_bitwise_unchanged_from_v53=True,Base_and_all_query_tasks_unchanged=True,
        purpose='Check selection variance behind the two residual Clean8192 ordering failures',
        calibration='retain original TRAIN calibration, add deterministic eligible TRAIN rows up to512, protect core256',
        validation_models='every source model refit excluding entire expanded partition; no old validation predictions reused',
        source_TFM_native='official v2 Native1, cap8192, original clean TRAIN only, no query data',
        auxiliary_candidates='same frozen final query models as v53 and v56; single and bagged recipes',
        original_query_labels_used_for_fit_or_selection=False,query_feedback_used_for_development_design=True,
        extra_clean_TRAIN_supervision=True,checkpoint_finetuned=False,development_only=True,original_paper_replication=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((bag.OUT/'paper_targets.json').read_text())
    for b in [256,1024,4096,8192]:
        for task in ['classification','regression']:
            for block in ['clean','five','seven']:
                if (b,task,block)==(8192,'regression','clean'):continue
                parent=RUN/'results/tabace_augmented_classifier_v54' if task=='classification' else bag.OUT
                for name in ['policies','predictions']:
                    dest=OUT/name/str(b)/task/block;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(parent/name/str(b)/task/block,target_is_directory=True)
    for meta in bag.parent.h.source.adaptation.raw.frozen.cohort('regression','clean'):partition(meta)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','native','experts','select','query']);p.add_argument('--index',type=int,default=0);p.add_argument('--count',type=int,default=8);a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='native':native_worker()
    elif a.phase=='experts':expert_worker(a.index,a.count)
    elif a.phase=='select':select()
    else:query()
