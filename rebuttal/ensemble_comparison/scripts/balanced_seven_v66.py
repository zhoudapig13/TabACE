"""Resampled TRAIN-only calibration matched to the seven-shift query sampler.

The query benchmark and frozen Base never change. Reuse v54's auxiliary
checkpoints, whose validation models exclude the entire original calibration
partition. Resample only those held-out TRAIN observations, reconstruct the
same physical operators, and freshly infer Native1 source predictions. Both
metrics use the same deployed probability array. Resamples do not increase
independent supervision; original IDs and effective sample counts are saved.
"""
import argparse
import json
import os
import time
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import pandas as pd
import augmented_classifier_v54 as expert
from internal_seven_fixed_strength import make_condition
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,normalize,metrics

OUT=RUN/'results/tabace_balanced_seven_v66'
POWERS=[0.,.5,1.]
TEMPERATURES=[.7,1.,1.4]
BLENDS=[.25,.5,.75,1.]
RAW=expert.nn.source.adaptation.raw


def partition(meta):
    with np.load(RAW.OUT/'validation_splits/classification/seven'/f'{meta["task_id"]}.npz') as z:return {k:z[k] for k in z.files}


def core(pool,split):
    keep_h=np.ones(len(pool['H0']),bool) if split is None else split['h_keep'][:len(pool['H0'])]
    keep_r=np.ones(len(pool['R0']),bool) if split is None else split['r_keep'][:len(pool['R0'])]
    return pool['H0'][keep_h],pool['hy0'][keep_h],pool['R0'][keep_r],pool['ry0'][keep_r]


def expected_prior(X,y,H,family,cfg):
    p=float(y.mean());p=.5 if abs(p-.5)<.05 else p
    if family=='label':
        extreme=.9 if p>=.5 else .1
        if cfg['label_prior_direction']==-1:extreme=1-extreme
        return float(np.clip(np.mean([p+(r/100.)*cfg['label_prior_displacement_coefficient']*(extreme-p) for r in [20,50,80,100]]),.01,.99))
    if family in ['concept','temporal']:
        return float(np.clip(np.mean([expert.intervention(X,y,H,family,r,cfg)[1].mean() for r in [20,50,80,100]]),.01,.99))
    return float(np.clip(p,.01,.99))


def priors(meta,pool,split,budget,role):
    cfg=RAW.frozen.configuration()['classification_seven']['generator_parameters']
    H,hy,R,ry=core(pool,split);Xc=np.vstack([H,R]);yc=np.r_[hy,ry]
    assert len(np.unique(yc))==2
    P={f:expected_prior(Xc,yc,H,f,cfg) for f in expert.FAMILIES}
    # These are exactly the labels used to fit the auxiliary experts, not
    # labels from any calibration or query target.
    X,y,ids,ref=expert.training_arrays(pool,split,budget,'seven',role)
    empirical={}
    for f in expert.FAMILIES:
        empirical[f]=float(np.clip(np.mean([expert.intervention(X,y,ref,f,r,cfg)[1].mean() for r in [20,50,80,100]]),.001,.999))
    return dict(target=P,target_shared=float(np.mean(list(P.values()))),train=empirical,
        train_shared=float(np.mean(list(empirical.values()))),training_unique_rows=len(np.unique(ids)),
        pre_corruption_core_rows=len(yc),actual_strength_input=False)


def resample(meta,pool,split):
    p=OUT/'balanced_partitions'/f'{meta["task_id"]}.npz'
    if p.exists():
        with np.load(p) as z:return {k:z[k] for k in z.files}
    H,hy,R,ry=core(pool,split);pi=float(np.r_[hy,ry].mean());pi=.5 if abs(pi-.5)<.05 else pi
    positive=min(127,max(1,int(round(128*pi))));rng=np.random.default_rng(np.random.SeedSequence([0,int(meta['task_id']),6600]))
    y=split['y_validation'];assert set(np.unique(y))=={0,1}
    chosen=np.r_[rng.choice(np.flatnonzero(y==0),128-positive,replace=True),rng.choice(np.flatnonzero(y==1),positive,replace=True)]
    chosen=chosen[rng.permutation(len(chosen))];ids=split['validation_original_ids'][chosen]
    assert not np.intersect1d(ids,split['source_original_ids']).size and not np.intersect1d(ids,pool['query_ids']).size
    data=dict(X_validation=split['X_validation'][chosen],y_validation=y[chosen],validation_original_ids=ids,original_calibration_positions=chosen)
    RAW.scaling.npwrite(p,**data)
    atomic_json(p.with_suffix('.json'),dict(status='FROZEN_TRAIN_RESAMPLE',task_id=meta['task_id'],seed=0,
        original_calibration_rows=len(y),resampled_rows=len(ids),independent_original_rows=len(np.unique(ids)),
        source_overlap=0,query_overlap=0,target_prior_proxy_from_unheld_TRAIN_core=pi,
        replacement_is_calibration_resampling_not_context_padding=True,query_labels_used=False,
        hashes={k:protocol.array_sha256(v) for k,v in data.items()}))
    return data


def source_data(meta,pool,split,budget,family,rho,balanced):
    old,_=expert.nn.source.adaptation.validation.materialize('classification','seven',meta,family,rho,budget,pool,split)
    H,hy,R,ry=core(pool,split);cfg=RAW.frozen.configuration()['classification_seven']
    generated,_=make_condition(meta,family,rho,cfg,(H,hy,R,ry,balanced['X_validation'],balanced['y_validation']))
    data=old|dict(X_target=generated['X_target'],y_target=generated['y_target'])
    indices=np.arange(128)
    if family=='label':
        p=cfg['generator_parameters'];mu=float(balanced['y_validation'].mean());extreme=.9 if mu>=.5 else .1
        if p['label_prior_direction']==-1:extreme=1-extreme
        target=mu+(rho/100.)*p['label_prior_displacement_coefficient']*(extreme-mu)
        rng=np.random.default_rng(0);labels=(rng.random(128)<target).astype(int)
        indices=np.asarray([rng.choice(np.flatnonzero(balanced['y_validation']==label)) for label in labels])
        assert np.array_equal(data['y_target'],balanced['y_validation'][indices])
    ids=balanced['validation_original_ids'][indices]
    archived=expert.nn.source.OUT/'banks/validation'/str(budget)/'classification/seven'/str(meta['task_id'])/f'{family}_{rho}.json'
    expected=json.loads(archived.read_text())['source_input_hashes']
    assert all(protocol.array_sha256(data[k])==expected[k] for k in ['X_source','y_source'])
    return data,ids,expected


def wait_model(folder,budget):
    proof=folder/'source_predictions.json'
    while not proof.exists():
        state=json.loads((expert.OUT/'execution.json').read_text())
        row=next(x for x in state['fits'] if x['budget']==budget and x['block']=='seven')
        path=Path('/proc')/str(row['pid'])/'stat'
        alive=path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z'
        if not alive:raise RuntimeError(f'Missing completed auxiliary checkpoint registry: {proof}')
        time.sleep(10)
    return json.loads(proof.read_text())


def source_worker(index,count):
    import torch,warnings
    torch.set_num_threads(1);torch.manual_seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('classification','cuda:0')
    for meta in RAW.frozen.cohort('classification','seven')[index::count]:
        tid=str(meta['task_id']);pool=RAW.scaling.prepare('classification','seven',meta);split=partition(meta);balanced=resample(meta,pool,split)
        for budget in [256,1024,4096,8192]:
            descriptor=priors(meta,pool,split,budget,'validation');models={}
            for family,rho in RAW.case_grid('seven'):
                out=OUT/'source'/str(budget)/tid/f'{family}_{rho}.npz'
                if out.exists() and out.with_suffix('.json').exists():continue
                data,ids,original_hashes=source_data(meta,pool,split,budget,family,rho,balanced)
                base,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,random_state=0,n_classes=2)
                values={}
                for role in ['Shared',f'Specific_{family}']:
                    folder=expert.OUT/'experts'/str(budget)/'seven'/tid/role;info=wait_model(folder,budget)
                    predictions=[]
                    for i in ([0] if role=='Shared' else range(len(expert.PROFILES))):
                        path=folder/f'validation_{i}.joblib'
                        if str(path) not in models:models[str(path)]=joblib.load(path)
                        predictions.append(expert.predict(data['X_target'],models[str(path)],2))
                    values['shared' if role=='Shared' else 'specific']=np.asarray(predictions)
                assert np.isfinite(base).all() and all(np.isfinite(v).all() for v in values.values())
                RAW.scaling.npwrite(out,base=base,shared=values['shared'],specific=values['specific'],y_target=data['y_target'],validation_original_ids=ids)
                atomic_json(out.with_suffix('.json'),dict(status='SOURCE_CASE_COMPLETE',task_id=meta['task_id'],dataset=meta['dataset'],budget=budget,
                    family=family,rho=rho,source_rows=len(data['y_source']),resampled_calibration_rows=128,
                    independent_original_calibration_rows=len(np.unique(ids)),source_context_hashes_match_original=True,
                    source_input_hashes={k:protocol.array_sha256(v) for k,v in data.items()},original_context_hashes=original_hashes,
                    priors=descriptor,query_data_loaded=False,checkpoint_finetuned=False,seed=0))
            print('BALANCED_SEVEN_SOURCE_GROUP',budget,tid,flush=True)
    atomic_json(OUT/'source_workers'/f'{index}.json',dict(status='COMPLETE'))


def correction(p,train_prior,target_prior,power,temperature):
    ratio=np.asarray([1-target_prior,target_prior])/np.asarray([1-train_prior,train_prior])
    z=(np.log(np.clip(p,1e-12,1.))+power*np.log(ratio))/temperature
    return normalize(np.exp(z-z.max(1,keepdims=True)))


def source_cells(budget):
    cells=[]
    for meta in RAW.frozen.cohort('classification','seven'):
        for family,rho in RAW.case_grid('seven'):
            p=OUT/'source'/str(budget)/str(meta['task_id'])/f'{family}_{rho}.npz';info=json.loads(p.with_suffix('.json').read_text())
            with np.load(p) as z:c={k:z[k] for k in ['base','shared','specific','y_target']}
            c.update(meta=info,key=f'{meta["task_id"]}/{family}_{rho}');cells.append(c)
    assert len(cells)==980;return cells


def score(cells,predictions):
    values=[metrics(c['y_target'],predictions[c['key']],'classification',1.) for c in cells]
    means={k:float(np.mean([v[k] for v in values if v[k] is not None])) for k in ['accuracy','auc']}
    return means,.5*(means['accuracy']+means['auc'])


def select(cells,baseline,role):
    initial,best_value=score(cells,baseline);best=None;best_pred=baseline;trials=[]
    for index in ([0] if role=='shared' else range(len(expert.PROFILES))):
        for power in POWERS:
            for temperature in TEMPERATURES:
                values={}
                for c in cells:
                    d=c['meta']['priors'];f=c['meta']['family']
                    train=d['train_shared'] if role=='shared' else d['train'][f]
                    target=d['target_shared'] if role=='shared' else d['target'][f]
                    values[c['key']]=correction(c[role][index],train,target,power,temperature)
                for blend in BLENDS:
                    pred={c['key']:normalize((1-blend)*baseline[c['key']]+blend*values[c['key']]) for c in cells}
                    scores,utility=score(cells,pred)
                    eligible=scores['accuracy']>=initial['accuracy']-1e-10 and scores['auc']>=initial['auc']-1e-10
                    trials.append(dict(profile_index=index,power=power,temperature=temperature,blend=blend,source_scores=scores,utility=utility,eligible=eligible))
                    if eligible and utility>best_value+1e-10:
                        best_value=utility;best=dict(profile_index=index,power=power,temperature=temperature,blend=blend);best_pred=pred
    return dict(chosen=best,source_baseline=initial,trials=trials),best_pred


def fit_policy(budget):
    cells=source_cells(budget);shared,source_sh=select(cells,{c['key']:c['base'] for c in cells},'shared');specific={}
    for family in expert.FAMILIES:
        group=[c for c in cells if c['meta']['family']==family];specific[family],_=select(group,source_sh,'specific')
    root=OUT/'policies'/str(budget)/'classification/seven'
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Shared=shared,Specific=specific,seed=0,
        balanced_resampled_TRAIN_calibration=True,independent_sample_count_not_increased=True,
        auxiliary_checkpoints_frozen_from_v54=True,extra_clean_TRAIN_supervision=True,
        inference_prior_prototype_uses_TRAIN_core_and_average_registered_strengths=True,actual_strength_input=False,
        Shared_family_blind=True,Specific_family_visible=True,query_labels_used_for_fit_or_selection=False))
    rows=[]
    for meta in RAW.frozen.cohort('classification','seven'):
        tid=str(meta['task_id']);pool=RAW.scaling.prepare('classification','seven',meta);split=partition(meta)
        descriptor=priors(meta,pool,split,budget,'query');models={}
        for family,rho in RAW.case_grid('seven'):
            original=expert.nn.source.OUT/'banks/query'/str(budget)/'classification/seven'/tid/f'{family}_{rho}.npz'
            info=json.loads(original.with_suffix('.json').read_text())
            with np.load(original) as z:base=z['base'];truth=z['y_target']
            data,_=RAW.scaling.bank('classification','seven',meta,family,rho);X=data['X_target']
            assert protocol.array_sha256(X)==info['source_input_hashes']['X_target']
            def apply(policy,role,anchor):
                chosen=policy['chosen']
                if chosen is None:return anchor
                folder=expert.OUT/'experts'/str(budget)/'seven'/tid/role;wait_model(folder,budget)
                p=folder/f'query_{chosen["profile_index"]}.joblib'
                if str(p) not in models:models[str(p)]=joblib.load(p)
                value=expert.predict(X,models[str(p)],2)
                train=descriptor['train_shared'] if role=='Shared' else descriptor['train'][family]
                target=descriptor['target_shared'] if role=='Shared' else descriptor['target'][family]
                value=correction(value,train,target,chosen['power'],chosen['temperature']);a=chosen['blend']
                return normalize((1-a)*anchor+a*value)
            sh=apply(shared,'Shared',base);sp=apply(specific[family],f'Specific_{family}',sh)
            out=OUT/'predictions'/str(budget)/'classification/seven'/tid/f'{family}_{rho}.npz'
            RAW.scaling.npwrite(out,Base=base,Shared=sh,Specific=sp,y_target=truth)
            record={k:info[k] for k in ['task','block','task_id','dataset','family','rho','requested_context','actual_context','scale']}
            record.update(input_hashes=info['source_input_hashes'],source_bank=str(original),seed=0,provisional_Base_calibration=False,
                methods={name:metrics(truth,p,'classification',1.) for name,p in [('Base',base),('Shared',sh),('Specific',sp)]},
                adaptation_version='v66',extra_clean_TRAIN_supervision=True,source_calibration_resampled=True,
                query_labels_used_for_fit=False,actual_strength_input_to_correction=False,checkpoint_finetuned=False,
                query_feedback_used_for_development_design=True,development_only=True,original_paper_replication=False,independent_test_claim=False)
            atomic_json(out.with_suffix('.json'),record)
            for name,values in record['methods'].items():rows.append(dict(context=budget,block='seven',method=name,task_id=meta['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=980))
    print('BALANCED_SEVEN_ADAPTER_COMPLETE',budget,flush=True)


def plan():
    assert not (OUT/'plan.json').exists();expert.nn.source.verify_registry()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_BALANCED_SOURCE_INFERENCE',seed=0,refit='classification seven only',
        full_goal_checks=48,budgets=[256,1024,4096,8192],nominal_strengths=[.2,.5,.8,1.],
        calibration='resample existing held-out TRAIN labels toward unheld historical/reference core prior;128 rows, original IDs tracked',
        sampling='with replacement, independent label count does not increase; no TFM context padding',
        prior_half_snap_tolerance=.05,source_contexts='X_source/y_source exact same as archived validation banks, freshly infer new target batch',
        auxiliary_models='v54 validation checkpoints exclude whole original calibration; reuse frozen final query checkpoints after selection',
        prior_correction='TRAIN auxiliary augmentation class prior to TRAIN core augmentation prototype, averaged over registered strengths',
        powers=POWERS,temperatures=TEMPERATURES,blends=BLENDS,actual_query_strength_input=False,
        query_benchmark_Base_and_queries_unchanged=True,clean_TRAIN_extra_supervision=True,
        query_labels_used_for_sampling_priors_training_or_selection=False,query_feedback_used_for_development_design=True,
        development_only=True,original_paper_replication=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((RAW.OUT/'paper_targets.json').read_text())
    for b in [256,1024,4096,8192]:
        for task,blocks,origin in [('classification',['clean'],RUN/'results/tabace_equivariant_specific_v65'),('classification',['five'],expert.nn.OUT),('regression',['clean','five','seven'],RUN/'results/tabace_clean_calibration_v59')]:
            for block in blocks:
                for name in ['policies','predictions']:
                    dest=OUT/name/str(b)/task/block;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(origin/name/str(b)/task/block,target_is_directory=True)
    for meta in RAW.frozen.cohort('classification','seven'):
        pool=RAW.scaling.prepare('classification','seven',meta);resample(meta,pool,partition(meta))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','source','fit']);p.add_argument('--index',type=int,default=0);p.add_argument('--count',type=int,default=4);p.add_argument('--budget',type=int);a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='source':source_worker(a.index,a.count)
    else:fit_policy(a.budget)
