"""Small balanced TRAIN bags for Specific seven-shift development adapters.

Every learner sees at most256 distinct authorized TRAIN rows. Four bags can
cover extra distinct TRAIN observations; this is explicit extra supervision
and inference cost, not a matched-budget original-TabACE reproduction.
"""
import argparse
import hashlib
import json
import os
import time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import joblib
import numpy as np
import moderate_seven_v75 as parent
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabace_specific_balanced_bags_v76'
PROFILE=dict(kind='catboost',iterations=256,depth=6)
MEMBERS=4
ROW_CAP=256


def sampling_arrays(meta,pool,split,member,role):
    X=np.vstack([pool['H'],pool['R']]);y=np.r_[pool['hy'],pool['ry']];ids=np.r_[pool['h_ids'],pool['r_ids']]
    positions=np.arange(len(y))
    if role=='validation':positions=positions[~np.isin(ids,split['validation_original_ids'])]
    H,hy,R,ry=parent.parent.core(pool,split if role=='validation' else None)
    pi=float(np.r_[hy,ry].mean());pi=.5 if abs(pi-.5)<.05 else pi
    n=min(ROW_CAP,len(positions));available=[positions[y[positions]==label] for label in [0,1]]
    n1=min(len(available[1]),max(0,int(round(n*pi))));n0=min(len(available[0]),n-n1)
    n1=min(len(available[1]),n-n0);assert n0+n1==n
    rng=np.random.default_rng(np.random.SeedSequence([0,int(meta['task_id']),7600,member,int(role=='query')]))
    chosen=np.r_[rng.permutation(available[0])[:n0],rng.permutation(available[1])[:n1]]
    chosen=chosen[rng.permutation(len(chosen))]
    assert len(np.unique(ids[chosen]))==len(chosen)<=ROW_CAP
    assert not np.intersect1d(ids[chosen],pool['query_ids']).size
    if role=='validation':assert not np.intersect1d(ids[chosen],split['validation_original_ids']).size
    return X[chosen],y[chosen],ids[chosen],H


def dump(path,model):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    joblib.dump(model,temporary,compress=3);temporary.replace(path)


def worker(index,count):
    expert=parent.parent.expert;raw=parent.parent.RAW
    for meta in raw.frozen.cohort('classification','seven')[index::count]:
        tid=str(meta['task_id']);pool=raw.scaling.prepare('classification','seven',meta);split=parent.parent.partition(meta)
        for family in expert.FAMILIES:
            folder=OUT/'experts'/tid/family;proof=folder/'models_complete.json'
            if proof.exists() and all((OUT/'source'/tid/f'{family}_{rho}.npz').exists() and
                (OUT/'source'/tid/f'{family}_{rho}.json').exists() for rho in [20,50,80,100]):continue
            started=time.monotonic();records=[];train_priors=[];query_priors=[]
            for member in range(MEMBERS):
                for role in ['validation','query']:
                    path=folder/f'{role}_{member}.joblib';record_path=path.with_suffix('.json')
                    if path.exists() and record_path.exists():record=json.loads(record_path.read_text())
                    else:
                        X,y,ids,H=sampling_arrays(meta,pool,split,member,role)
                        A,b,w=expert.augment(X,y,H,'seven',[family])
                        model=expert.fit(A,b,w,PROFILE);dump(path,model)
                        record=dict(task_id=meta['task_id'],family=family,member=member,role=role,seed=0,
                            profile=PROFILE,unique_rows=len(ids),augmented_rows=len(b),original_ids=ids.tolist(),
                            source_X_sha256=protocol.array_sha256(X),source_y_sha256=protocol.array_sha256(y),
                            augmented_X_sha256=protocol.array_sha256(A),augmented_y_sha256=protocol.array_sha256(b),
                            augmentation_positive_prior=float(np.clip(np.mean(b),.001,.999)),
                            clean_class_counts=np.bincount(y.astype(int),minlength=2).tolist(),
                            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            calibration_overlap=0 if role=='validation' else 'TRAIN refit includes calibration, excludes final query',
                            query_overlap=0,query_features_or_labels_used_for_fit=False)
                        atomic_json(record_path,record)
                    records.append(record)
                    (train_priors if role=='validation' else query_priors).append(record['augmentation_positive_prior'])
            atomic_json(proof,dict(status='ALL_MODELS_COMPLETE',task_id=meta['task_id'],dataset=meta['dataset'],family=family,
                members=MEMBERS,row_cap_per_learner=ROW_CAP,train_priors=train_priors,query_priors=query_priors,
                records=records,extra_clean_TRAIN_supervision=True,query_labels_used_for_fit=False,
                source_and_query_models_use_separate_balanced_TRAIN_subsets=True,seconds=time.monotonic()-started))
            balanced=parent.parent.resample(meta,pool,split);models=[joblib.load(folder/f'validation_{i}.joblib') for i in range(MEMBERS)]
            # Auxiliary validation features are independent of the Native1 row
            # cap; verify this from all four archived source input hashes.
            for rho in [20,50,80,100]:
                out=OUT/'source'/tid/f'{family}_{rho}.npz'
                if out.exists() and out.with_suffix('.json').exists():continue
                data,ids,_=parent.parent.source_data(meta,pool,split,256,family,rho,balanced)
                qhash=protocol.array_sha256(data['X_target']);yhash=protocol.array_sha256(data['y_target'])
                for budget in [256,1024,4096,8192]:
                    original=parent.parent.OUT/'source'/str(budget)/tid/f'{family}_{rho}.json';info=json.loads(original.read_text())
                    assert qhash==info['source_input_hashes']['X_target'] and yhash==info['source_input_hashes']['y_target']
                values=np.asarray([expert.predict(data['X_target'],m,2) for m in models])
                raw.scaling.npwrite(out,predictions=values,y_validation=data['y_target'],validation_original_ids=ids)
                atomic_json(out.with_suffix('.json'),dict(status='SOURCE_BAG_PREDICTIONS_COMPLETE',task_id=meta['task_id'],family=family,rho=rho,
                    members=MEMBERS,row_cap_per_learner=ROW_CAP,source_input_hashes=dict(X_target=qhash,y_target=yhash),
                    train_priors=train_priors,query_priors=query_priors,model_registry=str(proof),
                    validation_original_ids_sha256=protocol.array_sha256(ids),query_data_loaded=False,seed=0))
            print('BALANCED_SPECIFIC_BAGS_FAMILY_COMPLETE',tid,family,flush=True)
        print('BALANCED_SPECIFIC_BAGS_DATASET_COMPLETE',tid,flush=True)
    atomic_json(OUT/'workers'/f'{index}.json',dict(status='COMPLETE',index=index,count=count))


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_NEW_BAG_TRAINING',seed=0,budgets=[256,1024,4096,8192],
        refit='Specific classification seven only',Shared_frozen_from_v75=True,all_other_blocks_frozen_from_v75=True,
        full_scope_metric_checks=48,datasets=35,query_conditions_per_budget=980,
        diagnosis='At256 the best global existing-source recipe gains only0.0167 Accuracy above Shared; cannot satisfy the0.02–0.03 source range. Expand reliable TRAIN context coverage with small class-balanced bags.',
        profile=PROFILE,members=MEMBERS,row_cap_per_learner=ROW_CAP,total_unique_TRAIN_rows_can_exceed_single_context_cap=True,
        sampling='without replacement within each bag, toward unheld historical/reference TRAIN-core prior',
        validation_models='exclude the entire old TRAIN calibration partition; final models refit on clean TRAIN only',
        known_generator_augmentation='all four nominal strengths0.2/0.5/0.8/1.0 for the known family; query strength never an input',
        validation_features_shared_across_native_context_caps='verified by hashes before reusing auxiliary predictions',
        downstream_selection='family-specific recipes cross-validated by whole dataset; global residual regularization uses calibration uncertainty',
        matched_total_inference_budget_claim=False,additional_clean_TRAIN_supervision=True,
        original_TabACE_fusion_reproduction=False,checkpoint_finetuned=False,
        query_labels_used_for_training_or_selection=False,query_feedback_used_for_development_design=True,
        development_only=True,independent_test_claim=False,original_paper_replication=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((parent.parent.expert.OUT/'paper_targets.json').read_text())
    for budget in [256,1024,4096,8192]:
        for task in ['classification','regression']:
            for block in ['clean','five','seven']:
                if task=='classification' and block=='seven':continue
                for kind in ['policies','predictions']:
                    dest=OUT/kind/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.symlink_to(parent.OUT/kind/str(budget)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker']);p.add_argument('--index',type=int,default=0);p.add_argument('--count',type=int,default=8)
    a=p.parse_args();plan() if a.phase=='plan' else worker(a.index,a.count)
