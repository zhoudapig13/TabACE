"""Reassign 64 old query rows to labeled calibration; reserve new evaluation rows.

No evaluation label is stored with model inputs. Real-feature preprocessing
and projection fit only the original source rows. Synthetic query covariates
remain explicitly label-conditional benchmark generation, never natural data.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

import semisynthetic_abcd_v92 as parent
from prepare_data import RUN,LOCAL_CC18,atomic_json,protocol

OUT=RUN/'results/target_calibration_split_v97'
CALIBRATION_ROWS=64
MAX_EVALUATION_ROWS=96


def sha_rows(X):
    return [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in X]


def numeric_source_only(source,others):
    a=np.asarray(source,dtype=object);b=np.asarray(others,dtype=object)
    A=np.empty(a.shape,dtype=np.float32);B=np.empty(b.shape,dtype=np.float32);categorical=[]
    for j in range(a.shape[1]):
        try:
            A[:,j]=a[:,j].astype(np.float32)
            B[:,j]=pd.to_numeric(pd.Series(b[:,j]),errors='coerce').to_numpy(dtype=np.float32)
        except (TypeError,ValueError):
            categorical.append(j)
    if categorical:
        encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1)
        A[:,categorical]=encoder.fit_transform(a[:,categorical]);B[:,categorical]=encoder.transform(b[:,categorical])
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(A)
    return imputer.transform(A).astype(np.float32),imputer.transform(B).astype(np.float32),categorical


def prepare(meta,seven_ids):
    task_id=int(meta['task_id']);out=OUT/'inputs'/f'{task_id}.npz'
    if out.with_suffix('.json').exists():return json.loads(out.with_suffix('.json').read_text())
    folder=LOCAL_CC18/str(task_id)
    frame=pd.read_parquet(folder/'X.parquet');raw_y=pd.read_parquet(folder/'y.parquet').iloc[:,0]
    valid=np.flatnonzero(~raw_y.isna().to_numpy())
    with np.load(RUN/'cache/five/classification'/f'{task_id}.npz') as z:
        source_ids=z['source_indices'];old_query_ids=z['target_indices'];source_y=z['y_source'];old_y=z['y_target']
    label_encoder=LabelEncoder().fit(raw_y.iloc[source_ids])
    assert np.array_equal(label_encoder.transform(raw_y.iloc[source_ids]),source_y)
    assert np.array_equal(label_encoder.transform(raw_y.iloc[old_query_ids]),old_y)
    excluded=set(source_ids.tolist())|set(old_query_ids.tolist())
    prior_seven_query=[]
    if task_id in seven_ids:
        full_labels=LabelEncoder().fit_transform(raw_y.iloc[valid])
        selected=protocol._balanced_cap_indices(full_labels,448,0)
        _,q=train_test_split(selected,train_size=320,test_size=128,stratify=full_labels[selected],random_state=0)
        prior_seven_query=valid[q].tolist();excluded.update(prior_seven_query)
    candidates=np.array([i for i in valid if int(i) not in excluded],dtype=int)
    candidate_labels=label_encoder.transform(raw_y.iloc[candidates])
    positions=protocol._balanced_cap_indices(candidate_labels,min(len(candidates),512),97000+task_id)
    candidates=candidates[positions];candidate_labels=candidate_labels[positions]
    source_X,converted,categorical=numeric_source_only(frame.iloc[source_ids],frame.iloc[np.r_[old_query_ids,candidates]])
    old_X,candidate_X=converted[:128],converted[128:]
    hashes=set(sha_rows(np.vstack([source_X,old_X])))
    rng=np.random.default_rng(97000+task_id)
    queues={}
    for label in np.unique(candidate_labels):
        ids=np.flatnonzero(candidate_labels==label);rng.shuffle(ids);queues[int(label)]=list(ids)
    chosen=[]
    while len(chosen)<MAX_EVALUATION_ROWS and any(queues.values()):
        for label in sorted(queues):
            while queues[label]:
                index=queues[label].pop()
                signature=sha_rows(candidate_X[index:index+1])[0]
                if signature not in hashes:
                    chosen.append(index);hashes.add(signature);break
            if len(chosen)>=MAX_EVALUATION_ROWS:break
    chosen=np.asarray(chosen,dtype=int)
    assert len(chosen)>=16,(task_id,'insufficient distinct unused evaluation rows',len(chosen))
    fresh_ids=candidates[chosen];fresh_X=candidate_X[chosen];fresh_y=candidate_labels[chosen]
    cal_position,_=train_test_split(np.arange(128),train_size=CALIBRATION_ROWS,stratify=old_y,random_state=0)
    cal_position=np.asarray(cal_position,dtype=int);cal_ids=old_query_ids[cal_position];cal_y=old_y[cal_position]
    assert set(cal_ids).isdisjoint(fresh_ids) and set(source_ids).isdisjoint(fresh_ids)
    assert not (set(fresh_ids)&excluded)
    center=source_X.mean(0);scale=np.maximum(source_X.std(0),1e-6)
    A=np.clip((source_X-center)/scale,-5,5)
    B=np.clip((np.vstack([old_X[cal_position],fresh_X])-center)/scale,-5,5)
    if A.shape[1]>16:
        projector=TruncatedSVD(n_components=16,random_state=0,n_iter=5).fit(A)
        A,B=projector.transform(A),projector.transform(B)
    c=A.mean(0);s=np.maximum(A.std(0),1e-6);A,B=(A-c)/s,(B-c)/s
    old_latent=parent.generator.latent(meta);width=A.shape[1]
    assert width<=old_latent['source_noise'].shape[1]
    fresh_rng=np.random.default_rng(970000+task_id)
    latent={'source_real':A,'target_real':B,'source_noise':old_latent['source_noise'][:,:width],
        'target_noise':np.vstack([old_latent['target_noise'][cal_position,:width],fresh_rng.normal(size=(len(fresh_ids),width))]),
        'source_z_noise':old_latent['source_z_noise'],'source_corruption_noise':old_latent['source_corruption_noise'],
        'target_z_noise':np.r_[old_latent['target_z_noise'][cal_position],fresh_rng.normal(size=len(fresh_ids))],
        'y_source':source_y,'y_target':np.r_[cal_y,fresh_y]}
    selected_config=parent.selection()['configuration']
    cases={rho:parent.generator.materialize(meta,selected_config,rho,latent) for rho in (0,20,50,80,100)}
    # Evaluation labels are deliberately in a separate artifact, not inputs.
    arrays={'y_source':source_y,'y_calibration':cal_y,'X_calibration':cases[0]['X_target'][:CALIBRATION_ROWS],
        'X_evaluation':cases[0]['X_target'][CALIBRATION_ROWS:],'source_ids':source_ids,
        'calibration_raw_ids':cal_ids,'evaluation_raw_ids':fresh_ids,'calibration_old_query_positions':cal_position}
    for rho,case in cases.items():arrays[f'X_source_{rho}']=case['X_source']
    # Independent nested calibration design: 32 adaptation +32 selection rows.
    class_counts=np.unique(cal_y,return_counts=True)[1]
    nested_stratified=bool(class_counts.min()>=2)
    if nested_stratified:
        adapt,val=train_test_split(np.arange(CALIBRATION_ROWS),train_size=32,stratify=cal_y,random_state=0)
    else:
        # A singleton cannot appear in both halves. Keep known class coverage
        # in adaptation, and do not discard any dataset or evaluation sample.
        split_rng=np.random.default_rng(0)
        mandatory=[int(split_rng.choice(np.flatnonzero(cal_y==label))) for label in np.unique(cal_y)]
        if len(mandatory)>32:
            # e.g.46-class Devanagari:32 adaptation rows cannot cover46
            # classes. Maximize coverage without removing a dataset/class.
            mandatory=split_rng.choice(mandatory,32,replace=False).tolist()
        rest=np.setdiff1d(np.arange(CALIBRATION_ROWS),mandatory)
        fill=split_rng.choice(rest,32-len(mandatory),replace=False)
        adapt=np.r_[mandatory,fill].astype(int);split_rng.shuffle(adapt)
        val=np.setdiff1d(np.arange(CALIBRATION_ROWS),adapt)
    arrays['adapt_positions']=adapt;arrays['validation_positions']=val
    # Stratified two-fold base predictions; singleton classes use KFold-like
    # splitting and source anchors in the context construction downstream.
    for name,positions in [('adapt',adapt),('all',np.arange(CALIBRATION_ROWS))]:
        labels=cal_y[positions];counts=np.bincount(labels,minlength=len(label_encoder.classes_))
        if np.min(counts[counts>0])>=2:
            iterator=StratifiedKFold(2,shuffle=True,random_state=0).split(positions,labels)
            splits=[(positions[a],positions[b]) for a,b in iterator]
        else:
            order=np.random.default_rng(0).permutation(positions);parts=np.array_split(order,2)
            splits=[(parts[1],parts[0]),(parts[0],parts[1])]
        for i,(train,test) in enumerate(splits):arrays[f'{name}_fit_{i}']=train;arrays[f'{name}_query_{i}']=test
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(out)
    truth=OUT/'evaluation_truth'/f'{task_id}.npz';truth.parent.mkdir(exist_ok=True)
    np.savez_compressed(truth,y_evaluation=fresh_y,raw_ids=fresh_ids)
    info=meta|{'status':'COMPLETE','calibration_rows':CALIBRATION_ROWS,'evaluation_rows':len(fresh_ids),
        'source_rows':256,'classes':len(label_encoder.classes_),'evaluation_classes':len(np.unique(fresh_y)),
        'calibration_from_original_query':True,'new_evaluation_disjoint_from_original_five_and_seven_query_rows':True,
        'evaluation_feature_duplicates_of_source_and_old_queries_removed':True,
        'freshness_scope':'not used in previous ABCD query cohorts; global project-wide historical use is not asserted',
        'real_preprocessing_and_projection_fit_on_source_only':True,'categorical_columns':categorical,
        'nested_split_stratified':nested_stratified,
        'synthetic_feature_generation_uses_labels':True,'original_paper_replication':False,
        'generator_parameters_unchanged':selected_config,'evaluation_labels_separate_from_model_inputs':True,
        'source_ids_sha256':protocol.array_sha256(source_ids),'calibration_ids_sha256':protocol.array_sha256(cal_ids),
        'evaluation_ids_sha256':protocol.array_sha256(fresh_ids),'model_inputs_sha256':parent.original.sha(out),
        'evaluation_truth_sha256':parent.original.sha(truth),'Markdown_written':False}
    atomic_json(out.with_suffix('.json'),info);return info


def main():
    from seven_shifts import registry
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'status':'FROZEN_BEFORE_DATA_SPLIT','seed':0,'calibration_rows':CALIBRATION_ROWS,
        'calibration_source':'old128 evaluation cohort; these labels explicitly become training/validation data',
        'new_evaluation_max_rows':MAX_EVALUATION_ROWS,'minimum_unique_evaluation_rows':16,
        'exclude_evaluation_overlap':['original256 source','old128 five-query','old seven-query where applicable','duplicate feature rows'],
        'real_preprocessing_source_only':True,'semisynthetic_label_conditional_generation':True,
        'evaluation_labels_for_hyperparameters_or_training':False,'Base_recomputed_on_same_new_evaluation_rows':True,
        'partial_test_supervision_must_be_marked':True,'B_D_calibration_only_clean_or_other_mechanisms':True,
        'source_context_cap':256,'base_plus_target_calibration_control_required':True,
        'code_sha256':parent.original.sha(__file__),'Markdown_written':False}
    if (OUT/'plan.json').exists():
        old=json.loads((OUT/'plan.json').read_text());accepted={old['code_sha256']}
        amendments=OUT/'implementation_amendments.json'
        if amendments.exists():accepted.update(x['code_sha256'] for x in json.loads(amendments.read_text()))
        assert parent.original.sha(__file__) in accepted
    else:atomic_json(OUT/'plan.json',plan)
    seven_ids={int(m['task_id']) for m in registry() if m['task']=='classification'}
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(prepare,m,seven_ids) for m in parent.generator.cohort()]
        for f in as_completed(futures):
            m=f.result();results.append(m);print('SPLIT',m['task_id'],m['calibration_rows'],m['evaluation_rows'],flush=True)
    pd.DataFrame([{k:m[k] for k in ['task_id','data_id','dataset','calibration_rows','evaluation_rows','classes','evaluation_classes']} for m in results]).to_csv(OUT/'cohort.csv',index=False)
    atomic_json(OUT/'completion.json',{'status':'COMPLETE','datasets':len(results),
        'calibration_rows_per_dataset':CALIBRATION_ROWS,'evaluation_rows_total':sum(m['evaluation_rows'] for m in results),
        'evaluation_rows_min':min(m['evaluation_rows'] for m in results),'evaluation_rows_max':max(m['evaluation_rows'] for m in results),
        'evaluation_labels_not_in_model_input_files':True,'Markdown_bytes':(parent.original.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size})


if __name__=='__main__':main()
