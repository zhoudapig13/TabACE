"""Fresh source-domain development patients and unused target development rows.

Final ID/OOD patients and the64 target fitting patients do not change.
"""
import argparse
import hashlib
from pathlib import Path
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from prepare_target_development import encoder,candidates,cohort
from prepare_public_shift import ROOT,limit,hash_array

OUT=ROOT/'data/processed/expanded_medical_development_v18'


def fitting_indices(y):
    context,cal=train_test_split(np.arange(len(y)),test_size=1/3,random_state=0,stratify=y)
    if len(context)>32:context,_=train_test_split(context,train_size=32,random_state=0,stratify=y[context])
    if len(cal)>32:cal,_=train_test_split(cal,train_size=32,random_state=0,stratify=y[cal])
    return context,cal


def save(task,parent,source_validation,encode,source_hashes):
    meta=json.loads((parent/'metadata.json').read_text())
    with np.load(parent/'inputs.npz') as z:pack={k:z[k] for k in z.files}
    context,cal=fitting_indices(pack['y_dev_fit'])
    unused=np.setdiff1d(np.arange(len(pack['y_dev_fit'])),np.r_[context,cal])
    pack.update(X_id_dev=encode(source_validation),y_id_dev=source_validation.label.to_numpy(np.int64),
                X_ood_dev=np.r_[pack['X_dev_val'],pack['X_dev_fit'][unused]],
                y_ood_dev=np.r_[pack['y_dev_val'],pack['y_dev_fit'][unused]])
    assert pack['y_id_dev'].sum()*2==len(pack['y_id_dev'])
    assert pack['y_ood_dev'].sum()*2==len(pack['y_ood_dev'])
    assert len(context)==len(cal)==32 and not set(context).intersection(cal)
    hashes=[hashlib.sha256(x.encode()).hexdigest() for x in source_validation.patient_id]
    assert len(hashes)==len(set(hashes)) and not set(hashes).intersection(source_hashes)
    dest=OUT/task;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'inputs.npz',**pack)
    np.savez_compressed(dest/'target_fitting_membership.npz',context=context,calibration=cal,
                        additional_validation_from_old_fit=unused)
    newmeta=dict(meta,parent_development_inputs=str(parent.relative_to(ROOT)),
        external_id_development=True,id_development_n=len(source_validation),
        ood_development_n=len(pack['y_ood_dev']),target_fitting_labels=64,
        source_development_disjoint_from_old_source_fit_and_tests=True,
        target_additional_validation_excludes_all64_fitting_rows=True,
        input_hashes={k:hash_array(v) for k,v in pack.items()})
    (dest/'metadata.json').write_text(json.dumps(newmeta,indent=2))
    (dest/'source_development_patient_hashes.json').write_text(json.dumps(hashes))
    print('EXPANDED_DEVELOPMENT_PREPARED',task,'ID',len(source_validation),'OOD',len(pack['y_ood_dev']),
          'target_fit_budget',64,flush=True)


def sepsis():
    base=ROOT/'data/raw/sepsis_public'
    old=json.loads((base/'selection.json').read_text());new=json.loads((base/'development_selection.json').read_text())
    db=sqlite3.connect(':memory:');db.deserialize((base/'records.sqlite').read_bytes())
    old_cases,old_controls=candidates(db,old);new_cases,new_controls=candidates(db,new)
    for source,target in [('A','B'),('B','A')]:
        n=len(old_cases[source])
        train,_=train_test_split(np.arange(2*n),test_size=.4,random_state=0,stratify=np.r_[np.ones(n),np.zeros(n)])
        landmarks=np.asarray([index for _,index,_ in old_cases[source]])[train[train<n]]
        source_frame=cohort(db,source,old_cases,old_controls,landmarks)
        groups=[train[source_frame.iloc[train].label.to_numpy()==c] for c in [0,1]]
        rng=np.random.default_rng(20);count=min(map(len,groups))
        train=np.sort(np.r_[rng.choice(groups[0],count,replace=False),rng.choice(groups[1],count,replace=False)])
        task='sepsis_'+source+'_to_'+target
        original=ROOT/'data/processed/public_casecontrol_v1'/task
        original_meta=json.loads((original/'metadata.json').read_text())
        encode=encoder(source_frame,original_meta,train)
        with np.load(original/'inputs.npz') as z:assert hash_array(encode(source_frame.iloc[train]))==hash_array(z['X_train'])
        source_dev=cohort(db,source,new_cases,new_controls,landmarks)
        old_members=json.loads((original/'split_membership_hashes.json').read_text())
        excluded=set(sum(old_members.values(),[]))
        save(task,ROOT/'data/processed/target_development_v2'/task,source_dev,encode,excluded)
    db.close()


def readmission():
    raw=pd.read_csv(ROOT/'data/raw/diabetes_readmission/diabetic_data.csv',na_values='?')
    raw=raw[~raw.discharge_disposition_id.isin([11,19,20,21])]
    raw=raw.sort_values('encounter_id').drop_duplicates('patient_nbr',keep='first').copy()
    raw['patient_id']='readmission_'+raw.patient_nbr.astype(str);raw['label']=(raw.readmitted=='<30').astype(int)
    task='readmission_other_to_ER';original=ROOT/'data/processed/public_shift_v1'/task
    meta=json.loads((original/'metadata.json').read_text())
    for c in meta['categorical_features']:raw[c]=raw[c].map(lambda x:str(x) if pd.notna(x) else np.nan)
    source=raw[raw.admission_source_id!=7].reset_index(drop=True)
    train,_=train_test_split(np.arange(len(source)),test_size=.4,random_state=0,stratify=source.label)
    train=limit(train,8192);encode=encoder(source,meta,train)
    with np.load(original/'inputs.npz') as z:assert hash_array(encode(source.iloc[train]))==hash_array(z['X_train'])
    member=json.loads((original/'split_membership_hashes.json').read_text());excluded=set(sum(member.values(),[]))
    eligible=source[~source.patient_id.map(lambda x:hashlib.sha256(x.encode()).hexdigest() in excluded)]
    groups=[eligible[eligible.label==c] for c in [0,1]];rng=np.random.default_rng(0)
    count=min(256,*[len(g) for g in groups]);assert count>=128
    source_dev=pd.concat([g.iloc[rng.choice(len(g),count,replace=False)] for g in groups]).reset_index(drop=True)
    save(task,ROOT/'data/processed/target_development_readmission_cc_v5'/task,source_dev,encode,excluded)


if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['sepsis','readmission'],required=True);args=p.parse_args()
    sepsis() if args.dataset=='sepsis' else readmission()
