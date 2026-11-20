"""Fresh emergency-room adaptation/validation patients; old evaluation unchanged."""
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from prepare_target_development import encoder
from prepare_public_shift import limit
from run_public_shift import ROOT,array_hash,write_json

NAME='readmission_other_to_ER_ood'
DEST=ROOT/'data/processed/readmission_fresh_development_v32'/NAME


def main():
    raw=pd.read_csv(ROOT/'data/raw/diabetes_readmission/diabetic_data.csv',na_values='?',low_memory=False)
    raw=raw[~raw.discharge_disposition_id.isin([11,19,20,21])]
    raw=raw.sort_values('encounter_id').drop_duplicates('patient_nbr',keep='first').copy()
    raw['patient_id']='readmission_'+raw.patient_nbr.astype(str);raw['label']=(raw.readmitted=='<30').astype(int)
    original=ROOT/'data/processed/public_shift_v1/readmission_other_to_ER'
    meta=json.loads((original/'metadata.json').read_text())
    target_mask=raw.admission_source_id==7
    for c in meta['categorical_features']:raw[c]=raw[c].map(lambda x:str(x) if pd.notna(x) else np.nan)
    source=raw[~target_mask].reset_index(drop=True);target=raw[target_mask].copy()
    train,_=train_test_split(np.arange(len(source)),test_size=.4,random_state=0,stratify=source.label)
    train=limit(train,8192);encode=encoder(source,meta,train)
    with np.load(original/'inputs.npz') as z:assert array_hash(encode(source.iloc[train]))==array_hash(z['X_train'])
    excluded=set()
    for f in [original/'split_membership_hashes.json',ROOT/'data/processed/target_development_v2/readmission_other_to_ER/development_membership_hashes.json']:
        for values in json.loads(f.read_text()).values():excluded.update(values)
    target['hash']=target.patient_id.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    eligible=target[~target['hash'].isin(excluded)];rng=np.random.default_rng(0)
    cohort=pd.concat([g.iloc[rng.choice(len(g),416,replace=False)] for _,g in eligible.groupby('label')]).reset_index(drop=True)
    fit,rest=train_test_split(np.arange(len(cohort)),train_size=256,stratify=cohort.label,random_state=0)
    cal,val=train_test_split(rest,train_size=64,stratify=cohort.iloc[rest].label,random_state=0)
    fit,cal,val=map(np.sort,[fit,cal,val]);assert len(val)==512
    groups={name:cohort.iloc[i] for name,i in [('fit',fit),('calibration',cal),('validation',val)]}
    hashes={name:g['hash'].tolist() for name,g in groups.items()}
    assert len(set(sum(hashes.values(),[])))==832 and not set(sum(hashes.values(),[])).intersection(excluded)
    old=ROOT/'data/processed/medical_test_assisted_v26'/NAME
    oldmeta=json.loads((old/'metadata.json').read_text())
    with np.load(old/'inputs.npz') as z:p={k:z[k] for k in z.files}
    p.update(X_target_train=np.r_[p['X_target_train'],encode(groups['fit'])],
        y_target_train=np.r_[p['y_target_train'],groups['fit'].label.to_numpy(np.int64)],
        X_target_cal=np.r_[p['X_target_cal'],encode(groups['calibration'])],
        y_target_cal=np.r_[p['y_target_cal'],groups['calibration'].label.to_numpy(np.int64)],
        X_development=encode(groups['validation']),y_development=groups['validation'].label.to_numpy(np.int64))
    DEST.mkdir(parents=True,exist_ok=True)
    newmeta=dict(oldmeta,input_hashes={k:array_hash(v) for k,v in p.items()},
        fresh_patient_counts={k:len(v) for k,v in hashes.items()},fresh_patients_disjoint_from_all_prior_patients=True,
        fresh_validation_not_used_for_any_earlier_selection=True,total_target_fitting_labels=416,
        target_context_pool=304,target_calibration_pool=112,development_n=512,
        frozen_source_preprocessor_hash_verified=True,prior_data=str(old.relative_to(ROOT)),
        evaluation_truth_file=str((old/'evaluation_truth.npz').relative_to(ROOT)),
        prepared_at_utc=datetime.now(timezone.utc).isoformat())
    if (DEST/'inputs.npz').exists():
        assert json.loads((DEST/'metadata.json').read_text())['input_hashes']==newmeta['input_hashes']
    else:np.savez_compressed(DEST/'inputs.npz',**p)
    write_json(DEST/'metadata.json',newmeta);write_json(DEST/'fresh_patient_hashes.json',hashes)
    print('FRESH_READMISSION',newmeta['fresh_patient_counts'],'unchanged evaluation',len(p['X_evaluation']),flush=True)


if __name__=='__main__':main()
