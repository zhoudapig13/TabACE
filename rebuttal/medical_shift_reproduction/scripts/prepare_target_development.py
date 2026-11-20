"""Prepare additional labeled development patients, excluding every previous test patient."""
import argparse
from io import BytesIO
import hashlib
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from prepare_public_shift import ROOT, hash_array, limit
from prepare_casecontrol_shift import summarize

OUT = ROOT/'data/processed/target_development_v2'


def encoder(source, meta, training):
    numeric, categorical = meta['numeric_features'], meta['categorical_features']
    transform = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                  RobustScaler()), numeric),
        ('categorical', make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'),
                                      OneHotEncoder(handle_unknown='ignore', min_frequency=20,
                                                    max_categories=16, sparse_output=False, dtype=np.float32)), categorical),
    ], sparse_threshold=0)
    transform.fit(source.iloc[training])
    return lambda frame: np.clip(transform.transform(frame), -12, 12).astype(np.float32)


def save(task, original, source_frame, train, development, casecontrol):
    meta = json.loads((original/'metadata.json').read_text())
    with np.load(original/'inputs.npz') as z:
        pack = {k: z[k] for k in z.files}
    encode = encoder(source_frame, meta, train)
    reconstructed = encode(source_frame.iloc[train])
    assert hash_array(reconstructed) == hash_array(pack['X_train']), 'Original preprocessing not reproduced.'
    old_membership = json.loads((original/'split_membership_hashes.json').read_text())
    old_tests = set(old_membership['id_test']+old_membership['ood_test'])
    hashes = [hashlib.sha256(x.encode()).hexdigest() for x in development.patient_id]
    assert len(hashes) == len(set(hashes)) and not set(hashes).intersection(old_tests)
    fit, val = train_test_split(np.arange(len(development)), test_size=.4, random_state=0,
                                stratify=development.label)
    if casecontrol:
        def balance(rows, seed):
            rng = np.random.default_rng(seed)
            ids = [rows[development.iloc[rows].label.to_numpy() == label] for label in [0, 1]]
            n = min(map(len, ids))
            return np.sort(np.concatenate([rng.choice(x, n, replace=False) for x in ids]))
        fit, val = balance(fit, 0), balance(val, 1)
    arrays = dict(pack, X_dev_fit=encode(development.iloc[fit]),
                  y_dev_fit=development.iloc[fit].label.to_numpy(np.int64),
                  X_dev_val=encode(development.iloc[val]),
                  y_dev_val=development.iloc[val].label.to_numpy(np.int64))
    dest = OUT/task
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'inputs.npz', **arrays)
    (dest/'metadata.json').write_text(json.dumps(dict(
        task=task, original_inputs=str(original.relative_to(ROOT)), target_development_extra_supervision=True,
        source_preprocessor_exact_hash_verified=True, patient_disjoint_from_original_tests=True,
        casecontrol=casecontrol, fit_n=len(fit), fit_positive=int(arrays['y_dev_fit'].sum()),
        validation_n=len(val), validation_positive=int(arrays['y_dev_val'].sum()),
        input_hashes={k: hash_array(v) for k, v in arrays.items()}), indent=2))
    (dest/'development_membership_hashes.json').write_text(json.dumps(
        {'fit': [hashes[i] for i in fit], 'validation': [hashes[i] for i in val]}))
    print('DEVELOPMENT_PREPARED',task,'fit',len(fit),'validation',len(val),
          'fit_positive',int(arrays['y_dev_fit'].sum()),'validation_positive',int(arrays['y_dev_val'].sum()),flush=True)


def readmission():
    raw = pd.read_csv(ROOT/'data/raw/diabetes_readmission/diabetic_data.csv', na_values='?')
    raw = raw[~raw.discharge_disposition_id.isin([11, 19, 20, 21])]
    raw = raw.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first').copy()
    raw['patient_id'] = 'readmission_'+raw.patient_nbr.astype(str)
    raw['label'] = (raw.readmitted == '<30').astype(int)
    original = ROOT/'data/processed/public_shift_v1/readmission_other_to_ER'
    meta = json.loads((original/'metadata.json').read_text())
    for c in meta['categorical_features']:
        raw[c] = raw[c].map(lambda x: str(x) if pd.notna(x) else np.nan)
    source = raw[raw.admission_source_id != 7].reset_index(drop=True)
    target = raw[raw.admission_source_id == 7].reset_index(drop=True)
    train, _ = train_test_split(np.arange(len(source)), test_size=.4, random_state=0, stratify=source.label)
    train = limit(train, 8192)
    excluded = set(json.loads((original/'split_membership_hashes.json').read_text())['ood_test'])
    eligible = target[~target.patient_id.map(lambda x: hashlib.sha256(x.encode()).hexdigest() in excluded)]
    ids = np.random.default_rng(0).choice(len(eligible), min(4096,len(eligible)), replace=False)
    development = eligible.iloc[np.sort(ids)].reset_index(drop=True)
    save('readmission_other_to_ER', original, source, train, development, False)


def candidates(db, selection):
    cases, controls = {}, {}
    for hospital in ['A','B']:
        cases[hospital], controls[hospital] = [], []
        for key in selection[hospital]['keys']:
            raw, digest = db.execute('SELECT raw,sha256 FROM records WHERE key=?',(key,)).fetchone()
            assert hashlib.sha256(raw).hexdigest() == digest
            frame = pd.read_csv(BytesIO(raw),sep='|')
            pos = frame.loc[frame.SepsisLabel == 1,'ICULOS']
            if len(pos):
                index = float(pos.min())
                if index >= 12 and frame.ICULOS.max() >= index+6 and frame.ICULOS.min() <= index-11:
                    cases[hospital].append((key,index,summarize(frame,hospital,key,index,1)))
            else:
                controls[hospital].append((key,float(frame.ICULOS.min()),float(frame.ICULOS.max())))
        print('SEPSIS_DEVELOPMENT_CANDIDATES',hospital,'cases',len(cases[hospital]),flush=True)
    return cases,controls


def cohort(db,hospital,cases,controls,landmarks):
    table = [row for _,_,row in cases[hospital]]
    rng = np.random.default_rng(0 if hospital=='A' else 1)
    added=0
    for i in rng.permutation(len(controls[hospital])):
        key,first,last=controls[hospital][i]
        eligible=landmarks[(landmarks-11 >= first)&(landmarks+6 <= last)]
        if not len(eligible):continue
        index=float(rng.choice(eligible))
        frame=pd.read_csv(BytesIO(db.execute('SELECT raw FROM records WHERE key=?',(key,)).fetchone()[0]),sep='|')
        table.append(summarize(frame,hospital,key,index,0));added+=1
        if added==len(cases[hospital]):break
    assert added==len(cases[hospital])
    return pd.DataFrame(table)


def sepsis():
    base=ROOT/'data/raw/sepsis_public'
    old=json.loads((base/'selection.json').read_text())
    new=json.loads((base/'development_selection.json').read_text())
    db=sqlite3.connect(':memory:');db.deserialize((base/'records.sqlite').read_bytes())
    old_cases,old_controls=candidates(db,old)
    new_cases,new_controls=candidates(db,new)
    for source,target in [('A','B'),('B','A')]:
        n=len(old_cases[source])
        train,_=train_test_split(np.arange(2*n),test_size=.4,random_state=0,stratify=np.r_[np.ones(n),np.zeros(n)])
        landmarks=np.asarray([index for _,index,_ in old_cases[source]])[train[train<n]]
        source_frame=cohort(db,source,old_cases,old_controls,landmarks)
        # Reproduce balanced source training selection from the previous experiment.
        rng=np.random.default_rng(20)
        by_class=[train[source_frame.iloc[train].label.to_numpy()==label] for label in [0,1]]
        count=min(map(len,by_class))
        train=np.sort(np.concatenate([rng.choice(ids,count,replace=False) for ids in by_class]))
        development=cohort(db,target,new_cases,new_controls,landmarks)
        task='sepsis_'+source+'_to_'+target
        save(task,ROOT/'data/processed/public_casecontrol_v1'/task,source_frame,train,development,True)
    db.close()


if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['readmission','sepsis'],required=True)
    args=parser.parse_args()
    readmission() if args.dataset=='readmission' else sepsis()
