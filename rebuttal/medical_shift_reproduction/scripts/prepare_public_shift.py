"""Patient-disjoint, source-only preprocessing for two open clinical datasets."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT / 'configs/public_shift_protocol.json').read_text())
OUT = ROOT / 'data/processed/public_shift_v1'


def hash_array(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


def limit(rows, count, seed=0):
    if len(rows) <= count:
        return np.asarray(rows)
    return np.sort(np.random.default_rng(seed).choice(rows, count, replace=False))


def prepare_task(name, source, target, numeric, categorical, align_names, description, balance_splits=False):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    assert source.patient_id.is_unique and target.patient_id.is_unique
    assert not set(source.patient_id).intersection(target.patient_id)
    all_source = np.arange(len(source))
    train, held = train_test_split(all_source, test_size=.4, random_state=0,
                                   stratify=source.label)
    cal, id_test = train_test_split(held, test_size=.5, random_state=0,
                                    stratify=source.iloc[held].label)
    train = limit(train, PLAN['max_source_training_pool'])
    cal = limit(cal, PLAN['max_source_calibration'], seed=1)
    id_test = limit(id_test, PLAN['max_evaluation_per_domain'], seed=2)
    ood_test = limit(np.arange(len(target)), PLAN['max_evaluation_per_domain'], seed=3)
    if balance_splits:
        def balanced(rows, frame, seed):
            rng = np.random.default_rng(seed)
            by_class = [rows[frame.iloc[rows].label.to_numpy() == label] for label in [0, 1]]
            count = min(map(len, by_class))
            assert count > 0
            return np.sort(np.concatenate([rng.choice(ids, count, replace=False) for ids in by_class]))
        train, cal, id_test = [balanced(rows, source, seed) for rows, seed in
                               [(train, 20), (cal, 21), (id_test, 22)]]
        ood_test = balanced(ood_test, target, 23)
    # Drop high-missing features using SOURCE TRAINING ONLY.
    numeric = [c for c in numeric if source.iloc[train][c].notna().mean() >= .05]
    categorical = [c for c in categorical if source.iloc[train][c].notna().mean() >= .05]
    transformer = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                                keep_empty_features=True),
                                  RobustScaler()), numeric),
        ('categorical', make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'),
                                      OneHotEncoder(handle_unknown='ignore', min_frequency=20,
                                                    max_categories=16, sparse_output=False,
                                                    dtype=np.float32)), categorical),
    ], sparse_threshold=0)
    transformer.fit(source.iloc[train])
    def encode(frame):
        return np.clip(transformer.transform(frame), -12, 12).astype(np.float32)
    X_train = encode(source.iloc[train])
    X_cal = encode(source.iloc[cal])
    X_id = encode(source.iloc[id_test])
    X_ood = encode(target.iloc[ood_test])
    # No labels from target are present in this input file.
    arrays = dict(X_train=X_train, y_train=source.iloc[train].label.to_numpy(np.int64),
                  X_cal=X_cal, y_cal=source.iloc[cal].label.to_numpy(np.int64),
                  X_id=X_id, X_ood=X_ood,
                  align_columns=np.asarray([i for i, c in enumerate(numeric) if c in align_names], int))
    assert all(np.isfinite(v).all() for k, v in arrays.items() if k.startswith('X'))
    np.savez_compressed(out / 'inputs.npz', **arrays)
    # Labels are segregated: evaluation script opens this only AFTER policies freeze.
    np.savez_compressed(out / 'test_truth.npz',
                        y_id=source.iloc[id_test].label.to_numpy(np.int64),
                        y_ood=target.iloc[ood_test].label.to_numpy(np.int64))
    membership = {
        'train': source.iloc[train].patient_id.tolist(),
        'cal': source.iloc[cal].patient_id.tolist(),
        'id_test': source.iloc[id_test].patient_id.tolist(),
        'ood_test': target.iloc[ood_test].patient_id.tolist(),
    }
    for a in membership:
        for b in membership:
            if a != b:
                assert not set(membership[a]).intersection(membership[b]), (a, b)
    hashes = {k: hash_array(v) for k, v in arrays.items()}
    meta = dict(task=name, description=description, status='PREPARED_BEFORE_INFERENCE', seed=0,
                cohort_source=len(source), cohort_target=len(target),
                split_counts={k: len(v) for k, v in membership.items()},
                source_train_positives=int(arrays['y_train'].sum()),
                source_calibration_positives=int(arrays['y_cal'].sum()),
                encoded_features=X_train.shape[1], numeric_features=numeric,
                categorical_features=categorical, input_hashes=hashes,
                patient_disjoint=True, casecontrol_balanced_splits=balance_splits, preprocessor_fit_on_source_training_only=True,
                target_labels_in_input_file=False,
                domain_rule_from_tableshift=name=='readmission_other_to_ER',
                bit_exact_official_pipeline=False)
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    # Only hashed IDs are stored for local split auditing.
    (out / 'split_membership_hashes.json').write_text(json.dumps({
        k: [hashlib.sha256(str(x).encode()).hexdigest() for x in v]
        for k, v in membership.items()}))
    print('PREPARED', name, meta['split_counts'], 'features', X_train.shape[1],
          'source_train_positive', meta['source_train_positives'], flush=True)


def readmission():
    raw = pd.read_csv(ROOT / 'data/raw/diabetes_readmission/diabetic_data.csv', na_values='?')
    raw = raw[~raw.discharge_disposition_id.isin([11, 19, 20, 21])]
    raw = raw.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='first').copy()
    raw['patient_id'] = 'readmission_' + raw.patient_nbr.astype(str)
    raw['label'] = (raw.readmitted == '<30').astype(int)
    numeric = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
               'num_medications', 'number_outpatient', 'number_emergency',
               'number_inpatient', 'number_diagnoses']
    excluded = {'patient_id', 'label', 'patient_nbr', 'encounter_id',
                'readmitted', 'admission_source_id'}
    categorical = [c for c in raw.columns if c not in excluded and c not in numeric]
    for c in categorical:
        raw[c] = raw[c].map(lambda x: str(x) if pd.notna(x) else np.nan)
    source = raw[raw.admission_source_id != 7].reset_index(drop=True)
    target = raw[raw.admission_source_id == 7].reset_index(drop=True)
    prepare_task('readmission_other_to_ER', source, target, numeric, categorical, numeric,
                 '30-day readmission at discharge; TableShift admission-source split, '
                 'with additional one-eligible-encounter-per-patient restriction.')


def sepsis():
    base = ROOT / 'data/raw/sepsis_public'
    assert (base / 'selection.json').exists(), 'Sepsis download is not complete.'
    selection = json.loads((base / 'selection.json').read_text())
    # Sequentially read the cache once. Remote filesystem random page reads are
    # expensive; deserializing into RAM changes no records or cohort rules.
    db = sqlite3.connect(':memory:')
    db.deserialize((base / 'records.sqlite').read_bytes())
    cohorts = {}
    for hospital in ['A', 'B']:
        table, counts = [], {'downloaded': 0, 'short_stay': 0, 'onset_by12': 0,
                             'no_observation_by12': 0}
        for key in selection[hospital]['keys']:
            found = db.execute('SELECT raw, sha256 FROM records WHERE key=?', (key,)).fetchone()
            assert found is not None
            raw, digest = found
            assert hashlib.sha256(raw).hexdigest() == digest
            frame = pd.read_csv(BytesIO(raw), sep='|')
            counts['downloaded'] += 1
            if frame.ICULOS.max() < 18:
                counts['short_stay'] += 1
                continue
            positive = frame.loc[frame.SepsisLabel == 1, 'ICULOS']
            onset = float(positive.min() + 6) if len(positive) else np.inf
            if onset <= 12:
                counts['onset_by12'] += 1
                continue
            observed = frame.loc[frame.ICULOS <= 12]
            if not len(observed):
                counts['no_observation_by12'] += 1
                continue
            assert observed.ICULOS.max() <= 12
            row = {'patient_id': hospital + '_' + Path(key).stem,
                   'label': int(12 < onset <= 18)}
            variables = [c for c in frame.columns if c not in
                         ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']]
            for c in variables:
                values = observed[c].dropna()
                row.update({c+'_mean': values.mean(), c+'_min': values.min(),
                            c+'_max': values.max(), c+'_last': values.iloc[-1] if len(values) else np.nan,
                            c+'_observed': float(len(values)/len(observed))})
            for c in ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']:
                values = observed[c].dropna()
                row[c] = values.iloc[0] if len(values) else np.nan
            table.append(row)
            if counts['downloaded'] % 1000 == 0:
                print('SEPSIS_PREP_PROGRESS', hospital, counts['downloaded'], flush=True)
        cohorts[hospital] = pd.DataFrame(table)
        print('SEPSIS_COHORT', hospital, len(table), counts, flush=True)
    db.close()
    for source, target in [('A', 'B'), ('B', 'A')]:
        numeric = [c for c in cohorts[source].columns if c not in ['patient_id', 'label']]
        align = [c for c in numeric if not c.endswith('_observed') and c not in ['Gender', 'Unit1', 'Unit2']]
        prepare_task('sepsis_'+source+'_to_'+target, cohorts[source], cohorts[target],
                     numeric, [], align,
                     'Fixed ICU hour12 landmark, subsequent 6h clinical sepsis; '
                     'official early label +6 reconstructs clinical onset; one row per patient.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['readmission', 'sepsis', 'both'], default='both')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    if args.dataset in ['readmission', 'both']:
        readmission()
    if args.dataset in ['sepsis', 'both']:
        sepsis()
