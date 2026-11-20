"""Retrospective case-control development task; never overwrite natural cohorts."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sqlite3
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import prepare_public_shift as prep

ROOT = prep.ROOT
prep.PLAN = json.loads((ROOT/'configs/public_casecontrol_protocol.json').read_text())
prep.OUT = ROOT/'data/processed/public_casecontrol_v1'


def summarize(frame, hospital, key, index, label):
    window = frame.loc[(frame.ICULOS > index-12) & (frame.ICULOS <= index)]
    assert len(window) and window.ICULOS.max() <= index
    row = dict(patient_id=hospital+'_'+Path(key).stem, label=label)
    exclude = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
    for c in frame.columns:
        if c in exclude:
            continue
        values = window[c].dropna()
        row.update({c+'_mean': values.mean(), c+'_min': values.min(),
                    c+'_max': values.max(), c+'_last': values.iloc[-1] if len(values) else np.nan,
                    c+'_observed': float(len(values)/len(window))})
    for c in exclude[:5]:
        values = window[c].dropna()
        row[c] = values.iloc[0] if len(values) else np.nan
    return row


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    base = ROOT/'data/raw/sepsis_public'
    selection = json.loads((base/'selection.json').read_text())
    db = sqlite3.connect(':memory:')
    db.deserialize((base/'records.sqlite').read_bytes())
    cases, controls, counts = {}, {}, {}
    for hospital in ['A', 'B']:
        cases[hospital], controls[hospital] = [], []
        counts[hospital] = dict(downloaded=0, eligible_cases=0, never_label_positive=0,
                                excluded_early_or_incomplete_cases=0)
        for number, key in enumerate(selection[hospital]['keys']):
            raw, digest = db.execute('SELECT raw,sha256 FROM records WHERE key=?', (key,)).fetchone()
            assert hashlib.sha256(raw).hexdigest() == digest
            frame = pd.read_csv(BytesIO(raw), sep='|')
            positive = frame.loc[frame.SepsisLabel == 1, 'ICULOS']
            counts[hospital]['downloaded'] += 1
            if len(positive):
                index = float(positive.min())
                if index >= 12 and frame.ICULOS.max() >= index+6 and frame.ICULOS.min() <= index-11:
                    cases[hospital].append((key, index, summarize(frame, hospital, key, index, 1)))
                else:
                    counts[hospital]['excluded_early_or_incomplete_cases'] += 1
            else:
                controls[hospital].append((key, float(frame.ICULOS.min()), float(frame.ICULOS.max())))
            if (number+1) % 2000 == 0:
                print('CASECONTROL_SCAN', hospital, number+1, flush=True)
        counts[hospital]['eligible_cases'] = len(cases[hospital])
        counts[hospital]['never_label_positive'] = len(controls[hospital])
        print('COHORT_COUNTS', hospital, counts[hospital], flush=True)
    assert all(len(cases[h]) >= 100 for h in ['A', 'B']), 'Insufficient cases; inspect cohort rather than force results.'
    for source, target in [('A', 'B'), ('B', 'A')]:
        # A common source-derived index-time distribution for both domains avoids
        # using the target event-time distribution to tune negative landmarks.
        n_cases = len(cases[source])
        train_ids, _ = train_test_split(np.arange(2*n_cases), test_size=.4,
                                        random_state=0, stratify=np.r_[np.ones(n_cases), np.zeros(n_cases)])
        landmarks = np.asarray([index for _, index, _ in cases[source]])[train_ids[train_ids < n_cases]]
        cohorts = {}
        for hospital in [source, target]:
            rng = np.random.default_rng(0 if hospital == 'A' else 1)
            table = [row for _, _, row in cases[hospital]]
            added = 0
            for position in rng.permutation(len(controls[hospital])):
                key, first, last = controls[hospital][position]
                eligible = landmarks[(landmarks-11 >= first) & (landmarks+6 <= last)]
                if not len(eligible):
                    continue
                index = float(rng.choice(eligible))
                raw = db.execute('SELECT raw FROM records WHERE key=?', (key,)).fetchone()[0]
                frame = pd.read_csv(BytesIO(raw), sep='|')
                table.append(summarize(frame, hospital, key, index, 0))
                added += 1
                if added == len(cases[hospital]):
                    break
            assert added == len(cases[hospital])
            cohorts[hospital] = pd.DataFrame(table)
        numeric = [c for c in cohorts[source].columns if c not in ['patient_id', 'label']]
        align = [c for c in numeric if not c.endswith('_observed') and c not in ['Gender', 'Unit1', 'Unit2']]
        prep.prepare_task('sepsis_'+source+'_to_'+target, cohorts[source], cohorts[target],
                          numeric, [], align,
                          'Exploratory retrospective 1:1 event-centered case-control task; '
                          'source-only calibration; prior natural-cohort results retained separately.',
                          balance_splits=True)
    db.close()
    prep.OUT.mkdir(parents=True, exist_ok=True)
    (prep.OUT/'cohort_counts.json').write_text(json.dumps(counts, indent=2))
    print('CASECONTROL_PREPARATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
