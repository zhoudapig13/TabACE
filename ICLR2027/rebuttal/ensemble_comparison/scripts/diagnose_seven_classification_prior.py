"""One predeclared split diagnostic; never tune severity to a paper score.

The historical seven generator is missing. Preserving the original class prior
is a reconstruction hypothesis, not a claim of bit-exact reproduction. Existing
Clean/five results are read-only; this script writes an isolated seven branch.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from prepare_data import ROOT, RUN, LOCAL_CC18, atomic_json, protocol

TRACK = 'seven_prior_preserving_v1'
DEST = RUN / 'results' / TRACK
PLAN = RUN / 'configs' / 'seven_classification_prior_diagnostic.json'
FREEZE = RUN / 'configs' / 'tabicl_clean_five_frozen.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_files():
    paths = set()
    for root in [RUN/'cache', RUN/'results/tabicl']:
        for track in root.glob('five*'):
            paths.update(p for p in track.rglob('*') if p.is_file())
    # Also protect previously reported Clean cells on the old seven track.
    for p in (RUN/'results/tabicl/seven_reimplemented').rglob('covariate_0.*'):
        if p.is_file(): paths.add(p)
    for name in ['classification_adaptation_summary.json',
                 'random_dense_current_summary.json',
                 'regression_expansion_latest_summary.json',
                 'final_ensemble_summary.json']:
        paths.add(RUN/'results'/name)
    return {str(p.relative_to(RUN)): sha(p) for p in sorted(paths)}


def natural_clean(meta):
    folder = LOCAL_CC18 / str(meta['task_id'])
    frame = pd.read_parquet(folder/'X.parquet')
    labels = pd.read_parquet(folder/'y.parquet').iloc[:, 0]
    valid = ~labels.isna().to_numpy()
    X, _ = protocol._numeric_matrix(frame.loc[valid])
    y = LabelEncoder().fit_transform(labels.loc[valid])
    ids = np.arange(len(y))
    if len(ids) > 448:
        ids, _ = train_test_split(ids, train_size=448, stratify=y, random_state=0)
    source, target = train_test_split(ids, train_size=320, test_size=128,
                                      stratify=y[ids], random_state=0)
    history, reference = train_test_split(source, train_size=256, test_size=64,
                                          stratify=y[source], random_state=0)
    assert not (set(source) & set(target))
    arrays = (X[history], y[history], X[reference], y[reference], X[target], y[target])
    return arrays, {'raw_class_prior': float(y.mean()),
                    'target_class_prior': float(y[target].mean()),
                    'history_rows': history.tolist(), 'reference_rows': reference.tolist(),
                    'target_rows': target.tolist()}


def prepare():
    from seven_shifts import registry, classification_clean, FAMILIES, PARAMETERS
    if not FREEZE.exists():
        atomic_json(FREEZE, {'purpose': 'Freeze existing TabICL Clean/five artifacts',
                            'files_sha256': freeze_files(), 'Markdown_written': False})
    audit = []
    original = ROOT/'PaperResults/failure_recovery_20260913/tabicl/seven_point_lock/composite/classification/detail_oof.csv'
    old = list(csv.DictReader(original.open()))
    prior_evidence = [r for r in old if abs(float(r['base_balanced_accuracy'])-.5)<1e-12
                      and float(r['base_accuracy'])>.6]
    items = [m for m in registry() if m['task']=='classification']
    for m in items:
        new, rows = natural_clean(m)
        existing = classification_clean(m)
        audit.append({'task_id': m['task_id'], 'dataset': m['dataset'],
                      'raw_class_prior': rows['raw_class_prior'],
                      'balanced_target_prior': float(existing[-1].mean()),
                      'prior_preserving_target_prior': rows['target_class_prior']})
        for family, rho in [('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]:
            out=RUN/'cache'/TRACK/'classification'/str(m['task_id'])/f'{family}_{rho}.npz'
            if out.exists() and out.with_suffix('.json').exists(): continue
            from seven_shifts import materialize
            data, meta = materialize(m, family, rho, new)
            meta.update(track=TRACK, n_classes=2, original_paper_generator_recovered=False,
                        change='prior-preserving stratified cap instead of class-balanced cap',
                        severity_unchanged=True, condition_materialized=True)
            out.parent.mkdir(parents=True,exist_ok=True)
            tmp=out.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(out)
            atomic_json(out.with_suffix('.json'), meta)
        atomic_json(DEST/'splits'/f"{m['task_id']}.json", rows)
    plan={'status':'FROZEN_DIAGNOSTIC', 'track':TRACK, 'seed':0, 'datasets':35,
          'conditions_per_dataset':29, 'single_changed_factor':'class-prior preservation in row cap',
          'all_shift_parameters_unchanged':PARAMETERS, 'strengths':[.2,.5,.8,1.],
          'paper_score_used_for_selection':False, 'severity_search':False,
          'original_generator_recovered':False, 'Shared_run':False,
          'existing_Clean_and_five_results_overwritten':False,
          'new_seven_Clean_is_diagnostic_only':True, 'source_code_sha256':sha(Path(__file__)),
          'prior_diagnostics':audit, 'historical_prior_evidence_count':len(prior_evidence),
          'historical_prior_example':prior_evidence[0] if prior_evidence else None,
          'historical_point_lock_is_not_unbiased_policy_evaluation':True}
    atomic_json(PLAN,plan)
    print(json.dumps({'prepared':35,'balanced_target_prior_mean':float(np.mean([x['balanced_target_prior'] for x in audit])),
                      'natural_target_prior_mean':float(np.mean([x['prior_preserving_target_prior'] for x in audit])),
                      'freeze_files':len(json.loads(FREEZE.read_text())['files_sha256'])},ensure_ascii=False))


def run(shard, shards, limit):
    # Import before any helper that inserts ROOT/src: use the same installed
    # TabICL package as the original run_seven entry point.
    from run_tabicl import Predictor, metrics
    import tabicl
    import torch
    from seven_shifts import registry, FAMILIES
    torch.set_num_threads(1);warnings.filterwarnings('ignore',category=FutureWarning)
    plan=json.loads(PLAN.read_text());assert plan['status']=='FROZEN_DIAGNOSTIC'
    predictor=Predictor('classification','cuda:0')
    jobs=[(m,f,r) for m in registry() if m['task']=='classification'
          for f,r in [('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]]
    jobs=jobs[shard::shards]
    if limit: jobs=jobs[:limit]
    for m,family,rho in jobs:
        out=DEST/'base'/'classification'/str(m['task_id'])/f'{family}_{rho}.npz'
        if out.exists() and out.with_suffix('.json').exists():continue
        source=RUN/'cache'/TRACK/'classification'/str(m['task_id'])/out.name
        with np.load(source) as z:data={k:z[k] for k in z.files}
        prediction,_=predictor(data['X_source'],data['y_source'],data['X_target'],n_classes=2)
        values=metrics(data['y_target'],prediction,'classification',1.)
        out.parent.mkdir(parents=True,exist_ok=True)
        tmp=out.with_suffix('.tmp.npz');np.savez_compressed(tmp,prediction=prediction,y_target=data['y_target']);tmp.replace(out)
        atomic_json(out.with_suffix('.json'),{'task_id':m['task_id'],'dataset':m['dataset'],
             'family':family,'rho':rho,'track':TRACK,'seed':0,'metrics':values,
             'status':'COMPLETE','source_file_sha256':sha(source),'TabICL_library':tabicl.__file__})
        print(json.dumps({'task_id':m['task_id'],'family':family,'rho':rho,**values}),flush=True)
    print('WORKER_FINISHED',shard,flush=True)


def report():
    from seven_shifts import registry
    oldroot=RUN/'results/tabicl/seven_reimplemented/classification'
    old=[];new=[];clean=[];done=0
    for m in registry():
        if m['task']!='classification':continue
        rows=list((DEST/'base/classification'/str(m['task_id'])).glob('*.json'))
        if len(rows)!=29:continue
        done+=1
        for path in rows:
            row=json.loads(path.read_text());oldrow=json.loads((oldroot/str(m['task_id'])/path.name).read_text())
            if row['rho']==0:clean.append(row['metrics']);continue
            old.append(oldrow['methods']['base_protocol_check']);new.append(row['metrics'])
    result={'complete_datasets':done,'expected_datasets':35,'complete':done==35,
            'current_run_cases':len(list((DEST/'base').rglob('*.json'))),
            'old_Base_same_completed_cohort':{k:float(np.mean([x[k] for x in old])) for k in ['accuracy','auc']} if old else None,
            'prior_preserving_Base':{k:float(np.mean([x[k] for x in new])) for k in ['accuracy','auc']} if new else None,
            'new_seven_Clean_diagnostic_only':{k:float(np.mean([x[k] for x in clean])) for k in ['accuracy','auc']} if clean else None,
            'existing_results_overwritten':False,'original_protocol_restoration_proven':False}
    if done:
        fams=defaultdict(list)
        for path in (DEST/'base/classification').rglob('*.json'):
            row=json.loads(path.read_text())
            if row['rho']:fams[row['family']].append(row['metrics'])
        result['family_means_all_completed_cases']={f:{k:float(np.mean([x[k] for x in rows])) for k in ['accuracy','auc']} for f,rows in fams.items()}
    frozen=json.loads(FREEZE.read_text())['files_sha256']
    changed=[name for name,expected in frozen.items() if sha(RUN/name)!=expected]
    assert not changed, changed
    result['frozen_files_verified_unchanged']=len(frozen)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(DEST/'latest_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--report',action='store_true');parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1);parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.report:report()
    else:run(args.shard,args.shards,args.limit)
