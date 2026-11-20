"""Mechanism holdout diagnostic; reconstructed source protocols, not replication.

Both tables retain the real paired Base. Classification uses the manuscript's
five Shared slots. Regression explicitly tests Shared+smooth-repair (dagger):
the manuscript's five Shared slots do not contain a smooth response repair.
No attempt is made to force significance or to substitute historical means.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
import torch

from prepare_data import ROOT, RUN, atomic_json, protocol
from run_tabicl import Predictor, metrics, preserving_bootstrap
from shared_reference import load_condition
from fit_combiners import fold_mapping
import tabace_fusion_v28 as fusion

OUT = RUN / 'results/abcd_mechanism_v84'
TRACKS = {'classification': ('five_corrected', 'seven_reimplemented'),
          'regression': ('seven_reimplemented',)}
TARGETS = {'classification': ('five_corrected', 'shortcut'),
           'regression': ('seven_reimplemented', 'concept')}
REGS = (.0001, .001, .01)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(task):
    for track in TRACKS[task]:
        for path in sorted((RUN / 'results/reference/tabicl' / track / task).glob('*/*.npz')):
            if path.stem.endswith('_0'):
                # Historical five cache has one clean condition; seven has
                # one clean per family. Deduplicate the latter before fitting.
                if track.startswith('seven') and path.stem != 'covariate_0':
                    continue
            yield path


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / 'plan.json'
    if dest.exists():
        accepted = {json.loads(dest.read_text())['code_sha256']}
        amendments = OUT / 'implementation_amendments.json'
        if amendments.exists():
            accepted.update(x['code_sha256'] for x in json.loads(amendments.read_text()))
        assert sha(__file__) in accepted, 'implementation changed without a recorded amendment'
        return
    refs = {}
    for task, path, family_key, family in [
        ('classification', ROOT / 'fair_budget_experiments/unified_context_ensemble/results/tabicl_origin_safe_final_v2/five_shift/classification/summary_oof.csv', 'family', 'shortcut'),
        ('regression', ROOT / 'PaperResults/failure_recovery_20260913/tabicl/seven_point_lock/composite/regression/summary_oof.csv', 'shift', 'concept')]:
        frame = pd.read_csv(path)
        group = frame[frame[family_key] == family]
        severity = 'rho' if task == 'classification' else 'strength'
        refs[task] = {'path': str(path), 'sha256': sha(path), 'rows': group.to_dict('records'),
                      'positive_strength_mean': group[group[severity] > 0].select_dtypes('number').mean().to_dict()}
    manifest = []
    for task in TRACKS:
        for path in source_paths(task):
            manifest.append({'task': task, 'path': str(path), 'sha256': sha(path),
                             'metadata_sha256': sha(path.with_suffix('.json'))})
    atomic_json(OUT / 'input_manifest.json', manifest)
    atomic_json(dest, {
        'status': 'FROZEN_BEFORE_NEW_BRANCH_INFERENCE_AND_ABCD_FITTING',
        'seed': 0, 'context_cap': 256, 'estimators_per_branch': 1, 'alternative_slots': 5,
        'paper_reference': refs, 'Base': 'actual cached paired predictions; never historical mean substitution',
        'Base_replay': 'upstream cache reused, not a new full backbone replay',
        'benchmark': 'appendix-based source-side reconstruction; historical materializer missing',
        'reference_base_alignment_achieved': False, 'original_paper_replication': False,
        'upstream_development_feedback': True, 'independent_confirmation_claim': False,
        'classification_matching_constructor': 'observed-source strongest-correlation feature removal, slot 3',
        'classification_replacement': 'class-preserving uniform bootstrap of observed current source, slot 3',
        'regression_is_adapted_dagger': True,
        'regression_matching_constructor': 'replace uniform-bootstrap slot 0 with strongest observed tanh-component removal; source-only; four response strata',
        'regression_without_matching_constructor': 'original manuscript Shared five slots',
        'ABCD': {'A': [True, True], 'B': [False, True], 'C': [True, False], 'D': [False, False]},
        'unseen_classification_excludes': 'all five source-perturbation families, including Mixed; all alter the leading feature; retain clean and seven-shift calibration',
        'unseen_regression_excludes': ['concept', 'temporal'],
        'calibration': 'five outer dataset folds; labels of other datasets only; one inner held-out dataset fold selects regularization; policies family-blind at inference',
        'mechanism_exclusion_applies_to': 'all calibration and inner-selection episodes in B/D',
        'regularization_grid': REGS, 'raw_gate_gain': 1,
        'backbone_finetuning': False, 'clean_unperturbed_labels_used_for_repair': False,
        'extra_supervised_prediction_head': False, 'query_labels_received_by_constructor': False,
        'primary_metrics': {'classification': 'auc', 'regression': 'r2'},
        'averaging': 'average four positive strengths inside dataset; then equal-weight dataset mean',
        'statistics': 'paired dataset bootstrap 95% CI and two-sided paired sign-flip test; Holm over eight A-D primary comparisons',
        'guaranteed_significance': False, 'win_tie_loss_epsilon': 1e-6,
        'code_sha256': sha(__file__), 'manifest_sha256': sha(OUT / 'input_manifest.json'),
        'Markdown_written': False})
    print('PLAN_FROZEN', len(manifest), flush=True)


def repaired_context(X, y):
    """Observed source-only repair; neither generator coordinates nor clean y."""
    median = np.median(X, axis=0)
    scale = np.maximum(np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0), 1e-6)
    components = np.tanh(np.clip((X - median) / scale, -20, 20))
    components -= components.mean(axis=0)
    norm = np.sum(components**2, axis=0)
    cov = components.T @ (y - y.mean())
    utility = cov**2 / np.maximum(norm, 1e-12)
    j = int(np.argmax(utility))
    slope = float(cov[j] / max(norm[j], 1e-12))
    repaired = y - slope * components[:, j]
    rng = np.random.default_rng(0)
    strata = [p for p in np.array_split(np.argsort(repaired, kind='stable'), 4) if len(p)]
    counts = np.full(len(strata), len(y) // len(strata)); counts[:len(y) % len(strata)] += 1
    rows = np.concatenate([rng.choice(s, int(n), replace=True) for s, n in zip(strata, counts)])
    rng.shuffle(rows)
    return X[rows], repaired[rows], {'observed_component_feature': j, 'slope': slope,
                                    'source_rows_sha256': protocol.array_sha256(rows)}


def prepare_worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    predictors = {}
    done = 0
    for task in TRACKS:
        paths = list(source_paths(task))
        tasks = sorted({p.parent.name for p in paths}, key=int)
        assigned = {name for i, name in enumerate(tasks) if i % shards == shard}
        for path in paths:
            if path.parent.name not in assigned:
                continue
            meta = json.loads(path.with_suffix('.json').read_text())
            track = meta['track']
            out = OUT / 'new_branches' / task / track / path.parent.name / path.name
            if out.with_suffix('.json').exists():
                continue
            data, condition = load_condition(meta, meta['family'], int(meta['rho']))
            X, y, Q = (data[k] for k in ('X_source', 'y_source', 'X_target'))
            assert len(y) == 256, (path, len(y))
            with np.load(path) as previous:
                assert np.array_equal(previous['y_target'], data['y_target']), path
                n_classes = previous['base'].shape[1] if task == 'classification' else None
            if 'X_context_sha256' in meta:
                assert protocol.array_sha256(X) == meta['X_context_sha256'], path
            if task == 'classification':
                rows = preserving_bootstrap(y, np.random.default_rng(0))
                A, b = X[rows], y[rows]
                details = {'source_rows_sha256': protocol.array_sha256(rows)}
            else:
                A, b, details = repaired_context(X, y)
            if task not in predictors:
                predictors[task] = Predictor(task, 'cuda:0')
            predictor = predictors[task]
            # Avoid memory-dependent retained multiclass tree state across fits.
            for model in predictor.models.values():
                root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
                if root is not None and hasattr(root, 'root'):
                    root.root = None
            torch.cuda.empty_cache()
            p, _ = predictor(A, b, Q, n_classes=n_classes)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix('.tmp.npz'); np.savez_compressed(tmp, prediction=p); tmp.replace(out)
            atomic_json(out.with_suffix('.json'), {'status': 'COMPLETE', 'source_bank': str(path),
                'source_bank_sha256': sha(path), 'observed_X_sha256': protocol.array_sha256(X),
                'observed_y_sha256': protocol.array_sha256(y), 'query_X_sha256': protocol.array_sha256(Q),
                'constructed_X_sha256': protocol.array_sha256(A), 'constructed_y_sha256': protocol.array_sha256(b),
                'rows': len(b), 'features': A.shape[1], 'seed': 0, 'details': details,
                'repair_receives_target_labels': False, 'repair_receives_clean_y_or_generator_feature': False,
                'checkpoint': str(__import__('run_tabicl').CHECKPOINTS / f'tabicl-{"classifier" if task == "classification" else "regressor"}-v2-20260212.ckpt')})
            done += 1
            if done % 25 == 0:
                print('BRANCHES', shard, done, task, flush=True)
    atomic_json(OUT / f'prepare_{shard}.json', {'status': 'COMPLETE', 'new_conditions': done})


def load(task, matched):
    folds, _ = fold_mapping(task, 'five')
    cells = []
    for path in source_paths(task):
        meta = json.loads(path.with_suffix('.json').read_text())
        addition = OUT / 'new_branches' / task / meta['track'] / path.parent.name / path.name
        assert addition.with_suffix('.json').exists(), addition
        with np.load(path) as f:
            base, bank, y = f['base'].copy(), f['members'].copy(), f['y_target'].copy()
            mu, sigma = float(f['source_mean']), float(f['source_scale'])
        slot = 3 if task == 'classification' else 0
        use_addition = not matched if task == 'classification' else matched
        if use_addition:
            with np.load(addition) as f:
                bank[slot] = f['prediction']
        assert len(bank) == 5
        meta['data_id'] = int(meta['data_id']); meta['task_id'] = int(meta['task_id'])
        meta['block'] = 'five' if meta['track'].startswith('five') else 'seven'
        # Used only as an evaluation/loss normalization, never by constructors.
        meta['scale'] = max(float(np.std(y)), 1e-12) if task == 'regression' else 1.
        cells.append({'task': task, 'role': 'validation', 'meta': meta, 'path': path,
            'base': base, 'bank': bank, 'y': y, 'mu': mu, 'sigma': max(sigma, 1e-12),
            'fold': folds[meta['data_id']]})
    return cells


def exposed(cell, task):
    m = cell['meta']
    if m['rho'] == 0:
        return False
    return m['track'].startswith('five') if task == 'classification' else m['family'] in ('concept', 'temporal')


def validation_loss(cells, policy, task):
    values = {}
    for c in cells:
        prediction = fusion.predict(c, policy)
        if task == 'classification':
            loss = -np.log(np.clip(prediction[np.arange(len(c['y'])), c['y'].astype(int)], 1e-12, 1.)).mean()
        else:
            # Scale depends only on observed source labels, not target variance.
            loss = np.mean(((prediction - c['y']) / c['sigma'])**2)
        values.setdefault(c['meta']['data_id'], []).append(loss)
    return float(np.mean([np.mean(v) for v in values.values()]))


def fit_worker(task, fold):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    fusion.GATE_VARIANT = 'entropy'
    folder = OUT / 'folds' / task / str(fold)
    rows = []
    for method, seen, matched in [('A', True, True), ('B', False, True), ('C', True, False), ('D', False, False)]:
        cells = load(task, matched)
        train = [c for c in cells if c['fold'] != fold and (seen or not exposed(c, task))]
        track, family = TARGETS[task]
        test = [c for c in cells if c['fold'] == fold and c['meta']['track'] == track and
                (c['meta']['family'] == family or c['meta']['rho'] == 0)]
        train_ids = {c['meta']['data_id'] for c in train}
        test_ids = {c['meta']['data_id'] for c in test}
        assert train_ids.isdisjoint(test_ids)
        assert seen or not any(exposed(c, task) for c in train)
        inner = min(c['fold'] for c in train)
        inner_train = [c for c in train if c['fold'] != inner]
        inner_val = [c for c in train if c['fold'] == inner]
        # Fusion's training uses a source-normalized MSE; target spread is not
        # permitted to influence policy or regularization selection.
        for c in train:
            c['meta']['scale'] = c['sigma']
        trials = []
        for regularization in REGS:
            p = fusion.fit(inner_train, task, 5, 1., regularization)
            trials.append((validation_loss(inner_val, p, task), regularization))
        _, reg = min(trials)
        policy = fusion.fit(train, task, 5, 1., reg)
        atomic_json(folder / f'{method}_policy.json', {'policy': policy, 'seen': seen, 'matched': matched,
            'task': task, 'outer_fold': fold, 'training_data_ids': sorted(train_ids),
            'excluded_data_ids': sorted(test_ids), 'inner_fold': inner,
            'trials': trials, 'calibration_conditions': len(train), 'unseen_mechanism_audit': seen or not any(exposed(c, task) for c in train)})
        for c in test:
            meta = c['meta']
            preds = {method: fusion.predict(c, policy)}
            if method == 'A':
                preds['Base'] = c['base']
                preds['Single-matched'] = c['bank'][3 if task == 'classification' else 0]
            for name, prediction in preds.items():
                p = folder / 'predictions' / str(meta['task_id']) / f'{meta["rho"]}_{name}.npz'
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, prediction=prediction, y=c['y'])
                scale = max(float(c['y'].std()), 1e-12) if task == 'regression' else 1.
                rows.append({'task': task, 'dataset': meta['dataset'], 'data_id': meta['data_id'],
                    'task_id': meta['task_id'], 'rho': meta['rho'], 'method': name,
                    'fold': fold, **metrics(c['y'], prediction, task, scale)})
        print('POLICY', task, fold, method, len(train), len(test), flush=True)
    pd.DataFrame(rows).to_csv(folder / 'detail.csv', index=False)
    atomic_json(folder / 'completion.json', {'status': 'COMPLETE', 'rows': len(rows)})


def launch(phase, shards):
    initialize()
    if phase == 'prepare':
        jobs = [(str(i), ['prepare-worker', '--shard', str(i), '--shards', str(shards)], i) for i in range(shards)]
    else:
        jobs = [(f'{task}_{f}', ['fit-worker', '--task', task, '--fold', str(f)], None) for task in TRACKS for f in range(5)]
    def execute(job):
        name, args, gpu = job
        path = OUT / 'logs' / f'{phase}_{name}.log'; path.parent.mkdir(exist_ok=True)
        env = os.environ.copy()
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with path.open('a') as log:
            result = subprocess.run([sys.executable, __file__, *args], env=env, stdout=log, stderr=subprocess.STDOUT)
        return {'job': name, 'returncode': result.returncode, 'log': str(path)}
    results = []
    with ThreadPoolExecutor(max_workers=shards if phase == 'prepare' else 4) as executor:
        for f in as_completed([executor.submit(execute, j) for j in jobs]):
            record = f.result(); results.append(record); print(record, flush=True)
            atomic_json(OUT / f'{phase}_execution.json', {'status': 'RUNNING', 'workers': results})
    atomic_json(OUT / f'{phase}_execution.json', {'status': 'COMPLETE' if all(x['returncode'] == 0 for x in results) else 'FAILED', 'workers': results})
    if any(x['returncode'] for x in results):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['initialize', 'prepare', 'prepare-worker', 'fit', 'fit-worker'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--task', choices=list(TRACKS))
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    if args.phase == 'initialize':
        initialize()
    elif args.phase == 'prepare-worker':
        prepare_worker(args.shard, args.shards)
    elif args.phase == 'fit-worker':
        fit_worker(args.task, args.fold)
    else:
        launch(args.phase, args.shards)
