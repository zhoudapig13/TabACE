"""Single generic row-filter substitution, frozen before its TFM inference.

The filter has no shift-family, severity, attacked-coordinate, or target-label
argument. It preserves all input columns and observed labels. Earlier Shortcut
development diagnostics informed the hypothesis; unseen refers to exclusion
from calibration, not ignorance of the mechanism during method development.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
import torch

import abcd_mechanism_v84 as previous
import tabace_fusion_v28 as fusion
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor, metrics
from shared_reference import class_subset, load_condition, geometry
from sklearn.metrics import pairwise_distances

OUT = RUN / 'results/abcd_support_filter_v89'


def generic_context(X, y, Q, budget=256):
    """Class-preserving half-pool filter using an unlabeled support envelope."""
    lower, upper = np.quantile(Q, [.025, .975], axis=0)
    scale = np.maximum(np.asarray(X, dtype=np.float64).std(axis=0), 1e-6)
    excess = np.maximum(np.maximum(lower - X, X - upper), 0) / scale
    score = np.max(excess, axis=1)
    rows = class_subset(y, score, .5, budget)
    return X[rows], y[rows], rows


def fraction_control(X, y, Q, budget=256):
    """Original representative score with only retention changed to one half."""
    Z, V, _, _ = geometry(X, Q)
    distances = pairwise_distances(Z, V, metric='sqeuclidean')
    k = min(10, len(V))
    nearest = np.partition(distances, k - 1, axis=1)[:, :k].mean(axis=1)
    score = nearest + .35 * (np.minimum(np.abs(Z), 10.)**2).mean(axis=1)
    rows = class_subset(y, score, .5, budget)
    return X[rows], y[rows], rows


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text())['code_sha256'] == previous.sha(__file__)
        return
    atomic_json(p, {'status': 'FROZEN_BEFORE_NEW_BRANCH_INFERENCE_AND_POLICY_FITTING',
        'seed': 0, 'datasets': 72, 'context_cap': 256, 'alternative_contexts': 5,
        'estimators_per_branch': 1, 'slot_replaced': 1, 'old_slot': 'representative',
        'new_slot': 'generic max-support-excess row filter', 'retained_source_fraction': .5,
        'target_quantile_envelope': [.025, .975], 'scale': 'observed source std, floor1e-6',
        'candidates': ['original', 'fraction_only', 'support_filter'],
        'fraction_only_control': 'original representative score with the same50% retention as support_filter',
        'matching_feature_drop_slot': 3, 'matching_slot_removed_in_C_D': True,
        'constructor_uses_attacked_coordinate_or_mask': False,
        'constructor_uses_family_strength_or_target_labels': False,
        'all_columns_and_observed_labels_preserved': True,
        'Base_and_input_benchmark_unchanged': True, 'new_raw_feature_supervised_predictor': False,
        'extra_labeled_examples_relative_to_v84': False, 'backbone_finetuning': False,
        'policy_fit': 'original entropy-gated confidence-agreement fusion, gain1',
        'regularization_grid': previous.REGS,
        'selection': 'three bank candidates and regularization selected by inner-dataset validation CE; original frozen fits reused, new candidates refitted; save all outcomes',
        'inner_and_outer_dataset_folds': 'same as v84',
        'B_D_excludes_all_positive_strength_five_source_perturbations': True,
        'calibration_labels_from_evaluated_dataset': False,
        'development_diagnostics_informed_the_filter_design': True,
        'unseen_meaning': 'absent from calibration, not unseen to the researcher at method-design time',
        'independent_confirmation': False, 'adaptation_dagger': True,
        'original_paper_replication': False, 'paper_Base_alignment': False,
        'statistical_significance_not_used_for_selection': True, 'Markdown_written': False,
        'code_sha256': previous.sha(__file__), 'source_manifest_sha256': previous.sha(previous.OUT / 'input_manifest.json')})


def prepare(shard, shards):
    torch.set_num_threads(1); torch.manual_seed(0)
    paths = list(previous.source_paths('classification'))
    tasks = sorted({p.parent.name for p in paths}, key=int)
    assigned = {name for i, name in enumerate(tasks) if i % shards == shard}
    predictor = Predictor('classification', 'cuda:0')
    count = 0
    for path in paths:
        if path.parent.name not in assigned:
            continue
        meta = json.loads(path.with_suffix('.json').read_text())
        out = OUT / 'branches' / meta['track'] / path.parent.name / path.name
        if out.with_suffix('.json').exists():
            continue
        data, _ = load_condition(meta, meta['family'], int(meta['rho']))
        with np.load(path) as z:
            n_classes = z['base'].shape[1]
            assert np.array_equal(z['y_target'], data['y_target'])
        X, y, Q = data.get('X_pool', data['X_source']), data.get('y_pool', data['y_source']), data['X_target']
        predictions, details = {}, {}
        for name, factory in [('support_filter', generic_context), ('fraction_only', fraction_control)]:
            A, b, rows = factory(X, y, Q)
            for model in predictor.models.values():
                root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
                if root is not None and hasattr(root, 'root'):
                    root.root = None
            torch.cuda.empty_cache()
            prediction, _ = predictor(A, b, Q, n_classes=n_classes)
            predictions['prediction_' + name] = prediction
            predictions['rows_' + name] = rows
            details[name] = {'rows': len(rows), 'selected_X_sha256': protocol.array_sha256(A),
                'selected_y_sha256': protocol.array_sha256(b), 'selected_rows_sha256': protocol.array_sha256(rows)}
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix('.tmp.npz'); np.savez_compressed(temp, **predictions); temp.replace(out)
        atomic_json(out.with_suffix('.json'), {'status': 'COMPLETE', 'original_bank': str(path),
            'original_bank_sha256': previous.sha(path), 'details': details, 'features': X.shape[1],
            'source_X_sha256': protocol.array_sha256(X), 'source_y_sha256': protocol.array_sha256(y),
            'query_X_sha256': protocol.array_sha256(Q),
            'columns_deleted': 0, 'source_labels_modified': False, 'target_labels_received': False})
        count += 1
        if count % 50 == 0:
            print('SUPPORT_BRANCHES', shard, count, flush=True)


def load(matched, variant):
    cells = previous.load('classification', matched)
    if variant != 'original':
        for c in cells:
            m = c['meta']; p = OUT / 'branches' / m['track'] / str(m['task_id']) / c['path'].name
            with np.load(p) as z:
                c['bank'][1] = z['prediction_' + variant]
    return cells


def worker(fold):
    torch.set_num_threads(1); torch.manual_seed(0); fusion.GATE_VARIANT = 'entropy'
    folder = OUT / 'folds/classification' / str(fold)
    rows = []
    for method, seen, matched in [('A', True, True), ('B', False, True), ('C', True, False), ('D', False, False)]:
        candidates = []
        tests = {}
        for variant in ('original', 'fraction_only', 'support_filter'):
            cells = load(matched, variant)
            train = [c for c in cells if c['fold'] != fold and (seen or not previous.exposed(c, 'classification'))]
            test = [c for c in cells if c['fold'] == fold and c['meta']['track'] == 'five_corrected' and
                    (c['meta']['rho'] == 0 or c['meta']['family'] == 'shortcut')]
            train_ids = {c['meta']['data_id'] for c in train}; test_ids = {c['meta']['data_id'] for c in test}
            assert train_ids.isdisjoint(test_ids)
            assert seen or not any(previous.exposed(c, 'classification') for c in train)
            inner = min(c['fold'] for c in train)
            inner_train = [c for c in train if c['fold'] != inner]
            inner_val = [c for c in train if c['fold'] == inner]
            if variant == 'original':
                path = previous.OUT / 'folds/classification' / str(fold) / f'{method}_policy.json'
                saved = json.loads(path.read_text())
                assert saved['training_data_ids'] == sorted(train_ids)
                assert saved['excluded_data_ids'] == sorted(test_ids)
                best_loss = min(t[0] for t in saved['trials'])
                candidates.append({'variant': variant, 'trials': saved['trials'], 'inner_CE': best_loss,
                                   'policy': saved['policy'], 'reused_fit_sha256': previous.sha(path)})
                tests[variant] = test
                continue
            trials = []
            for reg in previous.REGS:
                model = fusion.fit(inner_train, 'classification', 5, 1., reg)
                loss = previous.validation_loss(inner_val, model, 'classification')
                trials.append({'regularization': reg, 'inner_CE': loss})
            best = min(trials, key=lambda t: t['inner_CE'])
            final = fusion.fit(train, 'classification', 5, 1., best['regularization'])
            candidates.append({'variant': variant, 'trials': trials, 'inner_CE': best['inner_CE'], 'policy': final})
            tests[variant] = test
        selected = min(candidates, key=lambda t: t['inner_CE'])
        atomic_json(folder / f'{method}_policy.json', {'method': method, 'seen': seen, 'matched': matched,
            'training_data_ids': sorted(train_ids), 'excluded_data_ids': sorted(test_ids),
            'inner_fold': inner, 'candidates': candidates, 'selected_variant': selected['variant'],
            'selection_uses_outer_test_metrics': False})
        for candidate in candidates:
            for c in tests[candidate['variant']]:
                m = c['meta']; prediction = fusion.predict(c, candidate['policy'])
                result_method = method if candidate['variant'] == selected['variant'] else method + '_' + candidate['variant']
                path = folder / 'predictions' / str(m['task_id']) / f'{m["rho"]}_{result_method}.npz'
                path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, prediction=prediction, y=c['y'])
                rows.append({'task': 'classification', 'method': result_method, 'data_id': m['data_id'],
                    'task_id': m['task_id'], 'dataset': m['dataset'], 'rho': m['rho'], 'fold': fold,
                    'variant': candidate['variant'], **metrics(c['y'], prediction, 'classification', 1.)})
        if method == 'A':
            for c in tests['original']:
                m = c['meta']
                for name, p in [('Base', c['base']), ('Single-matched', c['bank'][3])]:
                    path = folder / 'predictions' / str(m['task_id']) / f'{m["rho"]}_{name}.npz'
                    np.savez_compressed(path, prediction=p, y=c['y'])
                    rows.append({'task': 'classification', 'method': name, 'data_id': m['data_id'],
                        'task_id': m['task_id'], 'dataset': m['dataset'], 'rho': m['rho'], 'fold': fold,
                        'variant': 'original', **metrics(c['y'], p, 'classification', 1.)})
        print('SUPPORT_POLICY', fold, method, selected['variant'], [c['inner_CE'] for c in candidates], flush=True)
    pd.DataFrame(rows).to_csv(folder / 'detail.csv', index=False)
    atomic_json(folder / 'completion.json', {'status': 'COMPLETE', 'rows': len(rows)})


def launch(phase, shards):
    freeze()
    def execute(i):
        args = ['prepare-worker', '--shard', str(i), '--shards', str(shards)] if phase == 'prepare' else ['fit-worker', '--fold', str(i)]
        env = os.environ.copy()
        if phase == 'prepare':
            env['CUDA_VISIBLE_DEVICES'] = str(i)
        log = OUT / f'{phase}_{i}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, *args], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'job': i, 'returncode': p.returncode, 'log': str(log)}
    records = []
    with ThreadPoolExecutor(max_workers=shards if phase == 'prepare' else 4) as pool:
        for f in as_completed([pool.submit(execute, i) for i in range(shards if phase == 'prepare' else 5)]):
            r = f.result(); records.append(r); print(r, flush=True)
            atomic_json(OUT / f'{phase}_execution.json', {'status': 'RUNNING', 'workers': records})
    atomic_json(OUT / f'{phase}_execution.json', {'status': 'COMPLETE' if all(r['returncode'] == 0 for r in records) else 'FAILED', 'workers': records})
    if any(r['returncode'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'prepare-worker', 'fit', 'fit-worker'])
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=8); p.add_argument('--fold', type=int)
    a = p.parse_args()
    if a.phase == 'prepare-worker':
        prepare(a.shard, a.shards)
    elif a.phase == 'fit-worker':
        worker(a.fold)
    else:
        launch(a.phase, a.shards)
