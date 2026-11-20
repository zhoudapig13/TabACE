"""ABCD on the frozen, explicitly semisynthetic Shortcut development cohort.

Two versions are declared before any member inference: the reconstructed
five-slot registry and a dagger version replacing the representative slot
with a generic marginal-density resampling context. Both are fully reported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

import abcd_mechanism_v84 as original
import semisynthetic_shortcut_v90 as generator
import semisynthetic_base_refinement_v91 as calibration
import tabace_fusion_v28 as fusion
from prepare_data import RUN, atomic_json, protocol
from run_tabicl import Predictor, preserving_bootstrap, metrics
from shared_reference import contexts, load_condition
from fit_combiners import fold_mapping

OUT = RUN / 'results/semisynthetic_abcd_v92'
VERSIONS = ('original_registry', 'density_adapted')


def density_context(X, y, Q, budget=256):
    """All-column generic density emphasis; never reads family or column roles."""
    rng = np.random.default_rng(0)
    columns = np.sort(rng.choice(X.shape[1], min(64, X.shape[1]), replace=False))
    score = np.full(len(X), -np.inf)
    for j in columns:
        values = np.asarray(X[:, j], dtype=np.float64)
        target = np.asarray(Q[:, j], dtype=np.float64)
        bandwidth = max(.2 * float(np.std(np.r_[values, target])), 1e-6)
        kernel = np.exp(-.5 * ((values[:, None] - values[None]) / bandwidth)**2)
        source_density = np.maximum((kernel.sum(1) - 1) / max(len(values) - 1, 1), 1e-12)
        target_density = np.maximum(np.exp(-.5 * ((values[:, None] - target[None]) / bandwidth)**2).mean(1), 1e-12)
        score = np.maximum(score, np.log(source_density) - np.log(target_density))
    log_weights = -4. * np.clip(score, -20, 20)
    weights = np.exp(log_weights - log_weights.max()); weights /= weights.sum()
    mandatory = []
    for label in np.unique(y):
        ids = np.flatnonzero(y == label); w = weights[ids]; w = w / w.sum()
        mandatory.append(int(rng.choice(ids, p=w)))
    rows = np.r_[mandatory, rng.choice(len(y), budget - len(mandatory), replace=True, p=weights)].astype(int)
    rng.shuffle(rows)
    return X[rows], y[rows], rows, columns


def selection():
    p = calibration.OUT / 'Base_selection.json'
    selected = json.loads(p.read_text())
    assert selected['aligned_within_tolerance'] and selected['status'] == 'FROZEN_BEFORE_ABCD'
    return selected


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text())['code_sha256'] == original.sha(__file__)
        return
    selected = selection()
    atomic_json(p, {'status': 'FROZEN_BEFORE_MEMBER_INFERENCE', 'seed': 0, 'datasets': 72,
        'context_rows': 256, 'alternative_slots': 5, 'estimators_per_branch': 1,
        'Base_selection': selected, 'Base_selection_sha256': original.sha(calibration.OUT / 'Base_selection.json'),
        'versions': VERSIONS, 'primary_adapted_version_predeclared': 'density_adapted',
        'density_slot': 1, 'density_bandwidth_sd_multiplier': .2,
        'density_sampling_log_weight_multiplier': -4., 'density_scoring_columns_cap': 64,
        'density_scoring_columns_selected_without_labels': True,
        'density_context_source_label_modifications': False,
        'A_B_matching_slot3': 'observed-source leading-correlation feature removal; no synthetic-coordinate oracle',
        'C_D_slot3': 'class-preserving uniform current-context bootstrap',
        'Base_shared_across_versions_and_ABCD': True,
        'Base_backbone_finetuning': False, 'adaptation': 'generic constructor and train-only calibration weights; no new raw-feature prediction head',
        'B_D_exclusions': 'all positive-severity five-source-perturbation episodes including Mixed and semisynthetic Shortcut',
        'calibration_clean_and_positive_Shortcut': 'replace old conditions with frozen semisynthetic conditions',
        'other_calibration_conditions': 'unchanged real-data five/seven caches on other datasets',
        'outer_and_inner_dataset_folds': 'same v84 mapping', 'regularization_grid': original.REGS,
        'fusion': 'manuscript entropy/confidence/agreement gate, gain1',
        'target_labels_used_in_benchmark_generation': True,
        'target_labels_received_by_context_constructors_or_own_dataset_policy_training': False,
        'evaluation_scope': 'development component verification; prior generator calibration used development metrics',
        'original_paper_replication': False, 'independent_confirmation': False,
        'all_original_and_adapted_outcomes_will_be_retained': True,
        'code_sha256': original.sha(__file__), 'generator_sha256': original.sha(generator.__file__),
        'Markdown_written': False})


def prepare(shard, shards):
    torch.set_num_threads(1); torch.manual_seed(0)
    selected = selection(); config = selected['configuration']
    paths = list(original.source_paths('classification'))
    ids = sorted({p.parent.name for p in paths}, key=int)
    assigned = {identifier for i, identifier in enumerate(ids) if i % shards == shard}
    predictor = Predictor('classification', 'cuda:0'); latent = {}; done = 0
    def predict(A, b, Q, n_classes):
        for model in predictor.models.values():
            root = getattr(getattr(model, 'model_', None), 'icl_predictor', None)
            if root is not None and hasattr(root, 'root'):
                root.root = None
        torch.cuda.empty_cache()
        return predictor(A, b, Q, n_classes=n_classes)[0]
    for path in paths:
        if path.parent.name not in assigned:
            continue
        meta = json.loads(path.with_suffix('.json').read_text())
        out = OUT / 'banks' / meta['track'] / path.parent.name / path.name
        if out.with_suffix('.json').exists():
            continue
        new = meta['track'].startswith('five') and (meta['rho'] == 0 or meta['family'] == 'shortcut')
        if new:
            if meta['task_id'] not in latent:
                latent[meta['task_id']] = generator.latent(meta)
            data = generator.materialize(meta, config, int(meta['rho']), latent[meta['task_id']])
            base_path = Path(selected['prediction_root']) / str(meta['task_id']) / config['name'] / f'{meta["rho"]}.npz'
            base_meta = json.loads(base_path.with_suffix('.json').read_text())
            assert protocol.array_sha256(data['X_source']) == base_meta['source_X_sha256']
            assert protocol.array_sha256(data['X_target']) == base_meta['target_X_sha256']
            with np.load(base_path) as z:
                base = z['prediction']; assert np.array_equal(z['y'], data['y_target'])
            n_classes = base.shape[1]
            X, y, Q = data['X_source'], data['y_source'], data['X_target']
            views = contexts(X, y, Q, X, y, 256, 'classification')
            bank = np.stack([predict(A, b, T, n_classes) for _, A, b, T in views])
            rows = preserving_bootstrap(y, np.random.default_rng(0))
            bootstrap = predict(X[rows], y[rows], Q, n_classes)
            mu, sigma = float(y.mean()), max(float(y.std()), 1e-9)
            meta = meta | {'track': 'five_semisynthetic', 'semisynthetic': True,
                'source_configuration': config, 'reference_Base_file': str(base_path),
                'benchmark_generation_uses_target_labels': True, 'original_paper_replication': False}
        else:
            data, _ = load_condition(meta, meta['family'], int(meta['rho']))
            with np.load(path) as z:
                base, bank = z['base'], z['members']; mu, sigma = float(z['source_mean']), float(z['source_scale'])
                assert np.array_equal(z['y_target'], data['y_target'])
            n_classes = base.shape[1]
            p = original.OUT / 'new_branches/classification' / meta['track'] / path.parent.name / path.name
            with np.load(p) as z:
                bootstrap = z['prediction']
        X, y, Q = data.get('X_pool', data['X_source']), data.get('y_pool', data['y_source']), data['X_target']
        A, b, rows, columns = density_context(X, y, Q)
        density = predict(A, b, Q, n_classes)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, base=base, members=bank, bootstrap=bootstrap, density=density,
            y_target=data['y_target'], source_mean=mu, source_scale=sigma,
            density_rows=rows, density_columns=columns)
        tmp.replace(out)
        atomic_json(out.with_suffix('.json'), meta | {'status': 'COMPLETE', 'new_semisynthetic_condition': new,
            'original_bank_path': str(path), 'original_bank_sha256': original.sha(path),
            'density_source_X_sha256': protocol.array_sha256(X), 'density_source_y_sha256': protocol.array_sha256(y),
            'query_X_sha256': protocol.array_sha256(Q), 'selected_X_sha256': protocol.array_sha256(A),
            'selected_y_sha256': protocol.array_sha256(b), 'density_source_rows': len(rows),
            'density_constructor_deletes_features': False, 'density_constructor_receives_target_labels': False})
        done += 1
        if done % 50 == 0:
            print('SEMISYNTH_BANKS', shard, done, flush=True)


def load(matched, version):
    folds, _ = fold_mapping('classification', 'five')
    result = []
    for path in sorted((OUT / 'banks').glob('*/*/*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            bank = z['members'].copy()
            if not matched:
                bank[3] = z['bootstrap']
            if version == 'density_adapted':
                bank[1] = z['density']
            cell = {'base': z['base'], 'bank': bank, 'y': z['y_target'],
                    'mu': float(z['source_mean']), 'sigma': float(z['source_scale'])}
        meta['data_id'] = int(meta['data_id']); meta['task_id'] = int(meta['task_id'])
        meta['scale'] = 1.; meta['block'] = 'five' if meta['track'].startswith('five') else 'seven'
        result.append(cell | {'meta': meta, 'task': 'classification', 'role': 'validation',
                             'fold': folds[meta['data_id']], 'path': path})
    assert len(result) == 2527
    return result


def worker(version, fold):
    torch.set_num_threads(1); torch.manual_seed(0); fusion.GATE_VARIANT = 'entropy'
    folder = OUT / version / 'folds/classification' / str(fold); rows = []
    for method, seen, matched in [('A', True, True), ('B', False, True), ('C', True, False), ('D', False, False)]:
        cells = load(matched, version)
        train = [c for c in cells if c['fold'] != fold and (seen or not original.exposed(c, 'classification'))]
        test = [c for c in cells if c['fold'] == fold and c['meta']['track'] == 'five_semisynthetic']
        train_ids = {c['meta']['data_id'] for c in train}; test_ids = {c['meta']['data_id'] for c in test}
        assert train_ids.isdisjoint(test_ids)
        assert seen or not any(original.exposed(c, 'classification') for c in train)
        inner = min(c['fold'] for c in train)
        first = [c for c in train if c['fold'] != inner]; validation = [c for c in train if c['fold'] == inner]
        trials = []
        for reg in original.REGS:
            p = fusion.fit(first, 'classification', 5, 1., reg)
            trials.append([original.validation_loss(validation, p, 'classification'), reg])
        _, reg = min(trials)
        policy = fusion.fit(train, 'classification', 5, 1., reg)
        atomic_json(folder / f'{method}_policy.json', {'policy': policy, 'version': version,
            'seen': seen, 'matched': matched, 'trials': trials, 'inner_fold': inner,
            'training_data_ids': sorted(train_ids), 'excluded_data_ids': sorted(test_ids),
            'unseen_mechanism_audit': seen or not any(original.exposed(c, 'classification') for c in train)})
        for c in test:
            m = c['meta']; predictions = {method: fusion.predict(c, policy)}
            if method == 'A':
                predictions['Base'] = c['base']; predictions['Single-matched'] = c['bank'][3]
            for name, p in predictions.items():
                path = folder / 'predictions' / str(m['task_id']) / f'{m["rho"]}_{name}.npz'
                path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, prediction=p, y=c['y'])
                rows.append({'task': 'classification', 'method': name, 'data_id': m['data_id'],
                    'task_id': m['task_id'], 'dataset': m['dataset'], 'rho': m['rho'], 'fold': fold,
                    'version': version, **metrics(c['y'], p, 'classification', 1.)})
        print('SEMISYNTH_POLICY', version, fold, method, flush=True)
    pd.DataFrame(rows).to_csv(folder / 'detail.csv', index=False)
    atomic_json(folder / 'completion.json', {'status': 'COMPLETE', 'rows': len(rows)})


def launch(phase, shards):
    freeze()
    jobs = [(str(i), ['prepare-worker', '--shard', str(i), '--shards', str(shards)], i) for i in range(shards)] if phase == 'prepare' else [
        (f'{version}_{fold}', ['fit-worker', '--version', version, '--fold', str(fold)], None)
        for version in VERSIONS for fold in range(5)]
    def execute(job):
        name, args, gpu = job; env = os.environ.copy()
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        log = OUT / f'{phase}_{name}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, *args], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'job': name, 'returncode': p.returncode, 'log': str(log)}
    records = []
    with ThreadPoolExecutor(max_workers=shards if phase == 'prepare' else 4) as pool:
        for future in as_completed([pool.submit(execute, j) for j in jobs]):
            r = future.result(); records.append(r); print(r, flush=True)
            atomic_json(OUT / f'{phase}_execution.json', {'status': 'RUNNING', 'workers': records})
    atomic_json(OUT / f'{phase}_execution.json', {'status': 'COMPLETE' if all(r['returncode'] == 0 for r in records) else 'FAILED', 'workers': records})
    if any(r['returncode'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'prepare-worker', 'fit', 'fit-worker'])
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=8)
    p.add_argument('--version', choices=VERSIONS); p.add_argument('--fold', type=int)
    a = p.parse_args()
    if a.phase == 'prepare-worker':
        prepare(a.shard, a.shards)
    elif a.phase == 'fit-worker':
        worker(a.version, a.fold)
    else:
        launch(a.phase, a.shards)
