"""Frozen reconstructed Shared on six unregistered generators.

Uses raw official-data splits, not the label-conditional Shortcut development
feature or deliberately weakened ABCD versions. The existing v77 pooled,
dataset-OOF Shared policies and v28 constructors are unchanged. No new-family
calibration, parameter search, target-label adaptation, or score targets.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import xml.etree.ElementTree as ET

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tabace_context_reproduction_v28 as constructors
import tabace_fusion_v28 as fusion
import unseen_mechanisms_v117 as novel
import seven_shifts as seven
from fit_combiners import fold_mapping
from prepare_data import ROOT, RUN, LOCAL_CC18, atomic_json, protocol
from run_tabicl import Predictor, metrics, normalize, CHECKPOINTS

OUT = RUN / 'results/unseen_shared_robustness_v117'
POLICIES = RUN / 'results/tabace_paper_rule_reconstruction_v77/fold_runs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npwrite(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def numeric_mask(meta, width):
    if meta['task'] == 'classification':
        path = LOCAL_CC18 / str(meta['task_id']) / 'source.json'
        flags = json.loads(path.read_text())['categorical']
        if len(flags) != width:
            frame = pd.read_parquet(path.with_name('X.parquet'))
            kept = np.flatnonzero(frame.notna().any(axis=0).to_numpy())
            flags = [flags[i] for i in kept]
    else:
        folder = RUN / 'cache/openml/org/openml/www/datasets' / str(meta['data_id'])
        path = folder / 'dataset.pkl.py3'
        frame, categorical, names = pickle.loads(path.read_bytes())
        xml = ET.parse(folder / 'description.xml')
        ignored = set()
        for tag in ('default_target_attribute', 'ignore_attribute', 'row_id_attribute'):
            for item in xml.iter():
                if item.tag.split('}')[-1] == tag and item.text:
                    ignored.update(x.strip() for x in item.text.split(','))
        keep = [i for i, name in enumerate(names) if name not in ignored]
        if len(keep) != width:
            # Legacy SimpleImputer dropped globally all-missing columns.
            keep = [i for i in keep if frame.iloc[:, i].notna().any()]
        flags = [categorical[i] for i in keep]
    assert len(flags) == width, (meta['task'], meta['task_id'], len(flags), width)
    return np.flatnonzero(~np.asarray(flags, dtype=bool)), str(path), sha(path)


def raw_condition(meta, track):
    task, tid = meta['task'], meta['task_id']
    raw = RUN / 'cache/five' / task / f'{tid}.npz'
    with np.load(raw) as z:
        cache = {k: z[k] for k in z.files}
    if track == 'perturbation':
        rows = (np.arange(len(cache['y_source'])) if task == 'classification' else
            np.random.default_rng(2718).permutation(len(cache['y_source']))[:256])
        X, y, Q, target = cache['X_source'][rows], cache['y_source'][rows], cache['X_target'], cache['y_target']
        source_ids, target_ids = cache['source_indices'][rows], cache['target_indices']
    elif task == 'regression':
        _, _, _, _, X, y, Q, target = seven.regression_clean(meta)
        rng = np.random.default_rng(2718)
        rows = rng.permutation(len(cache['y_source']))[:min(len(cache['y_source']), 1024)][:256]
        qrows = rng.choice(len(cache['y_target']), min(256, len(cache['y_target'])), replace=False)
        source_ids, target_ids = cache['source_indices'][rows], cache['target_indices'][qrows]
        assert np.array_equal(X, cache['X_source'][rows]) and np.array_equal(target, cache['y_target'][qrows])
    else:
        H, hy, R, ry, Q, target = seven.classification_clean(meta)
        X, y = np.vstack([H[:192], R]), np.r_[hy[:192], ry]
        labels = pd.read_parquet(LOCAL_CC18 / str(tid) / 'y.parquet').iloc[:, 0]
        valid = np.flatnonzero(labels.notna().to_numpy())
        encoded = LabelEncoder().fit_transform(labels.iloc[valid])
        chosen = protocol._balanced_cap_indices(encoded, 448, 0)
        source, qrows = train_test_split(chosen, train_size=320, test_size=128,
            stratify=encoded[chosen], random_state=0)
        history, reference = train_test_split(source, train_size=256, test_size=64,
            stratify=encoded[source], random_state=0)
        rows = np.r_[history[:192], reference]
        source_ids, target_ids = valid[rows], valid[qrows]
        assert np.array_equal(y, encoded[rows]) and np.array_equal(target, encoded[qrows])
    assert len(y) <= 256 and len(set(source_ids) & set(target_ids)) == 0
    return X, y, Q, target, source_ids, target_ids, raw


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'plan.json').exists():
        plan = json.loads((OUT / 'plan.json').read_text())
        assert sha(__file__) == plan['code_sha256']
        assert sha(novel.__file__) == plan['generator_sha256']
        return
    snapshots = []
    for task in ('classification', 'regression'):
        for fold in range(5):
            source = POLICIES / task / f'fold_{fold}' / 'frozen_policy.json'
            old = json.loads(source.read_text())
            model = old['Shared']
            assert model['kind'] in ('base', 'gated_fusion')
            if model['kind'] != 'base':
                assert model['m'] == 5 and model['gain'] == 1.
            destination = OUT / 'policies' / task / f'{fold}.json'
            snapshot = {'Shared': model, 'training_data_ids': old['training_data_ids'],
                'excluded_data_ids': old['excluded_query_data_ids'], 'task': task, 'fold': fold,
                'original_file': str(source), 'original_sha256': sha(source),
                'new_mechanism_calibration': False}
            atomic_json(destination, snapshot)
            snapshots.append({'path': str(destination), 'sha256': sha(destination)})
    tasks = []
    for task in ('classification', 'regression'):
        tasks += [(json.loads(p.read_text()), 'perturbation')
            for p in sorted((RUN / 'cache/five' / task).glob('*.json'))]
    tasks += [(m, 'shift') for m in seven.registry()]
    cases = []
    for meta, track in tasks:
        task, tid = meta['task'], int(meta['task_id'])
        mapping, _ = fold_mapping(task, 'five')
        fold = mapping[int(meta['data_id'])]
        policy = json.loads((OUT / 'policies' / task / f'{fold}.json').read_text())
        assert int(meta['data_id']) not in policy['training_data_ids']
        assert int(meta['data_id']) in policy['excluded_data_ids']
        X, y, Q, target, source_ids, target_ids, original = raw_condition(meta, track)
        numeric, mask_file, mask_hash = numeric_mask(meta, X.shape[1])
        families, excluded = [], {}
        for family, spec in novel.FAMILIES.items():
            if spec['group'] != track:
                continue
            ok, reason = novel.applicable(X, numeric, family)
            if ok:
                families.append(family)
            else:
                excluded[family] = reason
        out = OUT / 'inputs' / track / task / f'{tid}.npz'
        npwrite(out, X_source=X, y_source=y, X_target=Q, numeric_columns=numeric,
            source_indices=source_ids, target_indices=target_ids)
        truth = OUT / 'truth' / track / task / f'{tid}.npz'
        npwrite(truth, y_target=target, target_indices=target_ids)
        info = {'task': task, 'task_id': tid, 'data_id': int(meta['data_id']),
            'dataset': meta['dataset'], 'track': track, 'fold': fold,
            'source_rows': len(y), 'query_rows': len(target), 'features': X.shape[1],
            'families': families, 'excluded_mechanisms': excluded,
            'numeric_features': len(numeric), 'nonconstant_numeric_features': len(novel.coordinates(X, numeric)),
            'input_file': str(out), 'input_sha256': sha(out), 'truth_file': str(truth),
            'truth_sha256': sha(truth), 'original_raw_cache': str(original),
            'original_cache_sha256': sha(original), 'categorical_schema': mask_file,
            'categorical_schema_sha256': mask_hash,
            'source_target_row_identities_disjoint': True,
            'no_target_labels_in_model_input_archive': True,
            'raw_legacy_preprocessing_uses_unlabeled_full_dataset_features': True,
            'source_or_target_y_modified_for_benchmark_alignment': False,
        }
        atomic_json(out.with_suffix('.json'), info)
        cases.append(info)
    assert len(cases) == 175
    coverage = []
    for task in ('classification', 'regression'):
        for family, spec in novel.FAMILIES.items():
            eligible = [c for c in cases if c['task'] == task and family in c['families']]
            coverage.append({'task': task, 'track': spec['group'], 'family': family,
                'datasets': len(eligible), 'excluded': [c['task_id'] for c in cases
                    if c['task'] == task and family in c['excluded_mechanisms']]})
    atomic_json(OUT / 'case_manifest.json', cases)
    code_files = [Path(__file__), Path(novel.__file__), Path(constructors.__file__),
        Path(fusion.__file__), Path(__file__).with_name('shared_reference.py'),
        Path(__file__).with_name('specific_reference.py'), Path(__file__).with_name('run_tabicl.py')]
    code_hashes = {}
    for path in code_files:
        dest = OUT / 'implementation_snapshot' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        code_hashes[path.name] = sha(path)
    plan = {
        'status': 'FROZEN_BEFORE_NEW_MECHANISM_INFERENCE', 'seed': 0, 'context_cap': 256,
        'backbone': 'TabICL-v2 official frozen weights', 'n_estimators_per_context': 1,
        'methods': ['Base', 'Mean6', 'Shared'], 'Shared_alternative_contexts': 5,
        'Shared_constructor': 'unchanged v28 shared_contexts',
        'Shared_policy': 'unchanged v77 pooled old-family dataset-OOF Shared',
        'new_context_constructor_or_fusion_training': False,
        'new_mechanisms_in_calibration': False, 'new_family_or_strength_given_to_Shared': False,
        'extra_target_supervision_or_ABCD_attenuation': False,
        'available_source_rows_for_every_method': 'same observed canonical context, at most256',
        'perturbation_split': 'CC18 raw256/128; CTR23 fixed raw256-source subset and official target up to512',
        'shift_split': 'original seven registry35 binary/33 regression; canonical256 and target128/at most256',
        'all_families': novel.FAMILIES, 'parameters': novel.PARAMETERS,
        'strengths': novel.STRENGTHS, 'clean': 'one identity condition per task and track',
        'new_statistical_categories_claimed': False,
        'interpretation': 'unseen generators within known broad perturbation/domain/covariate categories',
        'categorical_features_excluded_from_numeric_value_maps': True,
        'ineligibility_based_on_source_schema_not_performance': True,
        'coverage': coverage, 'new_score_targets_or_parameter_search': False,
        'scoring': 'four positive severities mean inside dataset, then equal dataset macro mean',
        'classification_primary': 'AUC', 'regression_primary': 'R2',
        'planned_comparisons': 'Shared minus Base for each new mechanism and each task;12 tests; Holm correction',
        'paired_statistics': 'dataset bootstrap95CI and dataset sign-flip tests; seed0 not seed variance',
        'old_datasets_have_been_used_before': True, 'new_operators_frozen_before_evaluation': True,
        'unseen_dataset_claim': False, 'original_paper_replication': False,
        'existing_reconstructed_calibration_used_development_and_semisynthetic_tasks': True,
        'raw_cache_preprocessing_is_transductive_X_only_shared_by_all_methods': True,
        'query_labels_received_by_generator_constructor_or_new_fit': False,
        'expected_cases': len(cases), 'expected_conditions': sum(1+4*len(c['families']) for c in cases),
        'policies': snapshots, 'code_sha256': sha(__file__), 'generator_sha256': sha(novel.__file__),
        'implementation_sha256': code_hashes, 'case_manifest_sha256': sha(OUT / 'case_manifest.json'),
        'checkpoints': {name: sha(CHECKPOINTS / name) for name in (
            'tabicl-classifier-v2-20260212.ckpt', 'tabicl-regressor-v2-20260212.ckpt')},
        'Markdown_written': False,
    }
    atomic_json(OUT / 'plan.json', plan)
    print('FROZEN_COVERAGE', json.dumps(coverage), flush=True)
    print('EXPECTED_CONDITIONS', plan['expected_conditions'], flush=True)


def worker(shard, shards):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    plan = json.loads((OUT / 'plan.json').read_text())
    assert sha(__file__) == plan['code_sha256'] and sha(novel.__file__) == plan['generator_sha256']
    cases = json.loads((OUT / 'case_manifest.json').read_text())
    order = np.random.default_rng(117).permutation(len(cases))
    predictors = {}
    for counter, position in enumerate(order):
        if counter % shards != shard:
            continue
        meta = cases[int(position)]
        task, track, tid = meta['task'], meta['track'], meta['task_id']
        if task not in predictors:
            predictors[task] = Predictor(task, 'cuda:0')
        predictor = predictors[task]
        input_path = Path(meta['input_file'])
        assert sha(input_path) == meta['input_sha256']
        with np.load(input_path) as z:
            assert 'y_target' not in z.files
            X0, y0, Q0, numeric = z['X_source'], z['y_source'], z['X_target'], z['numeric_columns']
        policy_file = OUT / 'policies' / task / f'{meta["fold"]}.json'
        saved = json.loads(policy_file.read_text())
        assert meta['data_id'] not in saved['training_data_ids']
        policy = saved['Shared']
        n_classes = len(np.unique(y0)) if task == 'classification' else None
        conditions = [('clean', 0)] + [(f, r) for f in meta['families'] for r in novel.STRENGTHS]
        for family, rho in conditions:
            out = OUT / 'predictions' / track / task / str(tid) / f'{family}_{rho}.npz'
            if out.with_suffix('.json').exists():
                continue
            if family == 'clean':
                X, y, Q = X0.copy(), y0.copy(), Q0.copy()
                intervention = {'family': 'clean', 'rho': 0, 'generator_receives_target_labels': False}
            else:
                X, y, Q, intervention = novel.generate(X0, y0, Q0, numeric, task, family, rho, tid)
            calls = 0
            cache = {}

            def infer(A, b, T):
                nonlocal calls
                key = tuple(protocol.array_sha256(np.asarray(v)) for v in (A, b, T))
                if key in cache:
                    return cache[key]
                for estimator in predictor.models.values():
                    root = getattr(getattr(estimator, 'model_', None), 'icl_predictor', None)
                    if root is not None and hasattr(root, 'root'):
                        root.root = None
                torch.cuda.empty_cache()
                p = predictor(A, b, T, members=1, random_state=0, n_classes=n_classes)[0]
                assert np.isfinite(p).all()
                calls += 1
                cache[key] = p
                return p

            base = infer(X, y, Q)
            views = constructors.shared_contexts(X, y, Q, len(y), task, base)
            assert len(views) == 5 and max(len(v) for _, _, v, _ in views) <= 256
            members = np.stack([infer(A, b, T) for _, A, b, T in views])
            cell = {'task': task, 'base': base, 'bank': members,
                'mu': float(y.mean()), 'sigma': max(float(y.std()), 1e-12)}
            shared = fusion.predict(cell, policy)
            mean = np.concatenate([base[None], members], axis=0).mean(axis=0)
            if task == 'classification':
                mean = normalize(mean)
            assert np.isfinite(shared).all() and np.isfinite(mean).all()
            npwrite(out, Base=base, Shared=shared, Mean6=mean, members=members,
                source_mean=cell['mu'], source_scale=cell['sigma'])
            record = {k: meta[k] for k in ('task', 'track', 'task_id', 'data_id', 'dataset', 'fold')}
            record.update(family=family, rho=rho, input_sha256=meta['input_sha256'],
                output_sha256=sha(out), policy_file=str(policy_file), policy_sha256=sha(policy_file),
                context_rows=len(y), query_rows=len(Q), branch_rows=[len(v) for _, _, v, _ in views],
                Shared_roles=[name for name, _, _, _ in views], actual_TFM_calls=calls,
                observed_source_X_sha256=protocol.array_sha256(X),
                observed_source_y_sha256=protocol.array_sha256(y),
                observed_target_X_sha256=protocol.array_sha256(Q),
                intervention=intervention, query_labels_opened_for_inference=False,
                new_mechanism_id_or_strength_passed_to_Shared=False)
            atomic_json(out.with_suffix('.json'), record)
        print('UNSEEN_CASE_COMPLETE', track, task, tid, len(conditions), flush=True)


def launch(shards):
    initialize()

    def execute(i):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(i)
        log = OUT / f'worker_{i}.log'
        with log.open('a') as handle:
            p = subprocess.run([sys.executable, __file__, 'worker', '--shard', str(i),
                '--shards', str(shards)], env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {'shard': i, 'returncode': p.returncode, 'log': str(log)}

    records = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(execute, i) for i in range(shards)]):
            records.append(future.result())
            print('UNSEEN_WORKER', records[-1], flush=True)
            atomic_json(OUT / 'execution.json', {'status': 'RUNNING', 'workers': records})
    success = all(r['returncode'] == 0 for r in records)
    atomic_json(OUT / 'execution.json', {'status': 'COMPLETE' if success else 'FAILED', 'workers': records})
    assert success


def paired_statistics(delta):
    values = np.asarray(delta, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    observed = abs(values.mean())
    exceed = 0
    for start in range(0, 100000, 2000):
        signs = 2*rng.integers(0, 2, size=(min(2000, 100000-start), len(values)))-1
        exceed += int((np.abs((signs*values).mean(axis=1)) >= observed-1e-15).sum())
    return {'gain': float(values.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
        'p_two_sided': (exceed+1)/100001, 'paired_datasets': len(values)}


def report():
    assert json.loads((OUT / 'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((OUT / 'plan.json').read_text())
    assert sha(__file__) == plan['code_sha256'] and sha(novel.__file__) == plan['generator_sha256']
    manifest = json.loads((OUT / 'case_manifest.json').read_text())
    paths = list((OUT / 'predictions').glob('*/*/*/*.npz'))
    assert len(paths) == plan['expected_conditions']
    cases = {(c['track'], c['task'], c['task_id']): c for c in manifest}
    rows = []
    for path in sorted(paths):
        m = json.loads(path.with_suffix('.json').read_text())
        info = cases[(m['track'], m['task'], m['task_id'])]
        assert sha(path) == m['output_sha256'] and sha(m['policy_file']) == m['policy_sha256']
        assert not m['query_labels_opened_for_inference'] and max(m['branch_rows']) <= 256
        assert not m['new_mechanism_id_or_strength_passed_to_Shared']
        assert sha(info['truth_file']) == info['truth_sha256']
        with np.load(info['truth_file']) as z:
            y = z['y_target']
        with np.load(path) as z:
            values = {name: z[name] for name in ('Base', 'Shared', 'Mean6')}
            cell = {'task': m['task'], 'base': z['Base'], 'bank': z['members'],
                'mu': float(z['source_mean']), 'sigma': float(z['source_scale'])}
        p = json.loads(Path(m['policy_file']).read_text())
        assert info['data_id'] not in p['training_data_ids']
        assert np.array_equal(values['Shared'], fusion.predict(cell, p['Shared']))
        for method, prediction in values.items():
            rows.append({k: m[k] for k in ('task', 'track', 'task_id', 'data_id', 'dataset', 'family', 'rho')}
                | {'method': method, **metrics(y, prediction, m['task'], max(float(y.std()), 1e-12))})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / 'detail.csv', index=False)
    summaries, statistics = [], []
    for (task, track, family), group in detail[detail.rho > 0].groupby(['task', 'track', 'family']):
        cols = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        units = group.groupby(['data_id', 'method'])[cols].mean()
        for method in ('Base', 'Mean6', 'Shared'):
            values = units.xs(method, level='method')
            summaries.append({'task': task, 'track': track, 'family': family, 'method': method,
                'datasets': len(values), **{k: float(values[k].mean()) for k in cols}})
        primary = cols[1] if task == 'classification' else cols[0]
        delta = units.xs('Shared', level='method')[primary]-units.xs('Base', level='method')[primary]
        statistics.append({'task': task, 'track': track, 'family': family, 'metric': primary,
            **paired_statistics(delta)})
    stats = pd.DataFrame(statistics)
    assert len(stats) == 12
    last = 0.
    for rank, index in enumerate(sorted(stats.index, key=lambda i: stats.loc[i, 'p_two_sided'])):
        last = max(last, min(1., (12-rank)*stats.loc[index, 'p_two_sided']))
        stats.loc[index, 'p_holm_12'] = last
    table = pd.DataFrame(summaries)
    table.to_csv(OUT / 'family_averages.csv', index=False)
    stats.to_csv(OUT / 'paired_statistics.csv', index=False)
    clean = []
    for (task, track, method), group in detail[detail.rho == 0].groupby(['task', 'track', 'method']):
        cols = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        clean.append({'task': task, 'track': track, 'method': method, 'datasets': group.data_id.nunique(),
            **{k: float(group[k].mean()) for k in cols}})
    pd.DataFrame(clean).to_csv(OUT / 'clean_averages.csv', index=False)
    for p in plan['policies']:
        assert sha(p['path']) == p['sha256']
        saved = json.loads(Path(p['path']).read_text())
        assert sha(saved['original_file']) == saved['original_sha256']
    for name, expected in plan['implementation_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == expected
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    atomic_json(OUT / 'audit.json', {'status': 'PASS', 'conditions': len(paths),
        'all_Shared_predictions_recomputed_from_frozen_policies': True,
        'own_evaluation_dataset_absent_from_fusion_training': True,
        'old_policies_and_Shared_implementation_unchanged': True,
        'no_new_family_calibration_or_extra_target_labels': True,
        'all_expected_cases_including_negative_results_retained': True,
        'mechanism_eligibility_determined_before_inference': True,
        'unseen_generator_not_new_statistical_category': True,
        'not_an_original_paper_replication': True, 'Markdown_bytes': 0})
    atomic_json(OUT / 'experiment_log.json', {
        'Purpose': 'Test transfer to six unregistered mechanisms with no new matching constructor or calibration',
        'Setting': plan,
        'Results': json.loads(table.to_json(orient='records', double_precision=15)),
        'Analysis': 'Interpret per-mechanism paired effects, including null effects or degradations. Broad shift classes overlap existing Domain/Covariate categories.',
        'Next Steps': 'Use actual paired confidence intervals to scope any rebuttal generalization claim. Do not tune operators after reading these results.',
    })
    print(table.to_string(index=False), flush=True)
    print(stats.to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['initialize', 'launch', 'worker', 'report'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    if a.phase == 'initialize':
        initialize()
    elif a.phase == 'worker':
        worker(a.shard, a.shards)
    elif a.phase == 'launch':
        launch(a.shards)
    else:
        report()
