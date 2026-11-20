"""Limited-reference label repair, retaining the observed 256-row context.

Frozen inputs/Base and original fusion formulas stay unchanged. A frozen TFM
uses 64 disjoint clean TRAIN references to identify inconsistent source labels.
The five Shared and five family-directed operators then use the repaired copy.
This is a supervised constructor adaptation, not original-paper replication.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import few_reference_contexts_v26 as few
from prepare_data import atomic_json, protocol

core = few.core
MIX_RATE = float(os.environ.get('TABACE_REPAIR_MIX_RATE', '0'))
OUT = core.RUN / ('results/reference_repair_contexts_v28' if MIX_RATE == 0
                  else 'results/reference_repair_mixed_contexts_v30')
few.OUT = OUT
VARIANT = 'reference_repair064' if MIX_RATE == 0 else 'reference_repair064_mix005'


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'plan.json'
    if p.exists():
        return
    previous = json.loads((core.OUT / 'plan.json').read_text())
    atomic_json(p, previous | dict(
        status='DECLARED_BEFORE_NEW_REPAIR_INFERENCE', variants=[VARIANT],
        adaptation='64-clean-reference-assisted source-label repair before registered context construction',
        source_reference_pool_cap=64, per_context_cap=256,
        maximum_clean_reference_row_replacement_fraction=MIX_RATE,
        Shared_alternatives=5, Specific_alternatives=10,
        classification_repair='replace observed source label only if frozen reference predictor disagrees and max probability >=0.8',
        regression_repair='replace observed source response only if its absolute residual from the reference prediction exceeds 3 reference-label standard deviations',
        target_label_access=False, benchmark_unchanged=True, source_features_unchanged=True,
        constructors_strength_blind=True, Shared_family_blind=True,
        source_label_repair_is_additional_clean_TRAIN_supervision=True,
        reference_fraction_per_view_upper_bound=None,
        reference_context_rows_are_not_inserted=MIX_RATE == 0,
        original_fusion_form_and_observed_normalization=True,
        hypothesis='Retain useful clean observed rows while removing gross label inconsistencies; avoid replacing the entire observed context by a weak small reference pool',
        thresholds_fixed_before_new_inference=True,
        development_followup_not_independent_confirmation=True,
        exact_archived_main_table_reproduction=False,
        start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    ))


def make(job, role, block, family, rho, predictor, pool, split, ref, cache=None):
    sub = Path('calibration') if role == 'validation' else Path('query') / block
    dest = OUT / core.rel(job) / sub / f'{family}_{rho}.npz'
    if dest.exists() and dest.with_suffix('.json').exists():
        return
    raw = core.RAW / core.rel(job) / sub / f'{family}_{rho}.npz'
    meta = json.loads(raw.with_suffix('.json').read_text())
    if role == 'query':
        data, _ = core.source.bank(job['task'], block, job['meta'], family, rho)
    else:
        data = core.source.materialize(job, pool, np.r_[split['train'], split['val']], split['cal'], family, rho)
    assert meta['input_hashes'] == {k: protocol.array_sha256(v) for k, v in data.items()}
    with np.load(raw) as z:
        base, mu, sigma = z['base'], float(z['mu']), float(z['sigma'])
    A, b, ids = ref
    X, y, Q = data['X_source'], data['y_source'], data['X_target']
    cache = {} if cache is None else cache
    calls_before = len(cache)
    started = time.monotonic()

    def predict(C, d, V):
        key = tuple(protocol.array_sha256(a) for a in (C, d, V))
        if key not in cache:
            p, _ = predictor(C, d, V, members=1, random_state=0,
                             n_classes=base.shape[1] if job['task'] == 'classification' else None)
            assert np.isfinite(p).all()
            cache[key] = p
        return cache[key], key

    source_prediction, _ = predict(A, b, X)
    repaired = y.copy()
    if job['task'] == 'classification':
        proposed = source_prediction.argmax(1)
        mask = (source_prediction.max(1) >= .8) & (proposed != y)
    else:
        proposed = source_prediction
        mask = np.abs(y - proposed) > 3 * max(float(np.std(b)), 1e-8)
    repaired[mask] = proposed[mask]
    if job['task'] == 'classification':
        # Original registered constructors retain every observed source class.
        for label in np.unique(y):
            if not np.any(repaired == label):
                repaired[np.flatnonzero(y == label)[0]] = label
    common = core.main.contexts.shared_contexts(X, repaired, Q, len(y), job['task'], base)
    extra = core.main.contexts.specific_contexts(X, repaired, Q, len(y), job['task'], block, family, base)
    assert len(common) == len(extra) == 5
    reference, _ = predict(A, b, Q)
    predictions, roles = [], []
    for index, (name, C, d, V) in enumerate(common + extra):
        replaced = 0
        if MIX_RATE and index < 5:
            reference_features = A
            if name == 'feature_subspace':
                top = int(core.main.contexts.specific.importance(X, repaired.astype(float))[0])
                columns = [j for j in range(Q.shape[1]) if j != top] or [0]
                assert np.array_equal(Q[:, columns], V)
                reference_features = A[:, columns]
            elif name in ('distribution_aligned', 'aligned_target_matched'):
                _, _, center, scale = core.main.contexts.shared.geometry(X, Q)
                target_center = np.median(Q, axis=0)
                target_scale = np.maximum(np.percentile(Q, 75, axis=0) - np.percentile(Q, 25, axis=0), 1e-6)
                reference_features = (A-center)/scale*target_scale+target_center
            core.RATE = MIX_RATE
            C, d, replaced = core.replace_rows(C, d, reference_features, b, job['task'], index)
        assert len(d) <= len(y) <= 256
        p, hashes = predict(C, d, V)
        predictions.append(p)
        roles.append(dict(role=name, rows=len(d), input_hashes=hashes,
                          clean_reference_rows_inserted=replaced,
                          clean_reference_row_fraction=replaced/len(d)))
    assert meta['input_hashes'] == {k: protocol.array_sha256(v) for k, v in data.items()}
    core.main.scaling.npwrite(dest, members=np.stack(predictions), reference=reference,
                             mu=np.array(mu), sigma=np.array(sigma))
    atomic_json(dest.with_suffix('.json'), dict(
        frozen_input_hashes=meta['input_hashes'], roles=roles,
        clean_reference_examples=len(b), clean_reference_ids_hash=protocol.array_sha256(ids),
        reference_disjoint_from_calibration_and_final_queries=True,
        changed_source_labels=int(np.count_nonzero(repaired != y)),
        changed_source_fraction=float(np.mean(repaired != y)),
        repaired_label_hash=protocol.array_sha256(repaired),
        original_features_and_Base_unchanged=True, original_fusion_form=True,
        query_labels_used_by_constructors=False, additional_clean_TRAIN_label_supervision=True,
        new_unique_TFM_inputs=len(cache)-calls_before, seconds=time.monotonic()-started))


def launch():
    (OUT / 'logs').mkdir(exist_ok=True)
    workers = []
    for gpu in range(8):
        with (OUT / 'logs' / f'inference_{gpu}.log').open('a') as log:
            p = subprocess.Popen([sys.executable, '-u', __file__, 'worker', '--shard', str(gpu)],
                env=os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'TABACE_REFERENCE_BUDGET': '64'},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        workers.append(dict(gpu=gpu, pid=p.pid, shard=gpu))
    atomic_json(OUT / 'workers.json', workers)
    print(workers)


def fit_with_optional_canonical(task, canonical):
    core.OUT = OUT
    if not canonical:
        core.load_cells = few.load_cells
    else:
        # The extra canonical context is a separately reported resource variant.
        # Both its source inputs and query rows are already frozen and cached.
        def load(task, variant, role):
            cells = few.load_cells(task, variant, role)
            other = core.RUN / 'results/few_reference_contexts_v27_ref128'
            for c in cells:
                raw = Path(c['path'])
                path = other / raw.relative_to(core.RAW)
                meta = json.loads(path.with_suffix('.json').read_text())
                original = json.loads(raw.with_suffix('.json').read_text())
                assert meta['frozen_input_hashes'] == original['input_hashes']
                with np.load(path) as z:
                    c['bank'] = np.concatenate([c['bank'], z['reference'][None]], axis=0)
            return cells
        original_fit = core.fit

        def fit(cells, task, m, device):
            return original_fit(cells, task, 11 if m == 10 else m, device)
        core.load_cells = load
        core.fit = fit
    core.fit_worker(task, VARIANT + ('_canonical128' if canonical else ''))
    if not canonical:
        few.report_reference_control(task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['declare', 'worker', 'launch', 'fit'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--task', choices=['classification', 'regression'])
    parser.add_argument('--canonical128', action='store_true')
    args = parser.parse_args()
    declare()
    few.make = make
    if args.phase == 'worker':
        few.worker(args.shard, args.limit)
    elif args.phase == 'launch':
        launch()
    elif args.phase == 'fit':
        fit_with_optional_canonical(args.task, args.canonical128)
