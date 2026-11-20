"""Base-only INTERNAL benchmark calibration, with all trials retained.

Changing this benchmark is a development exercise, not a reproduction of the
submitted paper's protocol. No ensemble or historical Shared result selects it.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import warnings

import numpy as np

from prepare_data import RUN, ROOT, atomic_json, protocol
from seven_shifts import registry, FAMILIES as SEVEN, materialize as seven_materialize, classification_clean
import internal_tabpfn_regression as previous

OUT = RUN / 'results/tabpfn_official_comparison_v2'
PLAN = RUN / 'configs/tabpfn_official_comparison_v2.json'
TARGETS = {
    'classification': {'clean': {'accuracy': .793, 'auc': .893}, 'five': {'accuracy': .457, 'auc': .636}, 'seven': {'accuracy': .758, 'auc': .809}},
    'regression': previous.TARGETS,
}
PRIMARY = {'classification': 'accuracy', 'regression': 'r2'}
_CACHE = {}


def tasks(task, track):
    if track == 'seven':
        return [m for m in registry() if m['task'] == task]
    return [json.loads(p.read_text()) for p in sorted((RUN / 'cache/five' / task).glob('*.json'))]


def declare():
    assert not PLAN.exists()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = {
        'classification': {'clean': [{'name': 'existing', 'kind': 'cached'}],
            'five': [{'name': 'feature12', 'feature_multiplier': 12}, {'name': 'feature16', 'feature_multiplier': 16}],
            'seven': [{'name': 'amplitude0.8', 'amplitude': .8}]},
        'regression': {'clean': [{'name': 'pristine1024', 'source_cap': 1024}, {'name': 'pristine512', 'source_cap': 512}],
            'five': [{'name': 'shift1_label0.8', 'shift_scale': 1., 'label_scale': .8}],
            'seven': [{'name': 'existing', 'kind': 'previous_frozen'}]},
    }
    atomic_json(PLAN, {'status': 'DECLARED_BEFORE_NEW_PREDICTIONS', 'internal_development_only': True,
        'paper_protocol_reproduction': False, 'seed': 0, 'outer_repetitions': 1,
        'nominal_strengths': [.2, .5, .8, 1.], 'targets': TARGETS,
        'primary_metric_allowed_signed_gap': [-.1, .003],
        'threshold_interpretation': 'Acc/R2 may be at most .003 above and .1 below historical paper Base; aim nearer',
        'secondary_metrics_reported_without_hard_threshold': True,
        'candidate_configs': settings, 'selection': 'Base-only nearest primary, secondary gap used only to break ties',
        'ensemble_outcomes_or_Shared_bounds_select_benchmark': False,
        'cross_backbone_benchmarks_differ': True, 'all_trials_retained': True,
        'methods': ['Native6', 'PHE6_official_portfolio', 'RF_PFN6_official', 'MixturePFN_MICP_component_classification', 'OligoICP_regression'],
        'MixturePFN_full_CaPFN_claimed_without_checkpoint_or_training': False,
        'OligoICP_original_reference': 'https://arxiv.org/html/2510.02476v2',
        'OligoICP_candidate_count': 400, 'OligoICP_retained_count': 40,
        'OligoICP_six_retained_budget_variant_from_same_bank': [60, 6],
        'OligoICP_generic_grouping_adaptation': '20 source-only KMeans groups; choose k uniformly from 1..20, then k groups without replacement',
        'adaptation_selection': 'source TRAIN validation only; never use benchmark query scores or historical Shared as a selection rule',
        'extra_clean_TRAIN_supervision_authorized': True, 'Markdown_written': False})
    old = previous.OUT
    atomic_json(OUT / 'previous_result_lock.json', {str(p.relative_to(RUN)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [old / 'latest_summary.json', old / 'averages.csv', old / 'prediction_audit.json']})
    print('TABPFN_OFFICIAL_INTERNAL_PLAN_DECLARED', flush=True)


def configs(task, track):
    return json.loads(PLAN.read_text())['candidate_configs'][task][track]


def canonical(task, meta, family, rho, settings, track):
    if task == 'regression' and track == 'seven':
        data, info, _ = previous.frozen_input('seven', meta, family, rho)
        return data, info | {'new_protocol_uses_previous_seven': True}
    if task == 'classification' and track == 'seven':
        key = ('seven', meta['task_id'])
        if key not in _CACHE:
            _CACHE[key] = classification_clean(meta)
        class_count = int(np.max(np.r_[_CACHE[key][1], _CACHE[key][3]])) + 1
        effective = int(round(rho * settings['amplitude']))
        data, info = seven_materialize(meta, family, effective, _CACHE[key])
        info = info | {'effective_operator_rho': effective, 'operator_amplitude_multiplier': settings['amplitude']}
    else:
        key = (task, meta['task_id'])
        if key not in _CACHE:
            with np.load(RUN / 'cache/five' / task / f"{meta['task_id']}.npz") as z:
                _CACHE[key] = {k: z[k] for k in ['X_source', 'y_source', 'X_target', 'y_target']}
        data = {k: v.copy() for k, v in _CACHE[key].items()}
        if task == 'classification':
            class_count = int(np.max(_CACHE[key]['y_source'])) + 1
        info = dict(meta)
        if task == 'regression':
            clean = settings if track == 'clean' else json.loads((OUT / 'frozen_regression_clean.json').read_text())['settings']
            ids = np.random.default_rng(0).permutation(len(data['y_source']))[:clean['source_cap']]
            data['X_source'], data['y_source'] = data['X_source'][ids], data['y_source'][ids]
            pristine_scale = max(float(data['y_source'].std()), 1e-12)
            if track == 'five' and rho:
                changed = protocol.manipulate_source_context(task, data['X_source'], data['y_source'], family=family,
                    rho=rho / 100., seed=1729 + rho, shift_scale=settings['shift_scale'], label_scale=settings['label_scale'])
                data['X_source'], data['y_source'] = changed.X_context, changed.y_context
        elif track == 'five' and rho:
            from sklearn.ensemble import ExtraTreesClassifier
            from scaled_historical_feature_protocol import apply_feature_scaled_shift
            teacher_key = ('teacher', meta['task_id'])
            if teacher_key not in _CACHE:
                _CACHE[teacher_key] = ExtraTreesClassifier(n_estimators=128, random_state=0, n_jobs=1).fit(data['X_source'], data['y_source'])
            pseudo = _CACHE[teacher_key].predict(data['X_target'])
            kind = 'aucband' if family == 'band' else family
            name = f"publicshift_d{meta['data_id']}_{kind}_rho{rho}_beta500"
            data['X_target'], details = apply_feature_scaled_shift(name, data['X_source'], data['y_source'],
                data['X_target'], pseudo, 0, feature_multiplier=settings['feature_multiplier'])
            info['operator_details'] = details
    data = {k: data[k] for k in ['X_source', 'y_source', 'X_target', 'y_target']}
    scale = pristine_scale if task == 'regression' and track != 'seven' else max(float(data['y_target' if track == 'seven' else 'y_source'].std()), 1e-12)
    return data, info | {'task': task, 'task_id': meta['task_id'], 'dataset': meta['dataset'],
        'track': track, 'family': family, 'rho': rho, 'nominal_strength': rho / 100.,
        'settings': settings, 'scale': scale, 'n_classes': class_count if task == 'classification' else None,
        'internal_development_only': True, 'paper_protocol_reproduction': False, 'benchmark_seed': 0}


def conditions(track):
    if track == 'clean':
        return [('original', 0)]
    return [(f, r) for f in (protocol.FAMILIES if track == 'five' else SEVEN) for r in [20, 50, 80, 100]]


def path_for(task, track, config, meta, family, rho):
    return OUT / 'base_calibration' / task / track / config / str(meta['task_id']) / f'{family}_{rho}.npz'


def run(task, track, shard, shards, limit=0):
    import torch
    from run_tabpfn import model, fit_predict
    from run_tabicl import metrics
    torch.set_num_threads(1)
    logging.disable(logging.INFO)
    warnings.filterwarnings('ignore')
    est = model(task, 'cuda:0')
    work = tasks(task, track)[shard::shards]
    if limit:
        work = work[:limit]
    for meta in work:
        for setting in configs(task, track):
            for family, rho in conditions(track):
                path = path_for(task, track, setting['name'], meta, family, rho)
                if path.exists() and path.with_suffix('.json').exists():
                    continue
                data, info = canonical(task, meta, family, rho, setting, track)
                pred = fit_predict(est, data['X_source'], data['y_source'], data['X_target'], task, info.get('n_classes'))
                assert np.isfinite(pred).all()
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.tmp.npz')
                np.savez_compressed(temp, prediction=pred, y_target=data['y_target'])
                temp.replace(path)
                atomic_json(path.with_suffix('.json'), info | {'status': 'COMPLETE', 'model_seed': 0,
                    'metrics': metrics(data['y_target'], pred, task, info['scale']),
                    'input_hashes': {k: protocol.array_sha256(v) for k, v in data.items()}})
                print('PFN_NEW_BASE_DONE', task, track, setting['name'], meta['dataset'], family, rho, flush=True)


def freeze(task, track):
    key = PRIMARY[task]
    keys = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
    expected = len(tasks(task, track)) * len(conditions(track))
    candidates = []
    for cfg in configs(task, track):
        paths = list((OUT / 'base_calibration' / task / track / cfg['name']).glob('*/*.json'))
        assert len(paths) == expected, (task, track, cfg['name'], len(paths), expected)
        rows = [json.loads(p.read_text()) for p in paths]
        avg = {k: float(np.mean([r['metrics'][k] for r in rows if r['metrics'][k] is not None])) for k in keys}
        gaps = {k: avg[k] - TARGETS[task][track][k] for k in keys}
        candidates.append({'settings': cfg, 'metrics': avg, 'gap_to_paper': gaps,
            'primary_bound_met': -.1 <= gaps[key] <= .003, 'conditions': expected})
    candidates.sort(key=lambda c: (not c['primary_bound_met'], abs(c['gap_to_paper'][key])))
    selected = candidates[0]
    atomic_json(OUT / f'calibration_{task}_{track}.json', {'all_trials': candidates, 'selection_uses_Base_only': True})
    assert selected['primary_bound_met'], ('No eligible candidate; diagnose before expanding calibration', task, track, selected)
    path = OUT / f'frozen_{task}_{track}.json'
    assert not path.exists()
    atomic_json(path, selected | {'status': 'FROZEN', 'internal_development_only': True,
        'historical_Shared_used_for_selection': False, 'paper_protocol_reproduction': False})
    print('PFN_NEW_BENCHMARK_FROZEN', task, track, json.dumps(selected), flush=True)


def frozen_input(task, track, meta, family, rho):
    clean_track = 'clean' if rho == 0 and track == 'five' else track
    frozen = json.loads((OUT / f'frozen_{task}_{clean_track}.json').read_text())
    setting = frozen['settings']
    data, info = canonical(task, meta, family, rho, setting, clean_track)
    path = path_for(task, clean_track, setting['name'], meta, family, rho)
    old = json.loads(path.with_suffix('.json').read_text())
    assert old['input_hashes'] == {k: protocol.array_sha256(v) for k, v in data.items()}
    with np.load(path) as z:
        assert np.array_equal(z['y_target'], data['y_target'])
        base = z['prediction']
    return data, info | {'track': track}, base


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['declare', 'run', 'freeze'])
    parser.add_argument('--task', choices=['classification', 'regression'])
    parser.add_argument('--track', choices=['clean', 'five', 'seven'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.phase == 'declare':
        declare()
    elif args.phase == 'run':
        run(args.task, args.track, args.shard, args.shards, args.limit)
    else:
        freeze(args.task, args.track)
