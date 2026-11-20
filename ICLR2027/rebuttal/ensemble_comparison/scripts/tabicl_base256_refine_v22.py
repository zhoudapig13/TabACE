"""Predeclared development refinements; all numerical trials retained."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from prepare_data import RUN, atomic_json, protocol
import tabicl_base256_calibration_v22 as base

OUT = base.OUT
PREPROCESS = ['power', 'quantile_normal', 'pca32', 'pca64', 'source_clip']
CLASS_PROFILES = [('beta450_features10', 4.5, 10), ('beta400_features12', 4., 12),
                  ('beta500_features10', 5., 10)]


def declare():
    path = OUT / 'refinement_plan.json'
    assert not path.exists()
    atomic_json(path, {
        'status': 'DECLARED_AFTER_PHASE1_BEFORE_NEW_PREDICTIONS',
        'phase1_summary_sha256': base.sha(OUT / 'clean_profile_summary.json'),
        'regression_context_profile': 'random7', 'preprocessing': PREPROCESS,
        'preprocessing_fit_data': 'only the selected 256 source feature rows',
        'additional_query_support_fractions': [.5, .75],
        'support_rule': 'nearest half/three-quarters of queries by source-standardized mean five-neighbor distance; no query labels in selection',
        'support_reference': 'random7 source context; labels and metric denominators unchanged',
        'classification_five_profiles': CLASS_PROFILES, 'outer_seed': 0,
        'classification_context_rows': 256, 'regression_context_rows': 256,
        'native_estimators': 1, 'all_trials_preserved': True,
        'Shared_Specific_not_run': True, 'model_weights_unchanged': True,
        'all_profile_selection_is_development_only_not_heldout_evaluation': True,
    })


def preprocess(X, Q, name):
    X, Q = X.astype(np.float64), Q.astype(np.float64)
    if name == 'source_clip':
        lo, hi = np.quantile(X, [.01, .99], axis=0)
        return np.clip(X, lo, hi), np.clip(Q, lo, hi)
    if name == 'quantile_normal':
        from sklearn.preprocessing import QuantileTransformer
        fit = QuantileTransformer(n_quantiles=64, output_distribution='normal', random_state=0, subsample=None)
        return fit.fit_transform(X), fit.transform(Q)
    if name == 'power':
        from sklearn.preprocessing import PowerTransformer
        A, T = X.copy(), Q.copy()
        for j in range(X.shape[1]):
            if np.ptp(X[:, j]) == 0: continue
            try:
                fit = PowerTransformer().fit(X[:, j:j+1])
                a, t = fit.transform(X[:, j:j+1]), fit.transform(Q[:, j:j+1])
                if np.isfinite(a).all() and np.isfinite(t).all(): A[:, j], T[:, j] = a[:, 0], t[:, 0]
            except (ValueError, RuntimeError, FloatingPointError):
                pass
        return A, T
    from sklearn.decomposition import PCA
    dimensions = int(name.removeprefix('pca'))
    if X.shape[1] <= dimensions: return X, Q
    center = X.mean(0); scale = np.maximum(X.std(0), 1e-6)
    fit = PCA(n_components=dimensions, svd_solver='randomized', random_state=0).fit((X-center)/scale)
    return fit.transform((X-center)/scale), fit.transform((Q-center)/scale)


def clean_worker(shard, shards):
    import torch
    import warnings
    from run_tabicl import Predictor, metrics
    from sklearn.metrics import pairwise_distances
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    warnings.filterwarnings('ignore')
    predictor = Predictor('regression', 'cuda:0')
    for m in base.manifest()[shard::shards]:
        data, scale = base.clean_pool(m)
        ids = base.context_indices(data['X_source'], data['y_source'], 'random7')
        X, y, Q = data['X_source'][ids], data['y_source'][ids], data['X_target']
        for name in PREPROCESS:
            path = OUT / 'clean_preprocessing' / name / f'{m["task_id"]}.npz'
            if path.exists() and path.with_suffix('.json').exists(): continue
            A, T = preprocess(X, Q, name)
            prediction, _ = predictor(A, y, T, members=1, random_state=0)
            assert np.isfinite(prediction).all()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, prediction=prediction, y_target=data['y_target'],
                                context_indices=ids, query_row_ids=data['target_indices'])
            tmp.replace(path)
            atomic_json(path.with_suffix('.json'), {'task_id': m['task_id'], 'profile': name,
                        'context_rows': 256, 'seed': 0, 'n_estimators': 1, 'scale': scale,
                        'source_only_feature_preprocessing': True,
                        'metrics': metrics(data['y_target'], prediction, 'regression', scale),
                        'input_hashes': {k: protocol.array_sha256(v) for k,v in
                            [('X_source', A), ('y_source', y), ('X_target', T), ('y_target', data['y_target'])]}})
        # These two label-blind query-cohort changes are new benchmark variants.
        center = np.median(X, axis=0)
        width = np.maximum(np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0), 1e-6)
        distances = pairwise_distances(np.clip((Q-center)/width, -10, 10),
                                      np.clip((X-center)/width, -10, 10), metric='sqeuclidean')
        score = np.partition(distances, 4, axis=1)[:, :5].mean(1)
        with np.load(OUT / 'clean_profiles/random7' / f'{m["task_id"]}.npz') as z:
            existing = z['prediction']
        for fraction in [.5, .75]:
            name = f'feature_support_{fraction:g}'
            selected = np.argsort(score, kind='stable')[:max(32, int(np.ceil(len(Q)*fraction)))]
            path = OUT / 'clean_query_support' / name / f'{m["task_id"]}.npz'
            if path.exists(): continue
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, prediction=existing[selected], y_target=data['y_target'][selected],
                                query_indices=selected, query_row_ids=data['target_indices'][selected], context_indices=ids)
            atomic_json(path.with_suffix('.json'), {'task_id': m['task_id'], 'profile': name,
                        'context_rows': 256, 'seed': 0, 'query_selection_reads_labels': False,
                        'query_rows': len(selected), 'scale': scale,
                        'metrics': metrics(data['y_target'][selected], existing[selected], 'regression', scale)})
        print('CLEAN_REFINEMENT_COMPLETE', m['task_id'], flush=True)


def class_worker(shard, shards):
    import torch
    import warnings
    from run_tabicl import Predictor, metrics
    from sklearn.ensemble import ExtraTreesClassifier
    from scaled_historical_feature_protocol import apply_feature_scaled_shift
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    warnings.filterwarnings('ignore')
    predictor = Predictor('classification', 'cuda:0')
    tasks = [json.loads(p.read_text()) for p in sorted((RUN / 'cache/five/classification').glob('*.json'))]
    for m in tasks[shard::shards]:
        with np.load(RUN / 'cache/five/classification' / f'{m["task_id"]}.npz') as z:
            X, y, Q, truth = (z[k] for k in ['X_source','y_source','X_target','y_target'])
        assert len(y) == 256
        pseudo = ExtraTreesClassifier(n_estimators=128, random_state=0, n_jobs=1).fit(X,y).predict(Q)
        for name, beta, feature_count in CLASS_PROFILES:
            for family in protocol.FAMILIES:
                for rho in [20,50,80,100]:
                    path = OUT / 'classification_five_profiles' / name / str(m['task_id']) / f'{family}_{rho}.npz'
                    if path.exists() and path.with_suffix('.json').exists(): continue
                    kind = 'aucband' if family == 'band' else family
                    op = f'publicshift_d{m["data_id"]}_{kind}_rho{rho}_beta{int(beta*100)}'
                    shifted, _ = apply_feature_scaled_shift(op, X,y,Q,pseudo,0,feature_multiplier=feature_count)
                    p, _ = predictor(X,y,shifted,members=1,random_state=0,n_classes=len(np.unique(y)))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix('.tmp.npz'); np.savez_compressed(tmp,prediction=p,y_target=truth); tmp.replace(path)
                    atomic_json(path.with_suffix('.json'), {'task_id':m['task_id'],'family':family,'rho':rho,
                        'context_rows':256,'seed':0,'beta':beta,'feature_multiplier':feature_count,
                        'generator_reads_query_labels':False,'metrics':metrics(truth,p,'classification',1.),
                        'input_hashes':{k:protocol.array_sha256(v) for k,v in
                            [('X_source',X),('y_source',y),('X_target',shifted),('y_target',truth)]}})
        print('CLASS_FIVE_PROFILE_DATASET_COMPLETE',m['task_id'],flush=True)


def launch(phase):
    def one(gpu):
        env = os.environ | {'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        with (RUN/'logs'/f'tabicl_base256_v22_{phase}_gpu{gpu}.log').open('a') as f:
            p=subprocess.run([str(RUN/'.venv/bin/python'),'-u',str(Path(__file__)),phase,
                              '--shard',str(gpu),'--shards','8'],env=env,stdout=f,stderr=subprocess.STDOUT)
        row={'gpu':gpu,'exit_code':p.returncode};print(json.dumps(row),flush=True);return row
    with ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(one,range(8)))
    atomic_json(OUT/f'{phase}_execution.json',rows)
    if any(r['exit_code'] for r in rows): raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['declare','clean_worker','class_worker','launch_clean','launch_class'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='declare':declare()
    elif a.phase.startswith('launch_'):launch(a.phase.removeprefix('launch_')+'_worker')
    elif a.phase=='clean_worker':clean_worker(a.shard,a.shards)
    else:class_worker(a.shard,a.shards)
