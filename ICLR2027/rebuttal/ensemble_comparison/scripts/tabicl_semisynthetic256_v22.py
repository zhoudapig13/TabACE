"""Explicit semisynthetic regression development track on real covariates.

Teacher uses original TRAIN labels only. Query labels come from the same frozen
teacher/noise law, never from TabICL predictions. All runs use 256 contexts and
the unchanged official TabICL-v2 single estimator.
"""
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

OUT = base.OUT / 'semisynthetic_regression'
PROFILES = [(mu, h) for mu in [.2, .35, .5, .65] for h in [0., .6, 1.]] + [(0., 0.)]


def name(mu, h): return f'noise{mu:g}_heterogeneity{h:g}'


def declare():
    path = OUT / 'plan.json'
    assert not path.exists()
    atomic_json(path, {
        'status': 'DECLARED_BEFORE_NEW_SEMISYNTHETIC_PREDICTIONS',
        'user_authorized_separate_semisynthetic_Clean_development_task': True,
        'paper_protocol_reproduction': False, 'seed': 0, 'n_estimators': 1, 'context_rows': 256,
        'real_covariates': 'unchanged original official TRAIN/TEST partitions, source cap 2048 and existing query cap',
        'feature_projection': 'up to 32 source-label-correlated nonconstant columns; source-standardized and clipped to [-5,5]',
        'teacher': 'Ridge alpha=1 on five most source-correlated selected features, fit on original TRAIN pool only',
        'signal_standardization': 'source teacher mean/std only',
        'label_law': 'standardized teacher signal + mu * exp(h * fixed_dataset_normal_draw) * independent standard Gaussian noise',
        'identical_train_and_query_noise_law': True,
        'noise_profiles': PROFILES, 'original_query_labels_used_to_construct_or_fit_teacher': False,
        'TabICL_predictions_used_to_construct_labels': False,
        'metric_nrmse_scale': 'actual semisynthetic source-population std, ddof0; not adjustable',
        'initial_profile_budget': 13, 'maximum_refinement_profiles': 12,
        'selection_uses_development_Base_metrics': True, 'all_trials_preserved': True,
        'targets': {'r2': .739, 'nrmse': .476}, 'absolute_tolerance': .005,
        'original_results_preserved': True, 'Shared_Specific_not_run': True,
        'script_sha256': base.sha(Path(__file__)), 'Markdown_written': False,
    })


def prepare(meta):
    path = OUT / 'generators' / f'{meta["task_id"]}.npz'
    if path.exists():
        with np.load(path) as z: return {k: z[k] for k in z.files}
    from sklearn.linear_model import Ridge
    data, _ = base.clean_pool(meta)
    X = data['X_source'].astype(np.float64); Q = data['X_target'].astype(np.float64)
    y = data['y_source'].astype(np.float64)
    mean = X.mean(0); sd = np.maximum(X.std(0), 1e-6)
    Z = np.clip((X-mean)/sd, -5, 5); V = np.clip((Q-mean)/sd, -5, 5)
    target = (y-y.mean()) / max(float(y.std()), 1e-8)
    correlations = np.abs((Z-Z.mean(0)).T @ target) / np.maximum(np.linalg.norm(Z-Z.mean(0), axis=0), 1e-8)
    valid = np.flatnonzero(Z.std(0) > 1e-8)
    assert len(valid), 'Semisynthetic teacher requires a nonconstant real covariate'
    ranked = valid[np.argsort(correlations[valid], kind='stable')[::-1]]
    selected = np.sort(ranked[:min(32, len(ranked))])
    active = ranked[:min(5, len(ranked))]
    teacher = Ridge(alpha=1.).fit(Z[:,active], target)
    signal, query_signal = teacher.predict(Z[:,active]), teacher.predict(V[:,active])
    center, scale = float(signal.mean()), max(float(signal.std()), 1e-8)
    signal = (signal-center)/scale; query_signal = (query_signal-center)/scale
    streams = np.random.SeedSequence([0, int(meta['task_id']), 2201]).spawn(3)
    eps_source = np.random.default_rng(streams[0]).normal(size=len(y))
    eps_query = np.random.default_rng(streams[1]).normal(size=len(Q))
    noise_draw = float(np.random.default_rng(streams[2]).normal())
    ids = base.context_indices(X,y,'random7')
    arrays = {'X_source_pool': Z[:,selected].astype(np.float32), 'X_query': V[:,selected].astype(np.float32),
              'signal_source': signal, 'signal_query': query_signal, 'epsilon_source':eps_source,
              'epsilon_query':eps_query, 'dataset_noise_draw':np.asarray(noise_draw),
              'context_indices':ids, 'source_row_ids':data['source_indices'], 'query_row_ids':data['target_indices']}
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**arrays);tmp.replace(path)
    atomic_json(path.with_suffix('.json'), {'task_id':meta['task_id'],'dataset':meta['dataset'],
        'selected_original_columns':selected.tolist(),'teacher_active_original_columns':active.tolist(),
        'teacher_coefficients':teacher.coef_.tolist(),'teacher_intercept':float(teacher.intercept_),
        'original_feature_center':mean.tolist(),'original_feature_scale':sd.tolist(),
        'signal_center':center,'signal_scale':scale,'dataset_noise_draw':noise_draw,
        'original_query_labels_used':False,'TabICL_predictions_used_for_labels':False,
        'source_query_row_ids_disjoint':not bool(np.intersect1d(data['source_indices'],data['target_indices']).size)})
    return arrays


def materialize(meta, mu, h):
    data=prepare(meta)
    sigma=float(mu*np.exp(h*float(data['dataset_noise_draw'])))
    source=data['signal_source']+sigma*data['epsilon_source']
    truth=data['signal_query']+sigma*data['epsilon_query']
    ids=data['context_indices']
    return {'X_source':data['X_source_pool'][ids],'y_source':source[ids],
            'X_target':data['X_query'],'y_target':truth},max(float(source.std()),1e-12),sigma


def worker(shard,shards,profile_file=None):
    import torch
    import warnings
    from run_tabicl import Predictor,metrics,CHECKPOINTS
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0')
    profiles=PROFILES if profile_file is None else [tuple(x) for x in json.loads(Path(profile_file).read_text())['profiles']]
    for meta in base.manifest()[shard::shards]:
        prepare(meta)
        for mu,h in profiles:
            path=OUT/'clean_profiles'/name(mu,h)/f'{meta["task_id"]}.npz'
            if path.exists() and path.with_suffix('.json').exists():continue
            data,scale,sigma=materialize(meta,mu,h)
            assert len(data['y_source'])==256
            p,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,random_state=0)
            assert np.isfinite(p).all()
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=p,y_target=data['y_target']);tmp.replace(path)
            atomic_json(path.with_suffix('.json'),{'task_id':meta['task_id'],'profile':name(mu,h),
                'noise_mu':mu,'noise_heterogeneity':h,'actual_noise_sigma':sigma,'context_rows':256,
                'seed':0,'n_estimators':1,'scale':scale,'metrics':metrics(data['y_target'],p,'regression',scale),
                'input_hashes':{k:protocol.array_sha256(v) for k,v in data.items()},
                'checkpoint_sha256':base.sha(CHECKPOINTS/'tabicl-regressor-v2-20260212.ckpt'),
                'semi_synthetic_development_task':True,'original_paper_replication':False})
        print('SEMISYNTHETIC_CLEAN_DATASET_COMPLETE',meta['task_id'],flush=True)


def launch(profile_file=None):
    def one(gpu):
        env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        cmd=[str(RUN/'.venv/bin/python'),'-u',str(Path(__file__)),'worker','--shard',str(gpu),'--shards','8']
        tag='initial' if profile_file is None else Path(profile_file).stem
        if profile_file is not None:cmd+=['--profile-file',str(profile_file)]
        with (RUN/'logs'/f'tabicl_semisynthetic256_v22_{tag}_gpu{gpu}.log').open('a') as f:
            p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        row={'gpu':gpu,'exit_code':p.returncode};print(json.dumps(row),flush=True);return row
    with ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(one,range(8)))
    atomic_json(OUT/('initial_execution.json' if profile_file is None else Path(profile_file).stem+'_execution.json'),rows)
    if any(r['exit_code'] for r in rows):raise SystemExit(1)


def report():
    import pandas as pd
    rows=[]
    for folder in (OUT/'clean_profiles').iterdir():
        paths=[p for p in folder.glob('*.json') if not p.name.endswith('.tmp.json')]
        if len(paths)!=35:continue
        data=[json.loads(p.read_text()) for p in paths]
        avg={k:float(np.mean([m['metrics'][k] for m in data])) for k in ['r2','nrmse']}
        dif={'r2':avg['r2']-.739,'nrmse':avg['nrmse']-.476}
        rows.append({'profile':folder.name,'mu':data[0]['noise_mu'],'h':data[0]['noise_heterogeneity'],**avg,
                     'within_005_each':all(abs(v)<=.005 for v in dif.values()),
                     'distance':sum((v/.005)**2 for v in dif.values())})
    rows.sort(key=lambda x:(not x['within_005_each'],x['distance']))
    pd.DataFrame(rows).to_csv(OUT/'clean_averages_all_trials.csv',index=False)
    atomic_json(OUT/'clean_selection.json',{'all_trials':rows,'best':rows[0],
                'selection_uses_development_Base_metrics':True,'is_original_task_reproduction':False})
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['declare','worker','launch','report'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--profile-file');a=p.parse_args()
    if a.phase=='declare':declare()
    elif a.phase=='worker':worker(a.shard,a.shards,a.profile_file)
    elif a.phase=='launch':launch(a.profile_file)
    else:report()
