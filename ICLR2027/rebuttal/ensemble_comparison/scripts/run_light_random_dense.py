"""Fixed light adaptation: source-fitted 1%-99% clipping, original bootstrap rows.

No target quantile transport, covariance repair, feature selection, label
adaptation or additional recent-source branch is used. Full source-pool access
matches the existing original Random/DENSE comparison and is explicitly recorded.
"""
import argparse
import hashlib
import json
import logging
import time
import traceback
import warnings
import numpy as np
import torch
from prepare_data import RUN, atomic_json, protocol
from shared_reference import specifications, load_condition
from run_tabicl import Predictor, metrics, preserving_bootstrap
from dense_full_context import candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0)
    args = p.parse_args()
    torch.set_num_threads(1)
    logging.disable(logging.INFO)
    warnings.filterwarnings('ignore')
    jobs = []
    for track in ['five_feature_calibrated','seven_reimplemented']:
        families = protocol.FAMILIES
        if track.startswith('seven'):
            from seven_shifts import FAMILIES
            families = FAMILIES
        conditions = [('original' if track.startswith('five') else 'covariate',0)] + [(f,r) for f in families for r in [20,50,80,100]]
        jobs.extend((m,f,r) for m in specifications(track) if m['task']=='classification' for f,r in conditions)
    model = Predictor('classification','cuda:0')
    work = jobs[args.shard::args.shards]
    if args.limit:
        work = work[:args.limit]
    failures = 0
    for meta,family,rho in work:
        out = RUN/'results/light_random_dense/tabicl'/meta['track']/'classification'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():
            continue
        try:
            data,info = load_condition(meta,family,rho)
            X,y,Q = (data[k] for k in ['X_source','y_source','X_target'])
            PX,py = data.get('X_pool',X),data.get('y_pool',y)
            low,high = np.quantile(X,[.01,.99],axis=0)
            A,P,T = np.clip(X,low,high),np.clip(PX,low,high),np.clip(Q,low,high)
            tick = time.perf_counter()
            bank,hashes = [],[]
            for i in range(6):
                rows = preserving_bootstrap(py,np.random.default_rng(1009*(i+1)),size=len(y))
                value,_ = model(P[rows],py[rows],T,n_classes=info['n_classes'],random_state=0)
                bank.append(value)
                hashes.append(hashlib.sha256(rows.tobytes()).hexdigest())
            bank = np.stack(bank)
            dense = candidates(bank,A,y,T,P,py)
            raw = RUN/'results/tabicl'/meta['track']/'classification'/str(meta['task_id'])/out.name
            raw_dense = RUN/'results/dense_full_context/tabicl'/meta['track']/'classification'/str(meta['task_id'])/out.name
            with np.load(raw) as f:
                base,raw_bank,truth = f['prediction__base_protocol_check'],f['bootstrap_bank'],f['y_target']
            with np.load(raw_dense) as f:
                dense_raw = f['candidates']
            assert np.array_equal(truth,data['y_target'])
            assert np.isfinite(bank).all() and np.isfinite(dense).all()
            out.parent.mkdir(parents=True,exist_ok=True)
            tmp = out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,bank=bank,dense=dense,base=base,raw_bank=raw_bank,
                raw_dense=dense_raw,y_target=truth,source_clip_low=low,source_clip_high=high)
            tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info | {
                'status':'COMPLETE','seed':0,'adaptation':'Fixed source-only 1%-99% per-feature clipping',
                'context_count':6,'rows_per_context':len(y),'accessible_source_pool_rows':len(py),
                'source_pool_access_unchanged_from_original_Random':True,
                'bootstrap_row_indices_sha256':hashes,'target_labels_used_for_preprocessing':False,
                'target_features_used_to_fit_preprocessing':False,
                'source_history_reference_extra_branch':False,'backbone_weights_frozen':True,
                'paper_Shared_or_requested_score_bounds_used':False,'Shared_run':False,
                'seconds':time.perf_counter()-tick,
                'methods':{'Random6_source_clip':metrics(truth,bank.mean(0),'classification',1.)}})
            print('LIGHT_DONE',meta['track'],meta['task_id'],family,rho,flush=True)
        except Exception as exc:
            failures += 1
            atomic_json(RUN/'results/inference_failures'/f"light_random_dense_{meta['task_id']}_{family}_{rho}.json",
                {'error':repr(exc),'traceback':traceback.format_exc()})
            print('LIGHT_FAILED',repr(exc),flush=True)
    if failures:
        raise SystemExit(1)


if __name__=='__main__':main()
