"""Paired conservative resampling control, with identical policy on both TFMs.

Use each condition's canonical labelled context; replace a fixed 10% of row
positions, predict with six frozen members. No history/reference rows are added.
The already-complete TabICL seven-shift control is reused instead of rerun.
"""
import argparse
import json
import logging
import time
import traceback
import warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from run_tabicl import Predictor,metrics
from run_canonical_regression import fingerprint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    jobs=[]
    tracks=['five'] if args.backbone=='tabicl' else ['five','seven_calibrated_v2']
    for track in tracks:
        families=protocol.FAMILIES
        if track.startswith('seven'):
            from seven_shifts import FAMILIES
            families=FAMILIES
        conditions=[('original' if track=='five' else 'covariate',0)]+[(f,r) for f in families for r in [20,50,80,100]]
        jobs.extend((m,f,r) for m in specifications(track) if m['task']=='regression' for f,r in conditions)
    jobs=jobs[args.shard::args.shards]
    if args.limit:jobs=jobs[:args.limit]
    predictor=None;failures=0
    for meta,family,rho in jobs:
        out=RUN/'results/regression_partial_matched'/args.backbone/meta['track']/'regression'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():continue
        try:
            data,info=load_condition(meta,family,rho)
            X,y,Q=(data[k] for k in ['X_source','y_source','X_target'])
            if predictor is None:
                if args.backbone=='tabicl':predictor=Predictor('regression','cuda:0')
                else:
                    from run_tabpfn import model
                    predictor=model('regression','cuda:0')
            bank=[];iqrs=[];indices=[];unique=[];tick=time.perf_counter()
            for i in range(6):
                rng=np.random.default_rng(1009*(i+1));rows=np.arange(len(y))
                positions=rng.choice(len(y),size=int(np.ceil(.1*len(y))),replace=False)
                rows[positions]=rng.choice(len(y),size=len(positions),replace=True)
                if args.backbone=='tabicl':
                    p,iqr=predictor(X[rows],y[rows],Q,quantiles=True)
                    iqrs.append(iqr)
                else:
                    from run_tabpfn import fit_predict
                    predictor.random_state=0
                    p=fit_predict(predictor,X[rows],y[rows],Q,'regression',None)
                bank.append(p);indices.append(fingerprint(rows));unique.append(len(np.unique(rows)))
            bank=np.stack(bank)
            raw=RUN/'results'/args.backbone/meta['track']/'regression'/str(meta['task_id'])/out.name
            key='base_protocol_check' if args.backbone=='tabicl' else 'tabpfn_v3_base_check'
            with np.load(raw) as f:base=f['prediction__'+key];truth=f['y_target']
            assert np.array_equal(truth,data['y_target'])
            assert np.isfinite(bank).all()
            mean=bank.mean(0)
            scale=max(float(truth.std()),1e-12) if meta['track'].startswith('seven') else info['scale']
            arrays={'bootstrap_bank':bank,'prediction__bootstrap_mean_6_partial010':mean,
                'prediction__base_protocol_check':base,'y_target':truth,
                'source_y_mean':np.asarray(float(y.mean())),
                'source_y_std':np.asarray(max(float(y.std()),1e-12))}
            if iqrs:arrays['bootstrap_iqr']=np.stack(iqrs)
            out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**arrays);tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info | {
                'status':'COMPLETE','backbone':args.backbone,'seed':0,
                'family':family,'rho':rho,'context_count':6,'rows_per_context':len(y),
                'resampling_fraction':.1,'original_full_pool_rows':len(data.get('y_pool',y)),
                'source_access':'Exact canonical labelled context; no extra history/reference rows',
                'bootstrap_indices_sha256':indices,'unique_source_rows_per_member':unique,
                'current_source_X_sha256':fingerprint(X),'current_source_y_sha256':fingerprint(y),
                'current_query_X_sha256':fingerprint(Q),'base_reused_from':str(raw.relative_to(RUN)),
                'target_labels_used_for_prediction_or_context_construction':False,
                'backbone_weights_frozen':True,'score_ordering_bounds_used':False,'Shared_run':False,
                'seconds':time.perf_counter()-tick,'methods':{
                    'base_protocol_check':metrics(truth,base,'regression',scale),
                    'bootstrap_mean_6_partial010':metrics(truth,mean,'regression',scale)}})
            print('MATCHED_DONE',args.backbone,meta['track'],meta['task_id'],family,rho,flush=True)
        except Exception as exc:
            failures+=1
            atomic_json(RUN/'results/inference_failures'/f"partial_matched_{args.backbone}_{meta['track']}_{meta['task_id']}_{family}_{rho}.json",
                {'error':repr(exc),'traceback':traceback.format_exc()})
            print('MATCHED_FAILED',repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
