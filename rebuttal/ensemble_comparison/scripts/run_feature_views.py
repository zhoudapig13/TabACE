"""Six source-fitted feature views for the explicitly adapted TabFM strategy."""
from __future__ import annotations
import argparse
import json
import shutil
import time
import traceback
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from run_tabicl import Predictor,metrics


def projections(X,Q):
    yield 'identity',X,Q
    center=X.mean(0);scale=np.maximum(X.std(0),1e-6)
    Z=np.clip((X-center)/scale,-8,8);V=np.clip((Q-center)/scale,-8,8)
    for k in [16,32,64]:
        if min(X.shape)<=k:yield f'svd{k}_identity_small_dimension',X,Q
        else:
            transform=TruncatedSVD(n_components=k,n_iter=3,random_state=0).fit(Z)
            yield f'svd{k}',transform.transform(Z).astype(np.float32),transform.transform(V).astype(np.float32)
    for count in [32,64]:
        if X.shape[1]<2:yield f'cross{count}_identity_single_feature',X,Q
        else:
            rng=np.random.default_rng(0);a=rng.integers(X.shape[1],size=count);b=rng.integers(X.shape[1]-1,size=count)
            b=b+(b>=a)
            yield f'cross{count}',np.column_stack([Z,Z[:,a]*Z[:,b]]).astype(np.float32),np.column_stack([V,V[:,a]*V[:,b]]).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--track',choices=['both','five','five_corrected','five_calibrated','five_feature_calibrated','seven_calibrated','seven_calibrated_v2','seven_reimplemented'],default='both');args=p.parse_args()
    torch.set_num_threads(1);predictors={};failures=0
    tracks=['five','seven_reimplemented'] if args.track=='both' else [args.track]
    jobs=[(track,m) for track in tracks for m in specifications(track)]
    if args.limit:jobs=jobs[:args.limit]
    for job_index,(track,meta) in enumerate(jobs):
        if track.startswith('five'):conditions=[('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        else:
            from seven_shifts import FAMILIES
            conditions=[('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]
        if args.smoke:conditions=conditions[:1]
        task=meta['task']
        if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
        for condition_index,(family,rho) in enumerate(conditions):
            if (job_index*len(conditions)+condition_index)%args.shards!=args.shard:continue
            out=RUN/'results/feature_views/tabicl'/track/task/str(meta['task_id'])/f'{family}_{rho}.npz'
            if out.exists() and out.with_suffix('.json').exists():continue
            if track=='five_corrected' and rho==0:
                from corrected_source_protocol import reuse_unchanged_clean
                old=RUN/'results/feature_views/tabicl/five'/task/str(meta['task_id'])/'original_0.npz'
                if reuse_unchanged_clean(old,out):continue
            print('START',track,task,meta['task_id'],family,rho,flush=True)
            try:
                data,info=load_condition(meta,family,rho)
                if rho==0 and track in ['five_feature_calibrated','seven_calibrated_v2']:
                    previous_track='five' if track.startswith('five') else 'seven_reimplemented'
                    previous=RUN/'results/feature_views/tabicl'/previous_track/task/str(meta['task_id'])/out.name
                    if previous.with_suffix('.json').exists():
                        previous_info=json.loads(previous.with_suffix('.json').read_text())
                        for key in ['X_source','y_source','X_target','y_target']:
                            assert previous_info[key+'_sha256']==protocol.array_sha256(data[key]),key
                        out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(previous,out)
                        atomic_json(out.with_suffix('.json'),previous_info|info|{'unchanged_clean_predictions_reused_from':str(previous.relative_to(RUN))})
                        continue
                X,y,Q,t=(data[k] for k in ['X_source','y_source','X_target','y_target'])
                original=RUN/'results/tabicl'/track/task/str(meta['task_id'])/f'{family}_{rho}.npz'
                while not original.with_suffix('.json').exists():time.sleep(2)
                with np.load(original) as f:base=f['prediction__base_protocol_check']
                values=[];names=[];widths=[];seconds=[];start=time.perf_counter()
                for name,A,T in projections(X,Q):
                    if 'identity' in name:pred=base;elapsed=0.
                    else:
                        torch.cuda.synchronize();tick=time.perf_counter()
                        pred,_=predictors[task](A,y,T,n_classes=info['n_classes'])
                        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                    values.append(pred);names.append(name);widths.append(A.shape[1]);seconds.append(elapsed)
                bank=np.stack(values)
                if not np.isfinite(bank).all():raise ValueError('nonfinite feature view')
                evaluation_scale=max(float(t.std()),1e-12) if task=='regression' and track.startswith('seven') else info['scale']
                mean=bank.mean(0);observed=metrics(t,mean,task,evaluation_scale)|{'seconds':time.perf_counter()-start}
                out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,bank=bank,y_target=t,source_mean=np.asarray(float(y.mean())),
                    source_scale=np.asarray(max(float(y.std()),1e-9)),prediction__Feature_view_mean6_adapted=mean)
                tmp.replace(out)
                atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','backbone':'tabicl','seed':0,
                    'views':names,'view_features':widths,'prediction_seconds':seconds,
                    'nonidentity_view_calls':sum('identity' not in name for name in names),
                    'methods':{'Feature_view_mean6_adapted':observed},'source_fitted_transformations':True,
                    'target_labels_used_for_projection':False,'NNLS_fit':'separate dataset-level OOF',
                    'original_TabFM_evaluated':False,'adaptations':'TabICL backbone; six views rather than original 32; dataset-OOF NNLS/calibration'})
                print('DONE',track,task,meta['task_id'],family,rho,flush=True)
            except Exception as exc:
                failures+=1;row=meta|{'family':family,'rho':rho,'error':repr(exc),'traceback':traceback.format_exc()}
                atomic_json(RUN/'results/inference_failures'/f"feature_views_{track}_{task}_{meta['task_id']}_{family}_{rho}.json",row)
                print('FAILED',json.dumps(row),flush=True);torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
