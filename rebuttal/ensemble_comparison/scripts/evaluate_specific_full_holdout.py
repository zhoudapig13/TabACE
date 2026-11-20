"""Independent-query validation of the full declared five-classification pool."""
import argparse,hashlib,json,logging,time,traceback,warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import Predictor,PFNPredictor,contexts
from specific_reference import specific_contexts
from specific_full_registry import extra_contexts
from run_specific_full_pool import signature
from evaluate_benchmark_tuning_holdout import cell
from feature_calibrated_protocol import materialize
from fit_benchmark_specific_uncapped import predict
from run_tabicl import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    track='five_feature_calibrated';root=RUN/'results/benchmark_specific_uncapped_full'/a.backbone/track/'classification'
    while len(list(root.glob('specific_*.json')))<5:time.sleep(5)
    policies={f.stem[len('specific_'):]:json.loads(f.read_text()) for f in root.glob('specific_*.json')}
    manifest=json.loads((RUN/'configs/benchmark_tuning_holdout_manifest.json').read_text())
    jobs=[(m,f,r) for m in manifest['classification'] if m['holdout_status']=='READY'
        for f in protocol.FAMILIES for r in [20,50,80,100]][a.shard::a.shards]
    predictor=Predictor('classification','cuda:0') if a.backbone=='tabicl' else PFNPredictor('classification');failures=0
    for meta,family,rho in jobs:
        out=RUN/'results/benchmark_specific_full_holdout'/a.backbone/track/'classification'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():continue
        source=RUN/'results/benchmark_tuning_holdout'/a.backbone/track/'classification'/str(meta['task_id'])/out.name
        while not source.with_suffix('.json').exists():time.sleep(5)
        try:
            with np.load(RUN/'cache/tuning_holdout/classification'/f"{meta['task_id']}.npz") as f:
                Q,t,ids=f['X_target'],f['y_target'],f['query_indices']
            data,info=materialize(meta,family,rho,query=(Q,t));X,y,T=data['X_source'],data['y_source'],data['X_target'];budget=len(y)
            with np.load(source) as f:base=f['prediction__Base'];sb=f['shared_bank'];db=f['specific_bank'];assert np.array_equal(f['y_target'],t)
            first=contexts(X,y,T,X,y,budget,'classification',base=base)+specific_contexts(X,y,T,X,y,budget,'classification',family=family,track=track,base=base)
            bank=list(sb)+list(db);cache={signature(view):prediction for view,prediction in zip(first,bank)}
            for view in extra_contexts(X,y,T,budget,'classification',family):
                key=signature(view)
                if key in cache:prediction=cache[key]
                else:
                    _,A,b,V=view;prediction,_=predictor(A,b,V,n_classes=info['n_classes']);cache[key]=prediction
                bank.append(prediction)
            bank=np.stack(bank);policy=policies[family];assert len(bank)==policy['candidate_count']
            c=cell(info,base,bank,t,0.,1.);prediction=predict(c,np.asarray(policy['parameters']),'classification',policy['gain'])
            observed=metrics(t,prediction,'classification',1.);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=prediction,y_target=t,query_indices=ids);tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','backbone':a.backbone,'method':'TabACE_Specific_full_declared_pool',
                'metrics':observed,'context_count_cap':None,'candidate_count':len(bank),'unique_contexts':len(cache),
                'query_labels_used_for_fit_or_selection':False,'results_role':'new queries on seen development datasets',
                'policy_sha256':hashlib.sha256((root/f'specific_{family}.json').read_bytes()).hexdigest()})
            print('FULL_HOLDOUT_DONE',a.backbone,meta['task_id'],family,rho,len(bank),len(cache),flush=True)
        except Exception as exc:
            failures+=1;atomic_json(RUN/'results/inference_failures'/f"full_holdout_{a.backbone}_{meta['task_id']}_{family}_{rho}.json",
                {'error':repr(exc),'traceback':traceback.format_exc()});print('FULL_HOLDOUT_FAILED',repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
