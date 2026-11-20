"""Fresh frozen-backbone predictions validate all recalibrated seven-shift inputs."""
import argparse,json,logging,time,traceback,warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from seven_shifts import FAMILIES
from run_tabicl import Predictor,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    model=Predictor('regression','cuda:0');failures=0
    conditions=[('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]
    jobs=[(m,f,r) for m in specifications('seven_calibrated_v2') for f,r in conditions][a.shard::a.shards]
    for meta,family,rho in jobs:
        out=RUN/'results/seven_regression_v2_fresh_base'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():continue
        cached=RUN/'results/calibrated_base/seven_calibrated_v2/regression'/str(meta['task_id'])/out.name
        if rho:
            while not cached.with_suffix('.json').exists():time.sleep(2)
        try:
            data,info=load_condition(meta,family,rho)
            for name,array in data.items():
                if name+'_sha256' in info:assert protocol.array_sha256(array)==info[name+'_sha256'],name
            prediction,_=model(data['X_source'],data['y_source'],data['X_target'])
            scale=max(float(data['y_target'].std()),1e-12);observed=metrics(data['y_target'],prediction,'regression',scale)
            difference=0.
            if rho:
                with np.load(cached) as f:previous=f['prediction'];assert np.array_equal(f['y_target'],data['y_target'])
                difference=float(np.max(np.abs(prediction-previous)))/scale
                assert difference<1e-5,('fresh inference differs',difference)
            out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=prediction,y_target=data['y_target']);tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','method':'Base','backbone':'tabicl',
                'metrics':observed,'fresh_backbone_inference':True,'scaled_max_cached_prediction_difference':difference,
                'target_labels_used_for_inference':False,'seed':0})
            print('VERIFIED_V2',meta['task_id'],family,rho,observed,flush=True)
        except Exception as exc:
            failures+=1;print('VERIFY_V2_FAILED',meta['task_id'],family,rho,repr(exc),flush=True)
            atomic_json(RUN/'results/inference_failures'/f"seven_v2_verify_{meta['task_id']}_{family}_{rho}.json",
                {'error':repr(exc),'traceback':traceback.format_exc()})
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
