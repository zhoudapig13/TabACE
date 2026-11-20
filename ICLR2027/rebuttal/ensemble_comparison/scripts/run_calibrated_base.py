"""Verify the frozen stress protocol on the original benchmark queries."""
import argparse,json,logging,warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from seven_shifts import FAMILIES
from run_tabicl import Predictor,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    jobs=[]
    for track in ['five_calibrated','seven_calibrated']:
        families=protocol.FAMILIES if track.startswith('five') else FAMILIES
        conditions=[('original' if track.startswith('five') else 'covariate',0)]+[(f,r) for f in families for r in [20,50,80,100]]
        jobs += [(m,f,r) for m in specifications(track) for f,r in conditions]
    predictors={};failures=0
    for meta,family,rho in jobs[args.shard::args.shards]:
        out=RUN/'results/calibrated_base'/meta['track']/meta['task']/str(meta['task_id'])/f'{family}_{rho}.json'
        if out.exists():continue
        try:
            data,info=load_condition(meta,family,rho);task=meta['task']
            if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
            prediction,_=predictors[task](data['X_source'],data['y_source'],data['X_target'],n_classes=info['n_classes'])
            scale=max(float(data['y_target'].std()),1e-12) if meta['track'].startswith('seven') else info['scale']
            out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.tmp.npz')
            np.savez_compressed(temp,prediction=prediction,y_target=data['y_target']);temp.replace(out.with_suffix('.npz'))
            atomic_json(out,info|{'status':'COMPLETE','method':'Base','seed':0,
                'metrics':metrics(data['y_target'],prediction,task,scale),'benchmark_queries':'development under user-authorized tuning'})
            print('CALIBRATED_BASE',meta['track'],meta['task_id'],family,rho,flush=True)
        except Exception as exc:failures+=1;print('FAILED_CALIBRATED_BASE',meta['task_id'],family,rho,repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
