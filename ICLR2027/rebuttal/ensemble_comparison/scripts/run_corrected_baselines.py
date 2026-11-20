"""Run all affected classification baselines after the polarity definition fix."""
import argparse
import json
import logging
import traceback
import warnings
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from corrected_source_protocol import reuse_unchanged_clean


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn','tabm'],default='tabicl')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    if args.backbone=='tabicl':
        from run_tabicl import Predictor,run_cell
        predictor=Predictor('classification','cuda:0')
    elif args.backbone=='tabpfn':from run_tabpfn import cell
    else:from run_tabm import run_one
    jobs=specifications('five_corrected')
    if args.limit:jobs=jobs[:args.limit]
    failures=0
    for job_index,meta in enumerate(jobs):
        conditions=[('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        if args.smoke:conditions=[('shortcut',100)]
        for condition_index,(family,rho) in enumerate(conditions):
            if (job_index*len(conditions)+condition_index)%args.shards!=args.shard:continue
            out=RUN/'results'/args.backbone/'five_corrected/classification'/str(meta['task_id'])/f'{family}_{rho}.npz'
            if out.with_suffix('.json').exists():continue
            if rho==0:
                old=RUN/'results'/args.backbone/'five/classification'/str(meta['task_id'])/'original_0.npz'
                if reuse_unchanged_clean(old,out):continue
            print('START',args.backbone,meta['task_id'],family,rho,flush=True)
            try:
                arrays,info=load_condition(meta,family,rho)
                if args.backbone=='tabicl':run_cell(predictor,arrays,info,family,rho,out)
                elif args.backbone=='tabpfn':cell('classification',arrays,info,family,rho,out,'cuda:0',True)
                else:run_one(arrays,info,out)
                print('DONE',args.backbone,meta['task_id'],family,rho,flush=True)
            except Exception as exc:
                failures+=1;row=meta|{'family':family,'rho':rho,'error':repr(exc),'traceback':traceback.format_exc()}
                atomic_json(RUN/'results/inference_failures'/f"corrected_{args.backbone}_{meta['task_id']}_{family}_{rho}.json",row)
                print('FAILED',json.dumps(row),flush=True);torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
