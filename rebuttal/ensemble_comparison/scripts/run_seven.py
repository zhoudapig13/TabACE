"""Evaluate explicitly reconstructed seven shifts; one shared clean per dataset."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
import numpy as np
import torch
from prepare_data import RUN,atomic_json
from run_tabicl import Predictor,run_cell
from seven_shifts import FAMILIES,PARAMETERS,registry


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true')
    p.add_argument('--task',choices=['both','classification','regression'],default='both')
    args=p.parse_args();torch.set_num_threads(1)
    manifest=[m for m in registry() if args.task in ['both',m['task']]][args.shard::args.shards]
    if args.limit:manifest=manifest[:args.limit]
    expected=hashlib.sha256(json.dumps(PARAMETERS,sort_keys=True).encode()).hexdigest()
    predictors={};failures=0
    for m in manifest:
        task=m['task'];task_id=m['task_id']
        conditions=[('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]
        if args.smoke:conditions=conditions[:1]
        for family,rho in conditions:
            source=RUN/'cache/seven_reimplemented'/task/str(task_id)/f'{family}_{rho}.npz'
            meta=json.loads(source.with_suffix('.json').read_text())
            assert meta['parameter_sha256']==expected,'stale reconstruction'
            out=RUN/'results/tabicl/seven_reimplemented'/task/str(task_id)/f'{family}_{rho}.npz'
            if out.exists() and out.with_suffix('.json').exists():continue
            print('START',task,task_id,family,rho,flush=True)
            try:
                if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
                with np.load(source) as data:run_cell(predictors[task],data,meta,family,rho,out)
                print('DONE',task,task_id,family,rho,flush=True)
            except Exception as exc:
                failures+=1
                info={'task':task,'task_id':task_id,'family':family,'rho':rho,
                      'error':repr(exc),'traceback':traceback.format_exc()}
                atomic_json(RUN/'results/inference_failures'/f'seven_tabicl_{task}_{task_id}_{family}_{rho}.json',info)
                print('FAILED',json.dumps(info),flush=True);torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
