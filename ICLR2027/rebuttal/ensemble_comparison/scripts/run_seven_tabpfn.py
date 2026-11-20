"""TabPFN PHE, bagging and unadapted MICP on reconstructed seven shifts."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import traceback
import warnings
import numpy as np
import torch
from run_tabpfn import cell
from seven_shifts import FAMILIES,PARAMETERS,registry
from prepare_data import RUN,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--smoke',action='store_true');p.add_argument('--rf',action='store_true')
    p.add_argument('--task',choices=['both','classification','regression'],default='both')
    args=p.parse_args();torch.set_num_threads(1);logging.disable(logging.INFO)
    warnings.filterwarnings('ignore',category=FutureWarning)
    warnings.filterwarnings('ignore',message='The y_pred values do not sum to one')
    manifest=[m for m in registry() if args.task in ['both',m['task']]][args.shard::args.shards]
    if args.limit:manifest=manifest[:args.limit]
    parameter_hash=hashlib.sha256(json.dumps(PARAMETERS,sort_keys=True).encode()).hexdigest()
    failures=0
    for m in manifest:
        task=m['task'];task_id=m['task_id']
        conditions=[('covariate',0)]+[(f,r) for f in FAMILIES for r in [20,50,80,100]]
        if args.smoke:conditions=conditions[:1]
        for family,rho in conditions:
            source=RUN/'cache/seven_reimplemented'/task/str(task_id)/f'{family}_{rho}.npz'
            meta=json.loads(source.with_suffix('.json').read_text())
            assert meta['parameter_sha256']==parameter_hash
            out=RUN/'results/tabpfn/seven_reimplemented'/task/str(task_id)/f'{family}_{rho}.npz'
            if out.exists() and out.with_suffix('.json').exists():continue
            print('START',task,task_id,family,rho,flush=True)
            try:
                with np.load(source) as data:cell(task,data,meta,family,rho,out,'cuda:0',args.rf)
                print('DONE',task,task_id,family,rho,flush=True)
            except Exception as exc:
                failures+=1;row={'task':task,'task_id':task_id,'family':family,'rho':rho,
                    'error':repr(exc),'traceback':traceback.format_exc()}
                atomic_json(RUN/'results/inference_failures'/f'seven_tabpfn_{task}_{task_id}_{family}_{rho}.json',row)
                print('FAILED',json.dumps(row),flush=True)
            torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
