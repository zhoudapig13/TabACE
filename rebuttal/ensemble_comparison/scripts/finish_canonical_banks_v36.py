"""Redistribute remaining canonical query cases without changing the estimator.

Only unfinished cases from old logical shards2/3 are eligible. Each case has
one deterministic owner; existing complete banks are immutable. Regression is
scheduled first to release completed task blocks while image cases continue.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'1')
import numpy as np
import tabace_context_adaptation_v29 as experiment
import source_constructor_memo_v35 as memo
from prepare_data import atomic_json
from run_tabicl import Predictor


def owner(key,count):return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%count


def jobs():
    return sorted([(t,tr,m) for i,(t,tr,m) in enumerate(experiment.raw.jobs()) if i%8 in [2,3]],
                  key=lambda row:(row[0]!='regression',row[1],int(row[2]['task_id'])))


def worker(index,count,limit=0):
    import torch,warnings
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    enabled=memo.install_if_verified(experiment)
    print('CANONICAL_CASE_SCHEDULER',index,count,'exact_source_memo',enabled,flush=True)
    predictors={};completed=0;failures=[]
    for task,track,meta in jobs():
        pool=None
        for budget in [1024,4096,8192]:
            for block in ['clean','five'] if track=='five' else ['seven']:
                for family,rho in experiment.raw.case_grid(block):
                    relative=Path(str(budget))/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                    key=str(relative)
                    if owner(key,count)!=index:continue
                    out=experiment.OUT/'banks/query'/relative
                    if out.exists() and out.with_suffix('.json').exists():continue
                    try:
                        if pool is None:pool=experiment.raw.scaling.prepare(task,track,meta)
                        rawpath=experiment.raw.OUT/'banks/query'/relative
                        assert rawpath.exists() and rawpath.with_suffix('.json').exists()
                        data,*_=experiment.raw.query_input(task,block,meta,family,rho,budget,pool)
                        if task not in predictors:predictors[task]=Predictor(task,'cuda:0')
                        experiment.run_case(predictors[task],data,rawpath,out,task,block,meta,family,rho,budget)
                        completed+=1
                        print('CANONICAL_CASE_COMPLETE',key,flush=True)
                        if limit and completed>=limit:return
                    except Exception:
                        error=traceback.format_exc();print(error,flush=True);failures.append(dict(case=key,error=error))
        print('CANONICAL_DATASET_VISITED',task,track,meta['dataset'],flush=True)
    atomic_json(experiment.OUT/'redistribution_v36/completion'/f'{index}.json',
        dict(status='COMPLETE' if not failures else 'FAILED',worker=index,new_cases=completed,failures=failures))
    if failures:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);parser.add_argument('--count',type=int,required=True);parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    worker(args.index,args.count,args.limit)
