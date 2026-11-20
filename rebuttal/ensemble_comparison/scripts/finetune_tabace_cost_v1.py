"""Synchronized warm inference profiling and recovery of first-pass failures."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import finetune_tabace_main_v1 as main
import finetune_tabace_reference_v3 as ref
from prepare_data import RUN,atomic_json
from run_tabicl import Predictor

OUT=RUN/'results/finetune_tabace_cost_v1'


def worker(shard,shards):
    torch.set_num_threads(1)
    # Recover exactly the six first-pass class-coverage failures. All completed
    # original datasets remain unchanged; the corrected implementation is logged.
    for job in main.jobs()[shard::shards]:
        if not (main.folder(job)/'completion.json').exists():
            try:
                pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
                split=main.prepare_split(job,pool);main.train(job,pool,split);main.evaluate(job,pool,split)
                atomic_json(main.folder(job)/'retry_implementation.json',dict(reason='single-class query subsample could not instantiate label-prior shift',
                    corrective_sampling='source-support preserving query class coverage; no benchmark change',
                    source_validation_only=True,final_query_labels_used=False,script_sha256=main.sha(main.__file__)))
            except Exception:
                print(traceback.format_exc(),flush=True)
    original_root=main.OUT
    for job in main.jobs()[shard::shards]:
        path=OUT/'profiles'/job['task']/job['track']/f"{job['meta']['task_id']}.json"
        if path.exists():continue
        try:
            main.OUT=ref.ROOTS[job['task']]
            base=Predictor(job['task'],'cuda:0');updated=main.UpdatedPredictor(job)
            if job['task']=='classification':
                pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
                global_n_classes=int(main.flat(pool)[1].max())+1
            else:global_n_classes=None
            grid=[('clean','original',0)] if job['track']=='five' else []
            grid += [(job['track'],family,50) for family in main.families(job['track'])]
            # All registered mechanisms, three warm repetitions, fixed inputs.
            rows=[]
            for block,family,rho in grid:
                data,info=main.scaling.bank(job['task'],block,job['meta'],family,rho)
                X,y,Q=(data[k] for k in ['X_source','y_source','X_target'])
                nc=global_n_classes
                for label,predictor in [('Base',base),('FT',updated)]:
                    predictor(X,y,Q,members=1,n_classes=nc,random_state=0)
                    times=[];torch.cuda.reset_peak_memory_stats()
                    for _ in range(3):
                        torch.cuda.synchronize();start=time.perf_counter()
                        p,_=predictor(X,y,Q,members=1,n_classes=nc,random_state=0)
                        torch.cuda.synchronize();times.append(time.perf_counter()-start)
                    rows.append(dict(task=job['task'],track=job['track'],block=block,task_id=job['meta']['task_id'],family=family,
                        rho=rho,method=label,context_rows=len(y),query_rows=len(Q),repetitions=3,
                        median_seconds=float(np.median(times)),mean_seconds=float(np.mean(times)),raw_seconds=times,
                        CUDA_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
            atomic_json(path,dict(status='COMPLETE',warmup_per_case=1,model_and_checkpoint_load_excluded=True,
                same_arrays_between_models=True,seed=0,rows=rows))
            del base,updated;gc.collect();torch.cuda.empty_cache()
            print('COST_PROFILE_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
        except Exception:
            print(traceback.format_exc(),flush=True)
        finally:main.OUT=original_root


def report():
    rows=[]
    for p in (OUT/'profiles').glob('*/*/*.json'):rows.extend(json.loads(p.read_text())['rows'])
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(OUT/'profile_detail.csv',index=False)
    summary=df.groupby(['task','track','method']).agg(datasets=('task_id','nunique'),conditions=('family','size'),
        mean_seconds=('median_seconds','mean'),median_seconds=('median_seconds','median'),peak_MiB=('CUDA_peak_allocated_MiB','max')).reset_index()
    summary.to_csv(OUT/'profile_averages.csv',index=False);print(summary.to_string(index=False),flush=True)


def launch_after_v3():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    pending={int(w['gpu']):int(w['pid']) for w in json.loads((ref.OUT/'workers.json').read_text())}
    children=[]
    while pending:
        for gpu,oldpid in list(pending.items()):
            try:os.kill(oldpid,0);continue
            except ProcessLookupError:pass
            log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
            p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
                env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();children.append(dict(gpu=gpu,pid=p.pid));atomic_json(OUT/'workers.json',children)
            print('COST_WORKER_STARTED',gpu,p.pid,flush=True);del pending[gpu]
        if pending:time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','report','launch_after_v3']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='worker':worker(a.shard,a.shards)
    elif a.phase=='report':report()
    else:launch_after_v3()
