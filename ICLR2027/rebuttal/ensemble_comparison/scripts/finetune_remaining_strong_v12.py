"""Second, stronger FT candidate for regression and seven-shift classification."""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
import torch
import finetune_tabace_main_v1 as main
import finetune_tabace_main_v2 as balanced
from prepare_data import RUN,atomic_json

OUT=RUN/'results/finetune_remaining_strong_v12'
BASE_JOBS=main.jobs
TRAINER=main.train
LOSS=main.loss_for
VALIDATE=main.validate
QUERY=main.query_sample


def jobs():return [j for j in BASE_JOBS() if j['task']=='regression' or j['track']=='seven']


def worker(shard,shards):
    torch.set_num_threads(1)
    for job in jobs()[shard::shards]:
        try:
            if job['task']=='classification':
                balanced.setup();lr=3e-6
            else:
                main.loss_for=LOSS;main.validate=VALIDATE;main.query_sample=QUERY;main.CLASS_QUERY_MODE='uniform';lr=1e-5
            main.OUT=OUT
            pool=main.scaling.prepare(job['task'],job['track'],job['meta']);split=main.prepare_split(job,pool)
            TRAINER(job,pool,split,steps=512,lr=lr)
            main.evaluate(job,pool,split)
            print('STRONG_REMAINING_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            atomic_json(main.folder(job)/'error.json',dict(job=job,error=error))
            gc.collect();torch.cuda.empty_cache()


def launch():
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_STRONG_REMAINING_WEIGHT_UPDATES',seed=0,context_cap=256,
        datasets=len(jobs()),regression_datasets=68,classification_seven_datasets=35,steps=512,
        regression_lr=1e-5,classification_lr=3e-6,training_seconds_cap_per_dataset=150,
        benchmark_unchanged=True,regression_validation_same_as_v1=True,classification_validation_same_as_v2=True,
        final_query_labels_or_scores_for_checkpoint_selection=False,all_candidates_preserved=True,
        purpose='fully optimized FT control; candidate selection on identical source validation partitions',
        Markdown_written=False,script_sha256=main.sha(__file__)))
    parent=RUN/'results/finetune_source_selected_reference_v10'
    while True:
        if (parent/'workers.json').exists():
            waiting=json.loads((parent/'workers.json').read_text())
            if len(waiting)==8:break
        time.sleep(2)
    while waiting:
        left=[]
        for w in waiting:
            try:os.kill(w['pid'],0);left.append(w)
            except ProcessLookupError:pass
        waiting=left
        if waiting:time.sleep(2)
    children=[];(OUT/'logs').mkdir(exist_ok=True)
    for gpu in range(8):
        log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
            env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(dict(gpu=gpu,pid=p.pid));atomic_json(OUT/'workers.json',children)
        print('STRONG_REMAINING_WORKER_STARTED',gpu,p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','launch']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='worker':worker(a.shard,a.shards)
    else:launch()
