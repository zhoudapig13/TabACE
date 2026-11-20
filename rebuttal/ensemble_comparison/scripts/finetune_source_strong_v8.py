"""Strong actual FT control on the SAME frozen source-only benchmark."""
import argparse
import json
import os
import subprocess
import sys
import time
import finetune_source_tabace_v5 as source
import finetune_tabace_main_v1 as main
import finetune_tabace_main_v2 as balanced
from prepare_data import RUN,atomic_json

OUT=RUN/'results/finetune_source_strong_v8'
TRAINER=balanced.base_train


def train(job,pool,split,steps=512,lr=1e-5):
    return TRAINER(job,pool,split,steps=steps,lr=lr)


def setup():
    source.setup();main.OUT=OUT;main.train=train


def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_STRONGER_WEIGHT_UPDATES',seed=0,context_cap=256,datasets=72,
        source_context_corruption_only=True,query_features_and_labels_unchanged=True,
        benchmark_sha256=main.sha(source.benchmark.OUT/'frozen.json'),
        initial_candidate=dict(root=str(source.OUT),steps=128,learning_rate=3e-6),
        stronger_candidate=dict(steps=512,learning_rate=1e-5,training_seconds_cap_per_dataset=150),
        selection='same disjoint source validation, 40% clean balanced NLL +60% corruption balanced NLL',
        final_query_labels_or_scores_used_for_checkpoint_selection=False,
        extra_clean_TRAIN_labels=True,unfinetuned_Shared_Specific_not_run=True,
        reason='provide a more fully optimized matched-corruption FT baseline, retain both real candidates',
        developmental=True,not_archived_paper_replication=True,script_sha256=main.sha(__file__),Markdown_written=False))


def report():
    source.OUT=OUT;source.report()


def launch():
    parent=RUN/'results/finetune_source_reference_v6'
    while True:
        if (parent/'workers.json').exists():
            waiting=json.loads((parent/'workers.json').read_text())
            if len(waiting)==8:break
        time.sleep(2)
    # Wait for existing GPU inference before timing/training this candidate.
    while waiting:
        left=[]
        for w in waiting:
            try:os.kill(w['pid'],0);left.append(w)
            except ProcessLookupError:pass
        waiting=left
        if waiting:time.sleep(2)
    plan();children=[];(OUT/'logs').mkdir(exist_ok=True)
    for gpu in range(8):
        log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
            env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(dict(gpu=gpu,pid=p.pid));atomic_json(OUT/'workers.json',children)
        print('STRONG_FT_WORKER_STARTED',gpu,p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report','launch']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    setup()
    if a.phase=='plan':plan()
    elif a.phase=='worker':main.worker(a.shard,a.shards,steps=512)
    elif a.phase=='report':report()
    else:launch()
