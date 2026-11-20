"""Warm Base/selected-FT profiling on the final frozen development protocols."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import finetune_tabace_main_v1 as main
import finetune_source_tabace_v5 as source
import finetune_tabace_reference_v3 as reference
import finetune_tabace_cost_v1 as cost
from prepare_data import RUN,atomic_json

OUT=RUN/'results/finetune_final_profile_v16'
SELECTED=RUN/'results/finetune_final_selected_v13'
REFERENCE=RUN/'results/finetune_final_reference_v14'


def setup():
    main.OUT=SELECTED;main.jobs=source._jobs;main.scaling.bank=source.bank
    reference.ROOTS['classification']=SELECTED;reference.ROOTS['regression']=SELECTED;cost.OUT=OUT


def launch():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_FINAL_PROFILING',seed=0,context_cap=256,
        selected_checkpoint_root=str(SELECTED),same_input_arrays_Base_and_FT=True,
        classification_five='frozen source-only v5; clean queries unchanged',other_tracks='frozen v22',
        warmup=1,timed_repetitions=3,grid='Clean plus each registered mechanism at rho=.5',
        model_load_excluded=True,FT_and_Base_models_both_resident_during_peak_memory_measurement=True,
        no_weight_updates_or_metric_selection=True,Markdown_written=False))
    while True:
        if (REFERENCE/'workers.json').exists():
            waiting=json.loads((REFERENCE/'workers.json').read_text())
            if len(waiting)==8:break
        time.sleep(2)
    while waiting:
        remaining=[]
        for w in waiting:
            try:os.kill(w['pid'],0);remaining.append(w)
            except ProcessLookupError:pass
        waiting=remaining
        if waiting:time.sleep(2)
    children=[]
    for gpu in range(8):
        log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
            env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(dict(gpu=gpu,pid=p.pid));atomic_json(OUT/'workers.json',children)
        print('FINAL_PROFILE_WORKER_STARTED',gpu,p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','launch','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    setup()
    if a.phase=='worker':cost.worker(a.shard,a.shards)
    elif a.phase=='report':cost.report()
    else:launch()
