"""Reference/hierarchical extensions for the frozen SOURCE-only v5 protocol."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import finetune_source_tabace_v5 as source
import finetune_tabace_main_v1 as main
import finetune_tabace_reference_v3 as reference
import finetune_tabace_hierarchical_v4 as hierarchy
from prepare_data import RUN,atomic_json

REFERENCE_OUT=RUN/'results/finetune_source_reference_v6'
HIERARCHY_OUT=RUN/'results/finetune_source_hierarchical_v7'
SELECTED=False
FINAL=False


def setup():
    source.setup();reference.OUT=REFERENCE_OUT;reference.ROOTS['classification']=source.OUT
    hierarchy.OUT=HIERARCHY_OUT
    if FINAL:
        main.jobs=source._jobs
        reference.ROOTS['regression']=source.OUT


def plan(stage):
    dest=REFERENCE_OUT if stage=='reference' else HIERARCHY_OUT
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'plan.json').exists():return
    atomic_json(dest/'plan.json',dict(status='DECLARED_BEFORE_THIS_EXTENSION_PREDICTIONS_OR_POLICY_SELECTION',
        stage=stage,task='both' if FINAL else 'classification',track='all' if FINAL else 'five',datasets=175 if FINAL else 72,context_cap=256,seed=0,
        source_context_corruption_only='classification/five' if FINAL else True,
        query_features_and_labels_unchanged='classification/clean_five; other tracks retain v22' if FINAL else True,
        regression_clean_five_semisynthetic=bool(FINAL),
        benchmark_sha256=main.sha(source.benchmark.OUT/'frozen.json'),
        weights=f'same actual source-corruption-matched checkpoint in {source.OUT} in every method',
        reference_labels='clean labels already accessible in backbone FT TRAIN partition',
        Reference1_control_included=True,reference_pool_cap=1024,
        constructor_library='five reference Shared roles; Specific includes these and original five observed-context family-specific roles',
        hierarchy='canonical clean reference replaces bootstrap; source-held-out selection of observed/reference bank; residual Specific refinement of Shared' if stage=='hierarchy' else None,
        heldout_original_row_ID_grouping_for_hierarchy=True,
        user_increment_objective='prefer FT→TabACE gain comparable to or larger than Base→FT; report actual values',
        final_query_labels_used_for_training_selection_or_calibration=False,
        development_adaptation=True,not_archived_paper_replication=True,
        script_sha256=main.sha(__file__),time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),Markdown_written=False))


def worker(stage,shard,shards):
    if stage=='reference':reference.worker(shard,shards)
    else:hierarchy.worker(shard,shards)


def report(stage):
    source.OUT=REFERENCE_OUT if stage=='reference' else HIERARCHY_OUT
    source.report()


def launch(stage):
    parent=source.OUT if stage=='reference' else REFERENCE_OUT
    if FINAL and stage=='reference':
        while not (parent/'completion.json').exists():time.sleep(2)
        parent=RUN/'results/finetune_remaining_strong_v12'
    elif SELECTED and stage=='reference':
        while not (parent/'completion.json').exists():time.sleep(2)
        parent=RUN/'results/finetune_source_strong_v8'
    while True:
        if (parent/'workers.json').exists():
            waiting=json.loads((parent/'workers.json').read_text())
            if len(waiting)==8:break
        time.sleep(2)
    dest=REFERENCE_OUT if stage=='reference' else HIERARCHY_OUT
    plan(stage);(dest/'logs').mkdir(exist_ok=True);children=[]
    for gpu in range(8):
        old=next(w for w in waiting if w['gpu']==gpu)
        while True:
            try:os.kill(old['pid'],0);time.sleep(2)
            except ProcessLookupError:break
        log=(dest/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        command=[sys.executable,'-u',__file__,'worker','--stage',stage,'--shard',str(gpu),'--shards','8']
        if SELECTED:command.append('--selected')
        if FINAL:command.append('--final')
        process=subprocess.Popen(command,
            env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(dict(gpu=gpu,pid=process.pid));atomic_json(dest/'workers.json',children)
        print('SOURCE_EXTENSION_WORKER_STARTED',stage,gpu,process.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','report','launch'])
    p.add_argument('--stage',choices=['reference','hierarchy'],default='reference');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--selected',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args()
    if a.selected:
        SELECTED=True;source.OUT=RUN/'results/finetune_source_selected_v9'
        REFERENCE_OUT=RUN/'results/finetune_source_selected_reference_v10'
        HIERARCHY_OUT=RUN/'results/finetune_source_selected_hierarchical_v11'
    if a.final:
        FINAL=True;SELECTED=False;source.OUT=RUN/'results/finetune_final_selected_v13'
        REFERENCE_OUT=RUN/'results/finetune_final_reference_v14'
        HIERARCHY_OUT=RUN/'results/finetune_final_hierarchical_v15'
    setup()
    if a.phase=='plan':plan(a.stage)
    elif a.phase=='worker':worker(a.stage,a.shard,a.shards)
    elif a.phase=='report':report(a.stage)
    else:launch(a.stage)
