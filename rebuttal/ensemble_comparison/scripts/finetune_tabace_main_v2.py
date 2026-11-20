"""Predeclared classification refinement: class-balanced robust fine-tuning.

The first-pass results remain untouched. Source-only class-balanced query
episodes and checkpoint validation address class-prior mismatch and forgetting;
the benchmark is unchanged. No final query labels choose this refinement.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import finetune_tabace_main_v1 as main
from prepare_data import atomic_json

BASE_OUT=main.OUT
OUT=BASE_OUT.with_name('finetune_tabace_main_v2')
base_jobs=main.jobs
base_train=main.train
base_tensors=main.fusion.training_tensors


def jobs():
    return [j for j in base_jobs() if j['task']=='classification']


def balanced_queries(eligible,y,size,rng,preferred=None):
    eligible=np.asarray(eligible,dtype=int)
    # Each class retains at least one in-context support observation.
    groups=[]
    for label in np.unique(y[eligible]):
        choices=eligible[y[eligible]==label]
        choices=choices[1:]
        if preferred is not None:
            choices=np.r_[rng.permutation(np.intersect1d(choices,preferred)),
                rng.permutation(np.setdiff1d(choices,preferred))]
        else:choices=rng.permutation(choices)
        groups.append(list(choices))
    result=[];n=min(max(size,len(groups)),sum(map(len,groups)))
    while len(result)<n:
        for group in groups:
            if group and len(result)<n:result.append(group.pop(0))
    return np.asarray(result,dtype=int)


def loss(model,b,task):
    assert task=='classification'
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(b['X'].cuda(),b['y'].cuda()).float()[...,:b['classes']]
    t=b['t'].cuda().reshape(-1)
    per_query=F.cross_entropy(out.reshape(-1,b['classes']),t,reduction='none')
    return torch.stack([per_query[t==label].mean() for label in torch.unique(t)]).mean()


def validate(model,batches,task):
    main.train_mode(model)
    with torch.no_grad():
        values=[float(loss(model,b,task)) for b in batches if b is not None]
    # First case is Clean; its protected 40% share discourages forgetting.
    return .4*values[0]+.6*float(np.mean(values[1:])) if len(values)>1 else float(np.mean(values))


def train(job,pool,split,steps=128,lr=3e-6):
    return base_train(job,pool,split,steps=steps,lr=lr)


def balanced_fusion_tensors(cells,m):
    result=base_tensors(cells,m)
    factors=[]
    for c in cells:
        y=c['y'].astype(int);labels,counts=np.unique(y,return_counts=True)
        lookup={int(k):len(y)/(len(labels)*n) for k,n in zip(labels,counts)}
        factors.extend(lookup[int(k)] for k in y)
    result['weights']*=torch.as_tensor(factors,dtype=result['weights'].dtype)
    return result


def setup():
    main.OUT=OUT;main.jobs=jobs;main.query_sample=balanced_queries
    main.CLASS_QUERY_MODE='class_balanced_v2'
    main.loss_for=loss;main.validate=validate;main.train=train
    main.fusion.training_tensors=balanced_fusion_tensors


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_REFINED_WEIGHT_UPDATES',parent=str(BASE_OUT),
        reason='class-prior mismatch and clean forgetting in the first corruption-matched fine-tuning pass',
        seed=0,context_cap=256,steps=128,learning_rate=3e-6,source_query_sampling='class balanced, preserve one source support per class',
        source_validation_objective='40% Clean balanced NLL + 60% mean corrupted balanced NLL',
        training_objective='mean per-class query cross-entropy',benchmark_sha256=main.sha(main.frozen.CONFIG),benchmark_unchanged=True,
        same_checkpoint_all_three_methods=True,methods=['FT','FT+Shared','FT+Specific'],
        training_or_selection_access_to_final_query_labels=False,extra_labeled_official_TRAIN_development_rows=True,
        first_pass_all_results_preserved=True,classification_only=True,regression_retains_v1=True,
        inference_contexts_and_fusion_form_unchanged=True,fusion_calibration_loss='class-balanced NLL; same source-only calibration labels',
        development_experiment=True,not_paper_replication=True,
        time_UTC=__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ',__import__('time').gmtime()),
        script_sha256=main.sha(__file__),Markdown_written=False))


def launch_after_v1():
    declare();workers=json.loads((BASE_OUT/'workers.json').read_text())
    children=[];(OUT/'logs').mkdir(exist_ok=True)
    pending={int(w['gpu']):int(w['pid']) for w in workers}
    while pending:
        for gpu,old_pid in list(pending.items()):
            try:os.kill(old_pid,0);continue
            except ProcessLookupError:pass
            logfile=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
            p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
                env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            logfile.close();children.append(dict(gpu=gpu,pid=p.pid))
            atomic_json(OUT/'workers.json',children);del pending[gpu]
            print('REFINED_WORKER_STARTED',gpu,p.pid,flush=True)
        if pending:__import__('time').sleep(2)


def launch(gpus):
    declare();children=[];(OUT/'logs').mkdir(exist_ok=True)
    for shard,gpu in enumerate(gpus):
        logfile=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(shard),'--shards',str(len(gpus))],
            env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
        logfile.close();children.append(dict(gpu=gpu,pid=p.pid))
    atomic_json(OUT/'workers.json',children);print(json.dumps(children),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['declare','worker','launch','launch_after_v1','report'])
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');a=p.parse_args()
    setup()
    if a.phase=='declare':declare()
    elif a.phase=='launch':launch([int(i) for i in a.gpus.split(',')])
    elif a.phase=='launch_after_v1':launch_after_v1()
    elif a.phase=='report':main.report()
    else:main.worker(a.shard,a.shards,a.limit,steps=128)
