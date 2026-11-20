"""Protocol-corrected source-context-only classification DEVELOPMENT benchmark.

Freeze using Base alone before any new FT or TabACE predictions. Queries and
their labels are exactly the clean v22 queries. No old query-feature-corruption
scores are relabeled as source-context corruption evidence.
"""
import argparse
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
import calibrated_five_protocol as corruptor
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,metrics

OUT=RUN/'results/finetune_source_benchmark_v5'
PROFILES=[2.65,2.70,2.75]


def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_NEW_BASE_OR_FT_PREDICTIONS',seed=0,context_cap=256,
        datasets=72,mechanisms=list(protocol.FAMILIES),strengths=[.2,.5,.8,1.],
        operator='calibrated_five_protocol.corrupt',source_context_features_or_labels_only=True,
        final_query_features_and_labels='identical to v22 Clean; never corrupted',
        profiles=dict(label_multiplier=PROFILES,feature_fraction=0.,feature_count=1,amplitude=0.),
        parameters_are_internal_operator_parameters_nominal_strengths_unchanged=True,
        Base_target=dict(accuracy=.494,auc=.676),tolerance=.005,
        freeze_rule='smallest squared Base deviation; no FT/TabACE scores exist for this new protocol at selection',
        Base_development_query_scores_used_for_benchmark_selection=True,not_independent_test_performance=True,
        reason='restore source-context corruption direction in reviewer-proposed fine-tuning comparison',
        previous_query_feature_corruption_protocol_preserved=True,development_only=True,not_archived_paper_replication=True,
        time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),script_sha256=main.sha(__file__),Markdown_written=False))


def condition(meta,family,rho,multiplier):
    data,_=main.scaling.bank('classification','clean',meta,'original',0)
    if rho:
        X,y,_=corruptor.corrupt(data['X_source'],data['y_source'],family=family,rho=rho/100.,
            label_multiplier=multiplier,feature_fraction=0.,feature_count=1,amplitude=0.)
        data=data|dict(X_source=X,y_source=y)
    return data


def worker(shard,shards):
    torch.set_num_threads(1);predictor=Predictor('classification','cuda:0')
    for meta in main.frozen.cohort('classification','five')[shard::shards]:
        for multiplier in PROFILES:
            for family,rho in main.contexts.case_grid('five'):
                path=OUT/'profiles'/f'{multiplier:.2f}'/str(meta['task_id'])/f'{family}_{rho}.npz'
                if path.exists() and path.with_suffix('.json').exists():continue
                data=condition(meta,family,rho,multiplier)
                clean,_=main.scaling.bank('classification','clean',meta,'original',0)
                nc=int(clean['y_source'].max())+1
                assert np.array_equal(data['X_target'],clean['X_target']) and np.array_equal(data['y_target'],clean['y_target'])
                p,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,n_classes=nc,random_state=0)
                main.scaling.npwrite(path,**data,Base_prediction=p)
                atomic_json(path.with_suffix('.json'),dict(task='classification',block='five',task_id=meta['task_id'],dataset=meta['dataset'],
                    family=family,rho=rho,label_multiplier=multiplier,scale=1.,metrics=metrics(data['y_target'],p,'classification',1.),
                    input_hashes={k:protocol.array_sha256(v) for k,v in data.items()},query_features_and_labels_unchanged=True,
                    corruption_location='source_context_only',seed=0,context_rows=256))
        print('SOURCE_ONLY_BASE_COMPLETE',meta['task_id'],flush=True)


def select():
    assert not (OUT/'frozen.json').exists()
    rows=[]
    for multiplier in PROFILES:
        records=[json.loads(p.read_text()) for p in (OUT/'profiles'/f'{multiplier:.2f}').glob('*/*.json')]
        assert len(records)==1440,(multiplier,len(records))
        avg={k:float(np.mean([r['metrics'][k] for r in records])) for k in ['accuracy','auc']}
        distance=((avg['accuracy']-.494)/.005)**2+((avg['auc']-.676)/.005)**2
        rows.append(dict(label_multiplier=multiplier,**avg,distance=distance,
            within_tolerance=abs(avg['accuracy']-.494)<=.005 and abs(avg['auc']-.676)<=.005))
    pd.DataFrame(rows).to_csv(OUT/'Base_candidates.csv',index=False)
    eligible=[r for r in rows if r['within_tolerance']]
    best=min(eligible or rows,key=lambda r:r['distance'])
    atomic_json(OUT/'frozen.json',dict(status='FROZEN_BEFORE_FINETUNING',**best,
        target=dict(accuracy=.494,auc=.676),source_context_only=True,query_unchanged=True,
        input_bank=str(OUT/'profiles'/f"{best['label_multiplier']:.2f}"),datasets=72,context_rows=256,seed=0,
        FT_Shared_Specific_used_for_selection=False,development_only=True,
        parameters=dict(label_multiplier=best['label_multiplier'],feature_fraction=0.,feature_count=1,amplitude=0.),
        time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('SOURCE_ONLY_BASE_FROZEN',best,flush=True)


def launch():
    plan();children=[];(OUT/'logs').mkdir(exist_ok=True)
    for gpu in range(8):
        log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu),'--shards','8'],
            env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(dict(gpu=gpu,pid=p.pid));atomic_json(OUT/'workers.json',children)
    print(json.dumps(children),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','launch','select']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='worker':worker(a.shard,a.shards)
    elif a.phase=='select':select()
    else:launch()
