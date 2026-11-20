"""Fresh pretrained Base/TabACE inference paired with the frozen FT benchmark.

Observed-context constructors only, original confidence/agreement/Base gate.
Development-source calibration chooses policies; final query labels only score.
The previous FT rows are preserved verbatim in the replacement comparison CSV.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
import pandas as pd
import torch
import finetune_tabace_main_v1 as main
import finetune_source_tabace_v5 as source
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,CHECKPOINTS,metrics

OUT=RUN/'results/pretrained_tabace_rerun_v21'
FT=RUN/'results/finetune_final_selected_v13'
FT_REPORT=RUN/'results/finetune_delivery_v17/averages.csv'
FT_SPECIFIC_REPORT=RUN/'results/ft_specific_context_adaptation_v20/averages.csv'
PAIRS=[(t,b) for t in ['classification','regression'] for b in ['clean','five','seven']]
TARGETS={
 'classification':{'clean':{'Base':[.804,.901],'Shared':[.804,.901],'Specific':[.804,.900]},
                   'five':{'Base':[.494,.676],'Shared':[.575,.720],'Specific':[.610,.752]},
                   'seven':{'Base':[.763,.826],'Shared':[.772,.831],'Specific':[.805,.854]}},
 'regression':{'clean':{'Base':[.739,.476],'Shared':[.738,.478],'Specific':[.738,.476]},
               'five':{'Base':[-1.077,1.270],'Shared':[.363,.783],'Specific':[.485,.708]},
               'seven':{'Base':[.273,.730],'Shared':[.467,.659],'Specific':[.655,.535]}}}


def relative(job):return Path('jobs')/job['task']/job['track']/str(job['meta']['task_id'])


def freeze_plan():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'plan.json'
    if path.exists():return
    atomic_json(path,dict(status='DECLARED_BEFORE_FRESH_PRETRAINED_INFERENCE',seed=0,context_cap=256,
        methods=['Base','Shared','Specific'],all_TFM_weights='official pretrained, no FT delta loaded',
        checkpoints={t:dict(path=str(CHECKPOINTS/f'tabicl-{t}-v2-20260212.ckpt'),
            sha256=main.sha(CHECKPOINTS/f'tabicl-{t}-v2-20260212.ckpt')) for t in ['classifier','regressor']},
        frozen_benchmark_sha256=main.sha(main.frozen.CONFIG),
        frozen_classification_five_sha256=main.sha(RUN/'results/finetune_source_benchmark_v5/frozen.json'),
        benchmark_and_final_query_rows_unchanged=True,semisynthetic_regression_clean_five=True,
        observed_context_only=True,extra_clean_reference_contexts=False,
        Shared_roles=5,Specific_roles=10,Specific_contains_Shared=True,
        native_members_per_role=1,nominal_passes=dict(Base=1,Shared=6,Specific=11),
        calibration='same held-out official TRAIN rows as FT; .5 and 1.0 plus clean',
        Shared='one family-blind policy per task/block; original entropy or regression disagreement gate',
        Specific='family-specific original policy on union of ten roles; Shared fallback',
        calibration_selection='held-out dataset IDs within source calibration; gains 1/2/4, regularization .0001/.01',
        source_labels_for_calibration_only=True,final_query_labels_used_for_selection=False,
        target_paper_values=TARGETS,paper_values_not_optimizer_objective=True,
        frozen_FT_report_sha256=main.sha(FT_REPORT),frozen_FT_Specific_report_sha256=main.sha(FT_SPECIFIC_REPORT),
        original_paper_exact_replication=False,development_reconstruction=True,Markdown_written=False))


def generate(job,predictor):
    dest=OUT/relative(job)
    if (dest/'bank_completion.json').exists():return
    pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
    with np.load(FT/relative(job)/'split.npz') as z:split={k:z[k] for k in z.files}
    _,labels,ids=main.flat(pool);nc=int(labels.max())+1 if job['task']=='classification' else None
    assert not np.intersect1d(ids[np.r_[split['train'],split['val'],split['cal']]],pool['query_ids']).size
    for family,rho in [('original',0)]+[(f,r) for f in main.families(job['track']) for r in [50,100]]:
        block='clean' if rho==0 else job['track'];path=dest/'calibration'/f'{family}_{rho}.npz'
        data=source.materialize(job,pool,np.r_[split['train'],split['val']],split['cal'],family,rho)
        scale=1. if job['task']=='classification' else max(float(data['y_target'].std()),1e-8)
        save_bank(job,predictor,data,block,family,rho,'validation',scale,path,nc)
    for block in (['clean','five'] if job['track']=='five' else ['seven']):
        for family,rho in main.contexts.case_grid(block):
            data,info=source.bank(job['task'],block,job['meta'],family,rho)
            path=dest/'query'/block/f'{family}_{rho}.npz'
            save_bank(job,predictor,data,block,family,rho,'query',info['scale'],path,nc)
    atomic_json(dest/'bank_completion.json',dict(status='COMPLETE',task=job['task'],track=job['track'],task_id=job['meta']['task_id'],
        pretrained_weights_only=True,final_input_provider='same source.bank as frozen FT comparison',
        official_TRAIN_calibration_disjoint_final_queries=True))


def save_bank(job,predictor,data,block,family,rho,role,scale,path,nc):
    # Reuse the audited constructor/inference implementation, never FT's Predictor.
    arrays,record=main.make_bank(job,predictor,data,block,family,rho,role,scale,path,nc)
    if record.get('pretrained_weights_only'):return
    reference_path=FT/relative(job)/('calibration' if role=='validation' else Path('query')/block)/f'{family}_{rho}.json'
    old=json.loads(reference_path.read_text())
    assert record['input_hashes']==old['input_hashes'],(job['meta']['task_id'],block,family,rho,'input mismatch')
    record['data_id']=job['meta']['data_id'];record['pretrained_weights_only']=True
    record['FT_delta_loaded']=False;record['paired_FT_input_hashes_match']=True
    record['Base_inference_seconds']=record.pop('FT_inference_seconds')
    record['Shared_inference_seconds']=record.pop('FT_Shared_inference_seconds')
    record['Specific_inference_seconds']=record.pop('FT_Specific_inference_seconds')
    record['inference_seconds']['Base']=record['inference_seconds'].pop('FT')
    record['member_rows']['Base']=record['member_rows'].pop('FT')
    atomic_json(path.with_suffix('.json'),record)


def worker(shard):
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    jobs=main.jobs();offset=shard*len(jobs)//8;jobs=jobs[offset:]+jobs[:offset];predictors={}
    for job in jobs:
        dest=OUT/relative(job);dest.mkdir(parents=True,exist_ok=True)
        with (dest/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (dest/'bank_completion.json').exists():continue
            try:
                print('PRETRAINED_BANK_START',job['task'],job['track'],job['meta']['task_id'],flush=True)
                if job['task'] not in predictors:predictors[job['task']]=Predictor(job['task'],'cuda:0')
                generate(job,predictors[job['task']]);(dest/'error.json').unlink(missing_ok=True)
                print('PRETRAINED_BANK_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
            except Exception:
                error=traceback.format_exc();print(error,flush=True);atomic_json(dest/'error.json',dict(error=error))
            gc.collect();torch.cuda.empty_cache()


def cells(task,block,role):
    track='five' if block=='clean' else block;result=[]
    for meta in main.frozen.cohort(task,block):
        root=OUT/'jobs'/task/track/str(meta['task_id'])
        paths=[root/'calibration'/'original_0.npz'] if role=='validation' and block=='clean' else (
            [root/'calibration'/f'{f}_{r}.npz' for f in main.families(block) for r in [50,100]] if role=='validation' else
            [root/'query'/block/f'{f}_{r}.npz' for f,r in main.contexts.case_grid(block)])
        for path in paths:
            record=json.loads(path.with_suffix('.json').read_text())
            with np.load(path) as z:arrays={k:z[k] for k in z.files}
            result.append(main.cell(arrays,record)|dict(path=path))
    return result


def fit_and_evaluate(task,block):
    torch.set_num_threads(1);main.fusion.GATE_VARIANT='entropy';dest=OUT/'policies'/task/block
    if (dest/'completion.json').exists():return
    cal=cells(task,block,'validation')
    shared,inner,trials,held=main.fusion.select(cal,task,5,dict(kind='base'))
    policies={};logs={}
    for family in sorted({c['meta']['family'] for c in cal}):
        group=[c for c in cal if c['meta']['family']==family]
        p,_,attempts,_=main.fusion.select(group,task,10,inner)
        if p is inner:p=shared
        policies[family]=p;logs[family]=attempts
    atomic_json(dest/'policies.json',dict(status='FROZEN_BEFORE_QUERY_SCORING',Shared=shared,Specific=policies,
        Shared_trials=trials,Specific_trials=logs,inner_validation_data_ids=held,
        query_labels_used_for_selection=False,pretrained_weights_only=True,original_gate=True,seed=0))
    rows=[]
    for c in cells(task,block,'query'):
        family,rho=c['meta']['family'],c['meta']['rho']
        predictions=dict(Base=c['base'],Shared=main.fusion.predict(c,shared),Specific=main.fusion.predict(c,policies[family]))
        p=OUT/'predictions'/task/block/str(c['meta']['task_id'])/f'{family}_{rho}.npz'
        main.scaling.npwrite(p,**predictions,y=c['y'])
        for name,pred in predictions.items():
            rows.append(dict(task=task,block=block,task_id=c['meta']['task_id'],data_id=c['meta']['data_id'],dataset=c['meta']['dataset'],
                family=family,rho=rho,method=name,**metrics(c['y'],pred,task,c['meta']['scale'])))
    pd.DataFrame(rows).to_csv(dest/'query_detail.csv',index=False)
    atomic_json(dest/'completion.json',dict(status='COMPLETE',conditions=len(rows)//3,policies_source_calibration_only=True))
    print('PRETRAINED_FUSION_COMPLETE',task,block,flush=True)


def report():
    paths=list((OUT/'policies').glob('*/*/query_detail.csv'))
    if not paths:return
    detail=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);detail.to_csv(OUT/'query_detail.csv',index=False)
    rows=[];diff=[]
    for (task,block,method),g in detail.groupby(['task','block','method']):
        fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        row=dict(task=task,block=block,method=method,datasets=g.task_id.nunique(),source='fresh pretrained rerun',
            **{f:float(g[f].mean()) for f in fields});rows.append(row)
        for i,f in enumerate(fields):diff.append(dict(task=task,block=block,method=method,metric=f,value=row[f],
            paper_value=TARGETS[task][block][method][i],difference=row[f]-TARGETS[task][block][method][i]))
    averages=pd.DataFrame(rows);averages.to_csv(OUT/'averages.csv',index=False)
    pd.DataFrame(diff).to_csv(OUT/'paper_reproduction_gaps.csv',index=False)
    a=pd.read_csv(FT_REPORT);a=a[a.method=='FT'].copy();a['source']='frozen FT experiment'
    b=pd.read_csv(FT_SPECIFIC_REPORT);b=b.copy();b['method']=b.method.map({'FT+Shared_frozen':'FT+Shared','FT+Specific_adapted':'FT+Specific'})
    b['source']='frozen adapted FT experiment'
    combined=pd.concat([averages,a,b],ignore_index=True)
    combined.to_csv(OUT/'replacement_comparison_long.csv',index=False)
    table=[]
    for task in ['classification','regression']:
        for method in ['Base','Shared','Specific','FT','FT+Shared','FT+Specific']:
            g=combined[(combined.task==task)&(combined.method==method)];row=dict(task=task,method=method)
            for block in ['clean','five','seven']:
                h=g[g.block==block]
                if len(h):
                    h=h.iloc[0];fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
                    row[block]=' / '.join(f'{h[f]:.4f}' for f in fields)
            table.append(row)
    pd.DataFrame(table).to_csv(OUT/'replacement_comparison.csv',index=False)
    assert main.sha(FT_REPORT)==json.loads((OUT/'plan.json').read_text())['frozen_FT_report_sha256']
    assert main.sha(FT_SPECIFIC_REPORT)==json.loads((OUT/'plan.json').read_text())['frozen_FT_Specific_report_sha256']
    if len(paths)==6:
        latest=RUN/'results/base_tabace_ft_comparison_latest.csv'
        temporary=latest.with_suffix('.tmp.csv');pd.DataFrame(table).to_csv(temporary,index=False);temporary.replace(latest)
        atomic_json(latest.with_suffix('.json'),dict(status='REPLACED_WITH_MEASURED_BASE_TABACE_ROWS',
            pretrained_results=str(OUT/'averages.csv'),FT_results=str(FT_REPORT),FT_Specific_results=str(FT_SPECIFIC_REPORT),
            all_reported_rows_measured=True,prior_paper_reference_rows_removed=True,
            pretrained_context_cap=256,FT_Shared_context_cap=256,FT_Specific_context_cap=2048,
            FT_rows_preserved=True,seed=0,Markdown_written=False))
    print(pd.DataFrame(table).to_string(index=False),flush=True)


def pipeline():
    freeze_plan();(OUT/'logs').mkdir(exist_ok=True);children=[];records=[]
    for gpu in range(8):
        log=(OUT/'logs'/f'gpu{gpu}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(gpu)],env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(p);records.append(dict(gpu=gpu,pid=p.pid))
    atomic_json(OUT/'workers.json',records)
    for p in children:p.wait()
    done=len(list(OUT.glob('jobs/*/*/*/bank_completion.json')))
    if done!=175:raise RuntimeError(f'Incomplete inference {done}/175; inspect errors')
    fitting=[]
    for task,block in PAIRS:
        log=(OUT/'logs'/f'fusion_{task}_{block}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'fusion','--task',task,'--block',block],stdout=log,stderr=subprocess.STDOUT)
        log.close();fitting.append(p)
    codes=[p.wait() for p in fitting]
    if any(codes):raise RuntimeError(f'Fusion failures {codes}')
    report();atomic_json(OUT/'completion.json',dict(status='COMPLETE',jobs=175,conditions=4151,methods=3,FT_rows_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pipeline','worker','fusion','report']);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--task');p.add_argument('--block');a=p.parse_args()
    if a.phase=='pipeline':pipeline()
    elif a.phase=='worker':worker(a.shard)
    elif a.phase=='fusion':fit_and_evaluate(a.task,a.block)
    else:report()
