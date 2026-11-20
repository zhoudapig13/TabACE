"""Refine FT+Specific while freezing the benchmark, FT weights and Shared.

New mixtures can use clean reference and existing alternative predictions
directly; they need not pass through an entropy gate anchored to Shared.
All policy choices use source calibration original-row-grouped CV only.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import finetune_tabace_main_v1 as main
import finetune_source_tabace_v5 as source
import finetune_tabace_hierarchical_v4 as old
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import metrics,normalize

FT=RUN/'results/finetune_final_selected_v13'
REF=RUN/'results/finetune_final_reference_v14'
SHARED=RUN/'results/finetune_final_hierarchical_v15'
OUT=RUN/'results/ft_specific_refine_v19'
TARGETS={'classification':{'five':{'accuracy':.035,'auc':.032},'seven':{'accuracy':.033,'auc':.023}},
         'regression':{'five':{'r2':.122,'nrmse':-.075},'seven':{'r2':.188,'nrmse':-.124}}}


def rel(job):return Path('jobs')/job['task']/job['track']/str(job['meta']['task_id'])


def frozen_case(job,block,family,rho,role):
    path=Path('calibration')/f'{family}_{rho}.npz' if role=='validation' else Path('query')/block/f'{family}_{rho}.npz'
    raw=old.read_cell(FT/rel(job)/path)
    ref=old.read_cell(REF/rel(job)/path,reference=True)
    shared=json.loads((SHARED/rel(job)/'policies.json').read_text())
    selected=raw if shared['selected_bank']=='observed' else ref
    shared_pred=main.fusion.predict(selected,shared['Shared'])
    prev=main.fusion.predict(selected|dict(base=shared_pred,bank=selected['bank'][5:]),shared['Specific'][family])
    members=np.concatenate([shared_pred[None],prev[None],raw['base'][None],ref['bank'][:5],raw['bank'][5:]],axis=0)
    names=['Shared_frozen','Specific_previous','FT_single']+[f'reference_{i}' for i in range(5)]+[f'observed_specific_{i}' for i in range(5)]
    return dict(task=job['task'],meta=raw['meta'],role=role,bank=members,y=raw['y'],names=names,shared=shared_pred)


def subset(c,mask):return c|dict(bank=c['bank'][:,mask],y=c['y'][mask],shared=c['shared'][mask])


def row_weights(c,mode,prior):
    y=c['y'];w=np.ones(len(y),dtype=float)
    if c['task']=='classification' and mode!='natural':
        labels,counts=np.unique(y,return_counts=True)
        masses=np.asarray([1. if mode=='balanced' else prior[int(k)] for k in labels]);masses/=masses.sum()
        for label,count,mass in zip(labels,counts,masses):w[y==label]=mass*len(y)/count
    return w/w.sum()


def predict(c,p):
    if p['kind']=='single':return c['bank'][p['index']]
    return np.tensordot(np.asarray(p['weights']),c['bank'],axes=(0,0))


def loss(cells,p,mode,prior):
    values=[]
    for c in cells:
        q=predict(c,p);w=row_weights(c,mode,prior)
        if c['task']=='classification':v=-np.log(np.clip(q[np.arange(len(c['y'])),c['y'].astype(int)],1e-12,1.))
        else:v=(q-c['y'])**2/max(c['meta']['scale']**2,1e-12)
        values.append(float(np.dot(w,v)))
    return float(np.mean(values))


def utility(cells,p,mode,prior):
    if cells[0]['task']!='classification':return -loss(cells,p,mode,prior)
    acc=[]
    for c in cells:
        q=predict(c,p);acc.append(float(np.dot(row_weights(c,mode,prior),q.argmax(1)==c['y'])))
    return float(np.mean(acc))-.001*loss(cells,p,mode,prior)


def fit(cells,mode,prior,regularization):
    m=len(cells[0]['bank']);initial=np.zeros(m);initial[0]=1.
    def objective(w):
        value=0.;gradient=np.zeros(m)
        for c in cells:
            a=c['bank'];weights=row_weights(c,mode,prior)
            if c['task']=='classification':
                true=a[:,np.arange(len(c['y'])),c['y'].astype(int)];p=np.clip(w@true,1e-12,1.)
                value-=float(np.dot(weights,np.log(p)));gradient-=true@(weights/p)
            else:
                error=w@a-c['y'];den=max(c['meta']['scale']**2,1e-12)
                value+=float(np.dot(weights,error**2))/den;gradient+=2*a@(weights*error)/den
        value/=len(cells);gradient/=len(cells)
        anchor=w-initial;value+=regularization*np.dot(anchor,anchor);gradient+=2*regularization*anchor
        return value,gradient
    result=minimize(objective,initial,jac=True,method='SLSQP',bounds=[(0.,1.)]*m,
        constraints=[dict(type='eq',fun=lambda w:w.sum()-1.,jac=lambda w:np.ones_like(w))],
        options=dict(maxiter=150,ftol=1e-9))
    w=np.maximum(result.x,0);w/=w.sum()
    assert np.isfinite(w).all()
    return dict(kind='convex',weights=w.tolist(),mode=mode,regularization=regularization,success=bool(result.success))


def folds(job,pool,split,cells):
    _,labels,_=main.flat(pool);positions=split['cal'];rng=np.random.default_rng(1901)
    mapping={}
    groups=[positions[labels[positions]==v] for v in np.unique(labels[positions])] if job['task']=='classification' else [positions]
    for group in groups:
        for j,pos in enumerate(rng.permutation(group)):mapping[int(pos)]=j%3
    result=[]
    for fold in range(3):
        train=[];held=[]
        for c in cells:
            ids=old.row_map(job,split,pool,c['meta']['family'],c['meta']['rho'])
            mask=np.asarray([mapping[int(i)]==fold for i in ids])
            if mask.any() and (~mask).any():train.append(subset(c,~mask));held.append(subset(c,mask))
        if train:result.append((train,held))
    return result


def select(job,pool,split,cells):
    _,y,_=main.flat(pool);k=cells[0]['bank'].shape[-1] if job['task']=='classification' else 1
    prior=np.bincount(y[:256].astype(int),minlength=k).astype(float)+1. if job['task']=='classification' else np.ones(1)
    prior/=prior.sum();groups=folds(job,pool,split,cells)
    modes=['balanced','source_prior'] if job['task']=='classification' else ['natural']
    evaluation_mode='source_prior' if job['task']=='classification' else 'natural'
    candidates=[]
    for index in range(len(cells[0]['bank'])):
        p=dict(kind='single',index=index);score=float(np.mean([utility(v,p,evaluation_mode,prior) for _,v in groups]))
        candidates.append(dict(kind='single',index=index,score=score))
    for mode in modes:
        for reg in [.01,.1]:
            values=[]
            for t,v in groups:values.append(utility(v,fit(t,mode,prior,reg),evaluation_mode,prior))
            candidates.append(dict(kind='convex',mode=mode,reg=reg,score=float(np.mean(values))))
    best=max(candidates,key=lambda c:c['score'])
    p=dict(kind='single',index=best['index']) if best['kind']=='single' else fit(cells,best['mode'],prior,best['reg'])
    return p,dict(selected=best,trials=candidates,evaluation_mode=evaluation_mode,prior=prior.tolist(),folds=len(groups))


def calibrations(job):
    return [frozen_case(job,job['track'],family,rho,'validation') for family in main.families(job['track']) for rho in [50,100]]


def evaluate(job,policies,root,case_loader=frozen_case):
    rows=[]
    if job['track']=='five':
        oldrows=pd.read_csv(SHARED/rel(job)/'query_detail.csv')
        oldrows=oldrows[(oldrows.block=='clean')&oldrows.method.isin(['FT+Shared_dev','FT+Specific_dev'])].copy()
        oldrows['method']=oldrows.method.map({'FT+Shared_dev':'FT+Shared_frozen','FT+Specific_dev':'FT+Specific_adapted'})
        oldrows['clean_frozen']=True;rows.extend(oldrows.to_dict('records'))
    for family,rho in main.contexts.case_grid(job['track']):
        c=case_loader(job,job['track'],family,rho,'query');p=predict(c,policies[family])
        for name,pred in [('FT+Shared_frozen',c['shared']),('FT+Specific_adapted',p)]:
            rows.append(dict(task=job['task'],block=job['track'],task_id=job['meta']['task_id'],dataset=job['meta']['dataset'],
                family=family,rho=rho,method=name,**metrics(c['y'],pred,job['task'],c['meta']['scale'])))
    pd.DataFrame(rows).to_csv(root/rel(job)/'query_detail.csv',index=False)


def run(job):
    dest=OUT/rel(job)
    if (dest/'completion.json').exists():return
    pool=main.scaling.prepare(job['task'],job['track'],job['meta'])
    with np.load(FT/rel(job)/'split.npz') as z:split={k:z[k] for k in z.files}
    cells=calibrations(job);policies={};trials={}
    for family in main.families(job['track']):policies[family],trials[family]=select(job,pool,split,[c for c in cells if c['meta']['family']==family])
    atomic_json(dest/'policies.json',dict(status='FROZEN_BEFORE_QUERY_SCORING',Specific=policies,trials=trials,
        Shared_unchanged=True,FT_unchanged=True,benchmark_unchanged=True,final_query_labels_used=False,
        calibration_original_row_grouped_CV=True,source_calibration_already_used_by_frozen_Shared=True,
        branch_names=cells[0]['names'],seed=0))
    evaluate(job,policies,OUT)
    atomic_json(dest/'completion.json',dict(status='COMPLETE',stage='cached_alternatives',new_TFM_calls=0))


def worker(shard,shards):
    for job in main.jobs()[shard::shards]:
        try:run(job);print('SPECIFIC_REFIT_COMPLETE',job['task'],job['track'],job['meta']['task_id'],flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True);atomic_json(OUT/rel(job)/'error.json',dict(error=error))
        gc.collect()


def report(root=OUT):
    paths=list((root/'jobs').glob('*/*/*/query_detail.csv'))
    if not paths:return
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);df.to_csv(root/'query_detail.csv',index=False)
    rows=[];changes=[]
    for (task,block),g in df.groupby(['task','block']):
        fields=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        for method,h in g.groupby('method'):rows.append(dict(task=task,block=block,method=method,datasets=h.task_id.nunique(),
            complete=h.task_id.nunique()==len(main.frozen.cohort(task,block)),**{f:float(h[f].mean()) for f in fields}))
        table=g.groupby('method')[fields].mean()
        for field in fields:
            gain=float(table.loc['FT+Specific_adapted',field]-table.loc['FT+Shared_frozen',field]);target=TARGETS.get(task,{}).get(block,{}).get(field,0.)
            changes.append(dict(task=task,block=block,metric=field,Specific_minus_Shared=gain,paper_reference_gap=target,difference_from_target=gain-target))
    pd.DataFrame(rows).to_csv(root/'averages.csv',index=False);pd.DataFrame(changes).to_csv(root/'gaps_vs_paper.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print('COMPLETE_JOBS',len(paths),'/175',flush=True)


def launch():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_SPECIFIC_REFIT',seed=0,FT_frozen=str(FT),Shared_frozen=str(SHARED),
        baseline_report_sha256=main.sha(RUN/'results/finetune_delivery_v17/averages.csv'),paper_reference_gaps=TARGETS,
        target='FT+Specific > FT+Shared, prefer increments close to paper Specific−Shared',Clean='retain prior Specific',
        new_weighting='ungated family-aware convex mixtures or source-CV selected expert',
        final_query_labels_for_selection=False,all_attempts_retained=True,Markdown_written=False))
    children=[]
    for i in range(8):
        log=(OUT/'logs'/f'worker{i}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--shard',str(i),'--shards','8'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();children.append(dict(worker=i,pid=p.pid))
    atomic_json(OUT/'workers.json',children);print(json.dumps(children),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','launch','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    if a.phase=='launch':launch()
    elif a.phase=='report':report()
    else:worker(a.shard,a.shards)
