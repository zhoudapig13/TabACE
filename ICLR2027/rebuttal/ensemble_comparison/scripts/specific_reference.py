"""Predeclared five-branch Specific reconstruction from registered operators.

Family is visible; strength and target labels are absent from constructors.
This fixed-R=5 variant is distinct from the archived uncapped Specific search.
"""
from __future__ import annotations
import argparse
from functools import partial
import json
import traceback
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from shared_reference import (RUN,atomic_json,protocol,Predictor,PFNPredictor,
    specifications,load_condition,run_one,geometry,class_subset,matched_rows)


def balanced(y,n,seed=0):
    rng=np.random.default_rng(seed);parts=[np.flatnonzero(y==v) for v in np.unique(y)]
    counts=np.full(len(parts),n//len(parts));counts[:n%len(parts)]+=1
    ids=np.concatenate([rng.choice(p,int(k),replace=True) for p,k in zip(parts,counts)]);rng.shuffle(ids)
    return ids


def response_balanced(y,n,seed=0):
    rng=np.random.default_rng(seed);parts=[p for p in np.array_split(np.argsort(y,kind='stable'),8) if len(p)]
    counts=np.full(len(parts),n//len(parts));counts[:n%len(parts)]+=1
    ids=np.concatenate([rng.choice(p,int(k),replace=True) for p,k in zip(parts,counts)]);rng.shuffle(ids)
    return ids


def importance(X,y):
    centered=X-X.mean(0);target=y-y.mean()
    scores=np.abs(np.sum(centered*target[:,None],axis=0))/np.maximum(np.sqrt(np.sum(centered**2,axis=0)*np.sum(target**2)),1e-12)
    return np.argsort(scores,kind='stable')[::-1]


def repair_labels(Z,y,threshold):
    d=pairwise_distances(Z,metric='sqeuclidean');np.fill_diagonal(d,np.inf);k=min(10,len(y)-1)
    neighbors=np.argpartition(d,k-1,axis=1)[:,:k];classes=np.unique(y)
    votes=np.stack([(y[neighbors]==c).mean(1) for c in classes],axis=1)
    proposed=classes[votes.argmax(1)];confidence=votes.max(1)
    result=y.copy();mask=confidence>=threshold;result[mask]=proposed[mask]
    # Preserve every observed source class without inventing unseen labels.
    for label in classes:
        if not np.any(result==label):result[np.flatnonzero(y==label)[0]]=label
    return result


def reflected_tails(y,q,shrink):
    values,counts=np.unique(y,return_counts=True)
    lo,hi=np.quantile(y,[q,1-q])
    repeated=values[(counts>=max(2,int(np.ceil(.01*len(y)))))&((values<=lo)|(values>=hi))]
    mask=np.isin(y,repeated);clean=y[~mask]
    center=float(np.median(clean)) if len(clean)>=max(8,int(.1*len(y))) else .5*(float(values.min())+float(values.max()))
    result=y.copy();result[mask]=center-shrink*(y[mask]-center)
    return result


def specific_contexts(X,y,Q,canonical_X,canonical_y,budget,task,*,family,track,history=None,reference=None,base=None):
    order=importance(X,y.astype(float));top=int(order[0]);center=np.median(X,axis=0)
    scale=np.maximum(np.percentile(X,75,axis=0)-np.percentile(X,25,axis=0),1e-6)
    if track.startswith('seven') and task=='regression':
        H,hy=history;R,ry=reference
        if not len(hy):H,hy=X,y
        Z,V,_,_=geometry(H,Q);rng=np.random.default_rng(0)
        a=rng.choice(len(hy),budget,replace=True);b=response_balanced(hy,budget)
        c=matched_rows(Z,V,budget,0);d=rng.choice(len(ry),budget,replace=True)
        pool=np.vstack([H,R]);labels=np.r_[hy,ry];P,T,_,_=geometry(pool,Q)
        e=matched_rows(P,T,budget,0)
        return [('history_bootstrap',H[a],hy[a],Q),('history_response_balanced',H[b],hy[b],Q),
            ('history_target_matched',H[c],hy[c],Q),('reference_bootstrap',R[d],ry[d],Q),
            ('history_reference_matched',pool[e],labels[e],Q)]
    if track.startswith('seven') and task=='classification':
        Z,V,_,_=geometry(X,Q)
        distances=pairwise_distances(Z,V,metric='sqeuclidean');k=min(10,len(V))
        distance=np.partition(distances,k-1,axis=1)[:,:k].mean(1)
        if family=='domain':
            target_center=np.median(Q,axis=0)
            target_scale=np.maximum(np.percentile(Q,75,axis=0)-np.percentile(Q,25,axis=0),1e-6)
            aligned=(canonical_X-center)/scale*target_scale+target_center
            return [(f'transport_a{a}',(1-a)*canonical_X+a*aligned,canonical_y,Q) for a in [.5,.75,1.,1.25,1.5]]
        if family in ['concept','temporal']:
            H,hy=history;R,ry=reference;rng=np.random.default_rng(0)
            a=rng.choice(len(ry),budget,replace=True);b=balanced(ry,budget)
            result=[('reference_bootstrap',R[a],ry[a],Q),('reference_balanced',R[b],ry[b],Q)]
            for count in [32,64,128]:
                if len(H):
                    d=pairwise_distances((H-center)/scale,(R-center)/scale,metric='sqeuclidean').min(1)
                    ids=np.argsort(d,kind='stable')[:min(count,len(H))];pool=np.vstack([R,H[ids]]);labels=np.r_[ry,hy[ids]]
                else:pool,labels=R,ry
                rows=balanced(labels,budget);result.append((f'reference_history{count}',pool[rows],labels[rows],Q))
            return result
        if family=='label':
            prior=float(np.mean(y));posterior=np.clip(base[:,1],1e-8,1-1e-8);estimate=np.clip(prior,.02,.98)
            for _ in range(100):
                positive=posterior*estimate/max(prior,1e-8)
                negative=(1-posterior)*(1-estimate)/max(1-prior,1e-8)
                updated=float(np.clip(np.mean(positive/(positive+negative)),.02,.98))
                if abs(updated-estimate)<1e-7:break
                estimate=updated
            pcc=float(np.mean(posterior));rng=np.random.default_rng(0);result=[]
            for label,p in [('source',prior),('em',estimate),('pcc',pcc),('balanced',.5),('em_pcc',.5*(estimate+pcc))]:
                if len(np.unique(y))<2:rows=rng.choice(len(y),budget,replace=True)
                else:
                    n=int(np.clip(round(p*budget),1,budget-1))
                    rows=np.r_[rng.choice(np.flatnonzero(y==0),budget-n,replace=True),rng.choice(np.flatnonzero(y==1),n,replace=True)];rng.shuffle(rows)
                result.append(('prior_'+label,X[rows],y[rows],Q))
            return result
        # Covariate, selection, support: remove feature-identical duplicates then
        # use nested target-support intervals with observed-class preservation.
        _,unique=np.unique(X,axis=0,return_index=True);unique=np.sort(unique)
        A,b,d=X[unique],y[unique],distance[unique];result=[]
        for fraction in [.2,.4,.6,.8,1.]:
            ids=class_subset(b,d,fraction,max(1,len(b)))
            rows=balanced(b[ids],budget)
            result.append((f'unique_support_q{int(100*fraction)}',A[ids][rows],b[ids][rows],Q))
        return result
    if task=='regression':
        if family=='shortcut':
            views=[]
            for count in [1,2]:
                columns=np.sort(order[min(count,len(order)-1):]);views.append((f'drop_top{count}',X[:,columns],y,Q[:,columns]))
            columns=np.sort(order[max(1,len(order)//2):]) if len(order)>1 else np.array([0])
            if not len(columns):columns=np.array([0])
            views.append(('weak_half',X[:,columns],y,Q[:,columns]))
            for radius in [2.,4.]:
                A=X.copy();T=Q.copy();A[:,top]=np.clip(A[:,top],center[top]-radius*scale[top],center[top]+radius*scale[top]);T[:,top]=np.clip(T[:,top],center[top]-radius*scale[top],center[top]+radius*scale[top])
                views.append((f'clip_top_iqr{radius}',A,y,T))
            return views
        views=[]
        for q,shrink in zip([.15,.2,.25,.3,.35],[.2,.3,.4,.5,.6]):
            repaired=reflected_tails(y,q,shrink);rows=response_balanced(repaired,budget)
            A=X[rows].copy();T=Q.copy()
            if family in ['band','local','mixed']:
                radius=1.5+5*shrink
                A[:,top]=np.clip(A[:,top],center[top]-radius*scale[top],center[top]+radius*scale[top]);T[:,top]=np.clip(T[:,top],center[top]-radius*scale[top],center[top]+radius*scale[top])
            views.append((f'tail_q{q}_reflection{shrink}',A,repaired[rows],T))
        return views
    if family=='shortcut':
        result=[]
        for count in [1,2,3]:
            columns=np.sort(order[min(count,len(order)-1):]);result.append((f'drop_top{count}',X[:,columns],y,Q[:,columns]))
        for fraction in [.5,.75]:
            count=max(1,int(np.ceil(len(order)*fraction)));columns=np.sort(order[-count:]);result.append((f'weak{fraction}',X[:,columns],y,Q[:,columns]))
        return result
    columns=np.array([j for j in range(X.shape[1]) if j!=top]);columns=columns if len(columns) else np.array([0])
    if family=='local':
        Z,V,_,_=geometry(X,Q);d=pairwise_distances(Z,V,metric='sqeuclidean').min(1);result=[]
        for fraction in [.2,.4,.6,.8,1.]:
            ids=class_subset(y,d,fraction,budget);rows=balanced(y[ids],max(len(np.unique(y)),len(ids)))
            result.append((f'local_support_q{fraction}',X[ids][rows],y[ids][rows],Q))
        return result
    # Boundary/rank-band/mixed use both original and dominant-feature-free
    # neighborhood repairs at the predeclared 60/70/80 percent consensus levels.
    Z,_,_,_=geometry(X,Q);small=Z[:,columns]
    result=[]
    for threshold in [.6,.7,.8]:
        repaired=repair_labels(small,y,threshold)
        result.append((f'drop_top_repair{threshold}',X[:,columns],repaired,Q[:,columns]))
    result.append(('all_features_repair.7',X,repair_labels(Z,y,.7),Q))
    clipped=X.copy();query=Q.copy();clipped[:,top]=np.clip(clipped[:,top],center[top]-2*scale[top],center[top]+2*scale[top]);query[:,top]=np.clip(query[:,top],center[top]-2*scale[top],center[top]+2*scale[top])
    result.append(('clip_top_repair.7',clipped,repair_labels(small,y,.7),query))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],default='tabicl')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--track',choices=['both','five','five_corrected','five_calibrated','seven_calibrated','seven_reimplemented'],default='both');args=p.parse_args()
    torch.set_num_threads(1);predictors={};failures=0
    tracks=['five','seven_reimplemented'] if args.track=='both' else [args.track]
    jobs=[(track,m) for track in tracks for m in specifications(track)]
    if args.limit:jobs=jobs[:args.limit]
    for job_index,(track,meta) in enumerate(jobs):
        if track.startswith('five'):families=protocol.FAMILIES
        else:
            from seven_shifts import FAMILIES
            families=FAMILIES
        conditions=[(f,r) for f in families for r in [0,20,50,80,100]]
        if args.smoke:conditions=conditions[:1]
        task=meta['task']
        if task not in predictors:predictors[task]=Predictor(task,'cuda:0') if args.backbone=='tabicl' else PFNPredictor(task)
        for condition_index,(family,rho) in enumerate(conditions):
            if (job_index*len(conditions)+condition_index)%args.shards!=args.shard:continue
            out=RUN/'results/reference_specific'/args.backbone/track/task/str(meta['task_id'])/f'{family}_{rho}.npz'
            if out.with_suffix('.json').exists():continue
            if track=='five_corrected' and rho==0:
                from corrected_source_protocol import reuse_unchanged_clean
                old=RUN/'results/reference_specific'/args.backbone/'five'/task/str(meta['task_id'])/f'{family}_0.npz'
                if reuse_unchanged_clean(old,out):continue
            print('START',args.backbone,track,task,meta['task_id'],family,rho,flush=True)
            try:
                data,info=load_condition(meta,family,rho)
                factory=partial(specific_contexts,family=family,track=track,
                    history=(data.get('X_history'),data.get('y_history')),reference=(data.get('X_reference'),data.get('y_reference')))
                run_one(predictors[task],data,info,out,args.backbone,context_factory=factory,variant='specific',include_native=False)
                print('DONE',args.backbone,track,task,meta['task_id'],family,rho,flush=True)
            except Exception as exc:
                failures+=1;row=meta|{'family':family,'rho':rho,'error':repr(exc),'traceback':traceback.format_exc()}
                atomic_json(RUN/'results/inference_failures'/f"specific_{args.backbone}_{track}_{task}_{meta['task_id']}_{family}_{rho}.json",row)
                print('FAILED',json.dumps(row),flush=True);torch.cuda.empty_cache()
    print('WORKER_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
