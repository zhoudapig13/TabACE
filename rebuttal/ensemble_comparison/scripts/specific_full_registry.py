"""Declared Appendix ranges for the controlled five-family reconstruction.

Every registered level below produces a candidate; no top-k or context-count
limit is applied. Shared and the original directed roles are retained too.
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from shared_reference import geometry,class_subset
from specific_reference import importance,repair_labels,reflected_tails,response_balanced

TAILS=[.15,.2,.25,.3,.35]
REFLECTIONS=[.2,.3,.4,.5,.6]
RADII=[1.5,2.,2.5,3.,3.5,4.]
FRACTIONS=[.7,.75,.8,.85,.9]
CONSENSUS=[.6,.7,.8]


def extra_contexts(X,y,Q,budget,task,family):
    order=importance(X,y.astype(float));top=int(order[0]);Z,V,center,scale=geometry(X,Q)
    d=pairwise_distances(Z,V,metric='sqeuclidean')/max(Z.shape[1],1)
    k=min(10,len(Q));distance=np.partition(d,k-1,axis=1)[:,:k].mean(1)
    if task=='classification':
        if family in ['shortcut','mixed']:
            if family=='shortcut':
                for count in [4,8,16,32]:
                    columns=np.sort(order[min(count,len(order)-1):])
                    yield f'drop_top_registered_{count}',X[:,columns],y,Q[:,columns]
            for fraction in [.5,.75]:
                columns=np.sort(order[-max(1,int(np.ceil(len(order)*fraction))):])
                yield f'weak_registered_{fraction}',X[:,columns],y,Q[:,columns]
        if family in ['shortcut','band','mixed']:
            for tail in [.01,.05,.1]:
                lo,hi=np.quantile(X[:,top],[tail,1-tail]);A=X.copy();T=Q.copy()
                A[:,top]=np.clip(A[:,top],lo,hi);T[:,top]=np.clip(T[:,top],lo,hi)
                yield f'feature_tail_clip_{tail}',A,y,T
        if family=='shortcut':
            for fraction in FRACTIONS:
                ids=class_subset(y,distance,fraction,budget)
                yield f'stable_support_{fraction}',X[ids],y[ids],Q
        if family in ['boundary','band','mixed']:
            for threshold in CONSENSUS:
                yield f'full_feature_repair_{threshold}',X,repair_labels(Z,y,threshold),Q
        if family in ['boundary','local','mixed']:
            fractions=[.2,.4,.6] if family=='local' else [.2,.4,.6,.8,1.]
            for fraction in fractions:
                ids=class_subset(y,distance,fraction,budget)
                for threshold in CONSENSUS:
                    repaired=repair_labels(Z[ids],y[ids],threshold)
                    yield f'cumulative_support_{fraction}_repair_{threshold}',X[ids],repaired,Q
        if family=='band':
            # The exact historical endpoints are absent; this declared median
            # band uses source quantiles only and is not called bit-exact.
            lo,hi=np.quantile(X[:,top],[.3,.7]);inside=(X[:,top]>=lo)&(X[:,top]<=hi)
            for share in [.2,.4,.6,.8,1.]:
                selected=[];rng=np.random.default_rng(0)
                classes=np.unique(y)
                for position,label in enumerate(classes):
                    quota=budget//len(classes)+(position<budget%len(classes))
                    inside_count=int(round(share*quota));outside_count=quota-inside_count
                    for mask,count in [(inside,inside_count),(~inside,outside_count)]:
                        ids=np.flatnonzero((y==label)&mask)
                        if not len(ids):ids=np.flatnonzero(y==label)
                        if count:selected.extend(rng.choice(ids,count,replace=True))
                ids=np.asarray(selected,dtype=int)
                if not len(ids):ids=np.arange(len(y))
                for label in np.unique(y):
                    if not np.any(y[ids]==label):ids=np.r_[ids,np.flatnonzero(y==label)[0]]
                yield f'band_offband_share_{share}',X[ids],y[ids],Q
        return
    if family=='shortcut':
        for radius in RADII:
            A=X.copy();T=Q.copy();lo=center[top]-radius*scale[top];hi=center[top]+radius*scale[top]
            A[:,top]=np.clip(A[:,top],lo,hi);T[:,top]=np.clip(T[:,top],lo,hi)
            yield f'registered_clip_iqr_{radius}',A,y,T
        for fraction in [.5,.75]:
            columns=np.sort(order[-max(1,int(np.ceil(len(order)*fraction))):])
            yield f'registered_weak_{fraction}',X[:,columns],y,Q[:,columns]
        return
    # The complete q x reflection grid replaces the old five diagonal pairs.
    for q in TAILS:
        for reflection in REFLECTIONS:
            repaired=reflected_tails(y,q,reflection);ids=response_balanced(repaired,budget)
            if family in ['band','mixed']:
                for radius in RADII:
                    A=X[ids].copy();T=Q.copy();lo=center[top]-radius*scale[top];hi=center[top]+radius*scale[top]
                    A[:,top]=np.clip(A[:,top],lo,hi);T[:,top]=np.clip(T[:,top],lo,hi)
                    yield f'tail_{q}_reflection_{reflection}_clip_{radius}',A,repaired[ids],T
            elif family=='local':
                for fraction in [.2,.4,.6,.8,1.]:
                    count=max(8,int(np.ceil(len(y)*fraction)));subset=np.argsort(distance,kind='stable')[:count]
                    rows=response_balanced(repaired[subset],budget)
                    yield f'local_{fraction}_tail_{q}_reflection_{reflection}',X[subset][rows],repaired[subset][rows],Q
            else:yield f'tail_{q}_reflection_{reflection}',X[ids],repaired[ids],Q
