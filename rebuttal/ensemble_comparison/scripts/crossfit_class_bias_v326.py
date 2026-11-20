"""Low-dimensional class-intercept adaptation, with honest leave-one-out fits.

Only label-independent probability features are accepted by callers. Fitting
all leave-one-out intercepts in a batch is computationally cheaper than KRR
and reduces the number of fitted parameters to one offset per class.
"""
import numpy as np
from run_tabicl import normalize


def softmax(z):
    z=z-z.max(axis=-1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(axis=-1,keepdims=True)


def fit_all_and_loo(p,y,penalty):
    p=normalize(p);y=np.asarray(y,dtype=int);n,K=p.shape
    assert n>=4 and np.min(y)>=0 and np.max(y)<K and penalty>0
    W=np.ones((n+1,n),dtype=float)
    W[np.arange(1,n+1),np.arange(n)]=0.
    count=W.sum(axis=1)
    means=(W@np.eye(K)[y])/count[:,None]
    offset=np.zeros((n+1,K),dtype=float);logp=np.log(p)
    for _ in range(60):
        probability=softmax(logp[None,:,:]+offset[:,None,:])
        gradient=(W[:,:,None]*probability).sum(axis=1)/count[:,None]-means+penalty*offset
        # A global stopping test would couple otherwise independent LOO fits:
        # another fit's label could change this fit's iteration count. Keeping
        # the iteration count fixed preserves exact held-out-label isolation.
        offset=np.clip(offset-gradient/(.5+penalty),-3.,3.)
    assert np.isfinite(offset).all()
    loo=softmax(logp+offset[1:])
    return offset[0],loo,offset[1:]


def score(y,p):
    p=normalize(p)
    return (float(np.mean(p.argmax(axis=1)==y)),float(np.log(np.maximum(p[np.arange(len(y)),y.astype(int)],1e-12)).mean()))


def fit(candidates,y):
    """All candidate rows belong to the adaptation set, not evaluation rows."""
    best=None;trials=[]
    for name,p in candidates.items():
        raw_score=score(y,p)
        trial={'source':name,'penalty':None,'LOO_score':list(raw_score)};trials.append(trial)
        if best is None or raw_score>tuple(best['score']):
            best={'kind':'identity','source':name,'score':list(raw_score),'offset':np.zeros(p.shape[1]).tolist(),'loo':normalize(p)}
        for penalty in (.01,.1,1.):
            offset,loo,_=fit_all_and_loo(p,y,penalty)
            value=score(y,loo)
            trials.append({'source':name,'penalty':penalty,'LOO_score':list(value)})
            if value>tuple(best['score']):
                best={'kind':'class_intercept','source':name,'penalty':penalty,'score':list(value),'offset':offset.tolist(),'loo':loo}
    loo=best.pop('loo');best['trials']=trials;best['adaptation_rows']=len(y)
    return best,loo


def apply(policy,candidates):
    p=normalize(candidates[policy['source']])
    if policy['kind']=='identity':return p
    return softmax(np.log(p)+np.asarray(policy['offset']))
