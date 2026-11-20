"""Fit convex deployment weights on target-adaptation cross-fit predictions."""
import numpy as np
from run_tabicl import normalize


def score(y,p,task,scale):
    if task=='classification':
        p=normalize(p)
        return (float(np.mean(np.argmax(p,axis=1)==y)),float(np.mean(np.log(np.maximum(p[np.arange(len(y)),y.astype(int)],1e-12)))))
    return (-float(np.mean(np.minimum(((p-y)/scale)**2,4.))),-float(np.mean((p-y)**2)))


def loss(y,p,task,scale):
    if task=='classification':return (np.argmax(p,axis=1)!=y).astype(float)
    return np.minimum(((p-y)/scale)**2,4.)


def select(y,cv,task,scale):
    """cv shape (3, adaptation_rows, [classes]); no evaluation data accepted."""
    assert len(cv)==3 and cv.shape[1]==len(y)
    scale=max(float(scale),1e-12)
    initial=score(y,cv[0],task,scale);baseline_loss=loss(y,cv[0],task,scale)
    best={'weights':[1.,0.,0.],'score':list(initial)};trials=[]
    for a in range(5):
        for b in range(5-a):
            weights=np.array([a,b,4-a-b])/4.
            p=np.tensordot(weights,cv,axes=1)
            if task=='classification':p=normalize(p)
            delta=baseline_loss-loss(y,p,task,scale)
            gain=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(y)))
            value=score(y,p,task,scale)
            eligible=gain>max(se,1e-12)
            if task=='classification':eligible=eligible and gain>=2/len(y)-1e-12 and value[1]>=initial[1]
            trials.append({'weights':weights.tolist(),'score':list(value),'loss_gain':gain,'paired_SE':se,'eligible':bool(eligible)})
            if eligible and value>tuple(best['score']):best={'weights':weights.tolist(),'score':list(value)}
    return best|{'trials':trials,'adaptation_rows':len(y),'source_scale':scale,
        'selection':'fixed four-fold target-label cross-fit; paired one-SE improvement guard',
        'components':['original frozen Base','source-finetuned same method','reference-pool and target-context adapted same method']}


def apply(policy,predictions,task):
    p=np.tensordot(np.asarray(policy['weights']),predictions,axes=1)
    return normalize(p) if task=='classification' else p
