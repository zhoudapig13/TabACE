"""Task-specific residual calibration selected by exact label-subset LOO.

No evaluation labels enter this API. TFM contexts and hidden features contain
no target adaptation labels, so LOO is not contaminated by a teacher trained
on its held-out calibration label.
"""
import numpy as np
from run_tabicl import normalize


def distances(A,B):
    return np.maximum((A*A).sum(1)[:,None]+(B*B).sum(1)[None]-2*A@B.T,0.)


def score(y,p,task):
    if task=='classification':
        p=normalize(p)
        return (float(np.mean(p.argmax(1)==y)),float(np.log(np.maximum(p[np.arange(len(y)),y.astype(int)],1e-12)).mean()))
    return (-float(np.mean((p-y)**2)),0.)


def fit(features,base,y,task):
    F=np.asarray(features,dtype=np.float64);p=np.asarray(base,dtype=np.float64);y=np.asarray(y)
    assert len(F)==len(p)==len(y) and np.isfinite(F).all() and np.isfinite(p).all()
    d=distances(F,F);positive=d[d>1e-12]
    bandwidth=float(np.median(positive)) if len(positive) else 1.
    best={'kind':'identity','score':list(score(y,p,task))};trials=[]
    target=np.eye(p.shape[1])[y.astype(int)] if task=='classification' else y
    residual=target-p
    for multiplier in (.5,1.,2.):
        K=np.exp(-d/(bandwidth*multiplier))
        for regularization in (.01,.1,1.,10.):
            inverse=np.linalg.inv(K+regularization*np.eye(len(y)))
            alpha=inverse@residual;diagonal=np.diag(inverse)
            heldout=p+residual-alpha/(diagonal[:,None] if p.ndim==2 else diagonal)
            value=score(y,heldout,task)
            trials.append({'bandwidth_multiplier':multiplier,'regularization':regularization,'LOO_score':list(value)})
            if value>tuple(best['score']):
                best={'kind':'kernel','score':list(value),'bandwidth':bandwidth,
                    'bandwidth_multiplier':multiplier,'regularization':regularization,'alpha':alpha.tolist()}
    best.update(adaptation_rows=len(y),selection='exact leave-one-out on adaptation labels only',trials=trials)
    return best


def predict(features,base,adaptation_features,model,task):
    p=np.asarray(base,dtype=np.float64)
    if model['kind']=='identity':return p.copy()
    K=np.exp(-distances(np.asarray(features,dtype=np.float64),np.asarray(adaptation_features,dtype=np.float64))/
             (model['bandwidth']*model['bandwidth_multiplier']))
    prediction=p+K@np.asarray(model['alpha'])
    return normalize(prediction) if task=='classification' else prediction
