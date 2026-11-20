"""Conservative per-task residual adaptation with source-scale bounds.

Guards are defined uniformly, then selected separately on each adaptation set.
Regression clips training residuals and corrections using source-label scale;
it uses bounded squared validation loss to avoid one rare extreme target
label moving predictions for every remaining row. No evaluation labels enter.
"""
import numpy as np
from task_residual_kernel_v313 import distances
from run_tabicl import normalize


def loss(y,p,task,scale):
    if task=='classification':return (np.argmax(p,axis=1)!=y).astype(float)
    return np.minimum(((p-y)/scale)**2,4.)


def score(y,p,task,scale):
    if task=='classification':
        p=normalize(p)
        return (float(np.mean(p.argmax(1)==y)),float(np.log(np.maximum(p[np.arange(len(y)),y.astype(int)],1e-12)).mean()))
    return (-float(loss(y,p,task,scale).mean()),-float(np.mean((p-y)**2)))


def correction(raw,task,scale,strength):
    if task=='classification':return raw*strength
    return np.clip(raw*strength,-.25*scale,.25*scale)


def fit(features,base,y,task,scale):
    F=np.asarray(features,dtype=float);p=np.asarray(base,dtype=float);y=np.asarray(y)
    scale=max(float(scale),1e-12)
    d=distances(F,F);positive=d[d>1e-12]
    bandwidth=float(np.median(positive)) if len(positive) else 1.
    identity_score=score(y,p,task,scale)
    best={'kind':'identity','score':list(identity_score)};trials=[]
    target=np.eye(p.shape[1])[y.astype(int)] if task=='classification' else y
    residual=target-p
    if task=='regression':residual=np.clip(residual,-2*scale,2*scale)
    baseline_loss=loss(y,p,task,scale)
    for multiplier in (.5,1.,2.):
        K=np.exp(-d/(bandwidth*multiplier))
        for regularization in (.01,.1,1.,10.):
            inverse=np.linalg.inv(K+regularization*np.eye(len(y)))
            alpha=inverse@residual;diagonal=np.diag(inverse)
            raw=residual-alpha/(diagonal[:,None] if p.ndim==2 else diagonal)
            for strength in (.25,.5,1.):
                heldout=p+correction(raw,task,scale,strength)
                if task=='classification':heldout=normalize(heldout)
                value=score(y,heldout,task,scale)
                delta=baseline_loss-loss(y,heldout,task,scale)
                gain=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(y)))
                eligible=gain>max(se,1e-12)
                if task=='classification':
                    eligible=eligible and gain>=2/len(y)-1e-12 and value[1]>=identity_score[1]
                trial={'bandwidth_multiplier':multiplier,'regularization':regularization,'strength':strength,
                    'LOO_score':list(value),'paired_loss_gain':gain,'paired_standard_error':se,'eligible':bool(eligible)}
                trials.append(trial)
                if eligible and value>tuple(best['score']):
                    best={'kind':'kernel','score':list(value),'bandwidth':bandwidth,
                        'bandwidth_multiplier':multiplier,'regularization':regularization,'strength':strength,
                        'alpha':alpha.tolist()}
    best.update(adaptation_rows=len(y),selection='adaptation-only exact LOO with paired one-standard-error guard',
        source_label_scale=scale,regression_training_residual_cap_in_source_std=2.,
        regression_output_correction_cap_in_source_std=.25,trials=trials)
    return best


def predict(features,base,adaptation_features,model,task):
    p=np.asarray(base,dtype=float)
    if model['kind']=='identity':return p.copy()
    K=np.exp(-distances(np.asarray(features,dtype=float),np.asarray(adaptation_features,dtype=float))/
        (model['bandwidth']*model['bandwidth_multiplier']))
    prediction=p+correction(K@np.asarray(model['alpha']),task,model['source_label_scale'],model['strength'])
    return normalize(prediction) if task=='classification' else prediction
