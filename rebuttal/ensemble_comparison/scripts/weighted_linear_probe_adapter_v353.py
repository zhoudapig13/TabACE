"""Class-prior weighted linear TFM readout; target priors use training-fold labels only."""
import numpy as np
from scipy.special import expit,softmax
from sklearn.linear_model import LogisticRegression
from run_tabicl import normalize

PENALTIES=(.01,.1,1.,10.)


def fit(X,y,C,weights):
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=int)
    mean=X.mean(axis=0);sd=X.std(axis=0);scale=np.where(sd>1e-8,sd,1.)
    classes=np.unique(y)
    if len(classes)==1:
        return {'kind':'constant','classes':classes.tolist(),'mean':mean.tolist(),'scale':scale.tolist()}
    model=LogisticRegression(C=C,solver='lbfgs',max_iter=1000,random_state=0)
    model.fit((X-mean)/scale,y,sample_weight=weights)
    return {'kind':'logistic','classes':model.classes_.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
        'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'iterations':model.n_iter_.tolist(),'C':C}


def predict(model,X,K):
    X=np.asarray(X,dtype=float);classes=np.array(model['classes'],dtype=int)
    out=np.zeros((len(X),K),dtype=float)
    if model['kind']=='constant':out[:,classes[0]]=1.;return out
    score=((X-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['coef']).T+np.asarray(model['intercept'])
    if len(classes)==2:
        p=expit(score[:,0]);local=np.c_[1-p,p]
    else:local=softmax(score,axis=1)
    out[:,classes]=local
    return normalize(out)


def score(target_y,target_p,source_y,source_p):
    target_p,source_p=normalize(target_p),normalize(source_p)
    target_acc=float(np.mean(target_p.argmax(axis=1)==target_y))
    source_acc=float(np.mean(source_p.argmax(axis=1)==source_y))
    target_nll=-float(np.log(np.maximum(target_p[np.arange(len(target_y)),target_y.astype(int)],1e-12)).mean())
    source_nll=-float(np.log(np.maximum(source_p[np.arange(len(source_y)),source_y.astype(int)],1e-12)).mean())
    return (target_acc,-.5*(target_nll+source_nll),source_acc)


def training_weights(reference_y,target_y,prior_probabilities,K):
    labels=np.r_[reference_y,target_y].astype(int)
    base=np.r_[np.ones(len(reference_y)),np.full(len(target_y),4.)]
    prior=normalize(np.asarray(prior_probabilities)).mean(axis=0)
    desired=(np.bincount(np.asarray(target_y,dtype=int),minlength=K)+8.*prior)/(len(target_y)+8.)
    observed=np.bincount(labels,weights=base,minlength=K)/base.sum()
    ratios=np.clip(desired/np.maximum(observed,1e-12),.05,20.)
    weights=base*ratios[labels];weights*=base.sum()/weights.sum()
    return weights,{'desired_prior':desired.tolist(),'observed_weighted_prior':observed.tolist(),'class_factors':ratios.tolist()}


def adapt(F,raw,reference_y,source_validation_y,adaptation_y,apos,epos):
    """No held-out evaluation-label argument. All query positions are features."""
    F=np.asarray(F,dtype=float);raw=normalize(raw);K=raw.shape[1]
    m,s=len(reference_y),len(source_validation_y);offset=m+s;folds=np.arange(len(apos))%4
    reference=np.arange(m);source_val=np.arange(m,offset)
    target=offset+apos;evaluation=offset+epos
    best={'kind':'raw','C':None,'alpha':0.,'score':list(score(adaptation_y,raw[target],source_validation_y,raw[source_val]))}
    trials=[];oof={}
    for C in PENALTIES:
        cv=np.full((len(apos),K),np.nan);cal=np.zeros((s,K))
        for fold in range(4):
            tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            rows=np.r_[reference,target[tr]];labels=np.r_[reference_y,adaptation_y[tr]]
            weights,_=training_weights(reference_y,adaptation_y[tr],raw[target[tr]],K)
            model=fit(F[rows],labels,C,weights)
            cv[va]=predict(model,F[target[va]],K);cal+=predict(model,F[source_val],K)/4.
        assert np.isfinite(cv).all()
        oof[str(C)]={'target':cv,'source_validation':cal}
        for alpha in (.25,.5,.75,1.):
            pt=normalize((1-alpha)*raw[target]+alpha*cv)
            ps=normalize((1-alpha)*raw[source_val]+alpha*cal)
            value=score(adaptation_y,pt,source_validation_y,ps)
            trials.append({'C':C,'alpha':alpha,'score':list(value)})
            if value>tuple(best['score']):best={'kind':'linear_probe','C':C,'alpha':alpha,'score':list(value)}
    model=None;weight_info=None
    if best['kind']=='linear_probe':
        rows=np.r_[reference,target];labels=np.r_[reference_y,adaptation_y]
        weights,weight_info=training_weights(reference_y,adaptation_y,raw[target],K)
        model=fit(F[rows],labels,best['C'],weights)
        final=normalize((1-best['alpha'])*raw[evaluation]+best['alpha']*predict(model,F[evaluation],K))
    else:final=raw[evaluation].copy()
    best.update(trials=trials,model=model,reference_training_rows=m,source_validation_rows=s,
        target_adaptation_rows=len(target),target_training_weight=4.,
        class_prior_weighting=weight_info,target_prior_smoothing_mass=8.,class_weight_ratio_clip=[.05,20.],
        selection='target four-fold accuracy, then equal-domain NLL, then source accuracy')
    return final,best,oof
