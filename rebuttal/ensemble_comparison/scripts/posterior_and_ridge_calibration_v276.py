"""Fixed source-prior alignment and independent-reference ridge calibration."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


def align_probability_marginal(p, y_source):
    p=np.asarray(p,dtype=np.float64)
    K=p.shape[1]
    prior=np.bincount(np.asarray(y_source,dtype=int),minlength=K)/len(y_source)
    assert len(prior)==K and np.all(prior>0)
    logp=np.log(np.maximum(p,1e-12))
    logp-=logsumexp(logp,axis=1)[:,None]
    def objective(b):
        logits=logp+b
        q=softmax(logits,axis=1)
        return float(logsumexp(logits,axis=1).mean()-prior@b),q.mean(0)-prior
    result=minimize(objective,np.zeros(K),jac=True,method='L-BFGS-B',
        options={'maxiter':300,'ftol':1e-13,'gtol':1e-9})
    b=result.x-result.x.mean()
    pred=softmax(logp+b,axis=1)
    assert np.isfinite(pred).all()
    return pred,{'source_prior':prior.tolist(),'log_class_correction':b.tolist(),
        'maximum_marginal_error':float(np.max(np.abs(pred.mean(0)-prior))),
        'optimizer_success':bool(result.success),
        'target_labels_used':False,'source_target_prior_stability_assumed':True}


def fit_ridge(P_reference,y_reference,y_source,anchor):
    P=np.asarray(P_reference,dtype=np.float64)
    y=np.asarray(y_reference,dtype=np.float64)
    anchor=np.asarray(anchor,dtype=np.float64)
    assert P.shape==(len(anchor),len(y))
    mu=float(np.mean(y_source));scale=max(float(np.std(y_source)),1e-12)
    A=np.c_[((P-mu)/scale).T,np.ones(len(y))]
    target=(y-mu)/scale
    prior=np.r_[anchor,0.]
    penalty=.1*np.diag(np.r_[np.ones(len(anchor)),.1])
    theta=np.linalg.solve(A.T@A/len(y)+penalty,A.T@target/len(y)+penalty@prior)
    assert np.isfinite(theta).all()
    return {'beta':theta[:-1].tolist(),'intercept':float(theta[-1]),'mu':mu,'scale':scale,
        'regularization':.1,'intercept_penalty_multiplier':.1,
        'calibration_rows':len(y),'target_labels_used':False}


def predict_ridge(P,model):
    P=np.asarray(P,dtype=np.float64)
    return (np.asarray(model['beta'])@((P-model['mu'])/model['scale'])+model['intercept'])*model['scale']+model['mu']
