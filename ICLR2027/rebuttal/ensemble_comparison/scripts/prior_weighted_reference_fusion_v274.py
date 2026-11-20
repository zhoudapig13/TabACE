"""One-factor ablation: source-prior weighting of reference calibration CE."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
import reference_convex_fusion_v271 as parent


def fit(P, y_calibration, y_source, task, scale):
    if task=='regression':
        return parent.fit(P,y_calibration,task,scale)
    P,y=np.asarray(P,dtype=np.float64),np.asarray(y_calibration,dtype=int)
    assert P.shape[0]==4 and P.shape[1]==len(y)
    K=P.shape[2]
    a=np.bincount(np.asarray(y_source,dtype=int),minlength=K)/len(y_source)
    b=np.bincount(y,minlength=K)/len(y)
    sample=np.minimum(a[y]/b[y],10.)
    sample/=sample.sum()
    A=P[:,np.arange(len(y)),y]
    prior=np.array([.5,.5,0.,0.]);regularization=.02
    def objective(logits):
        w=softmax(logits);p=w@A
        loss=-np.sum(sample*np.log(np.maximum(p,1e-12)))
        gradient=-np.sum(A/np.maximum(p,1e-12)*sample,axis=1)
        loss+=regularization*np.sum((w-prior)**2)
        gradient+=2*regularization*(w-prior)
        return float(loss),w*(gradient-w@gradient)
    result=minimize(objective,np.array([2.,2.,0.,0.]),jac=True,method='L-BFGS-B',
        bounds=[(-8.,8.)]*4,options={'maxiter':200,'ftol':1e-11})
    assert np.isfinite(result.x).all()
    return {'weights':softmax(result.x).tolist(),'regularization':regularization,
        'calibration_rows':len(y),'optimizer_success':bool(result.success),'loss':float(result.fun),
        'source_prior':a.tolist(),'reference_prior':b.tolist(),'importance_cap':10.,
        'effective_calibration_rows':float(1/np.sum(sample**2))}


predict=parent.predict
