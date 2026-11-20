"""One-dimensional calibration search across all class-decision intervals."""
import numpy as np
from scipy.optimize import minimize_scalar
from run_tabicl import normalize

BETAS=(0.,.25,.5,.75,1.)


def candidate_points(base, probe, y):
    delta=probe-base
    boundaries=[0.,1.]
    for i in range(base.shape[1]):
        for j in range(i+1,base.shape[1]):
            den=delta[:,i]-delta[:,j]
            ok=np.abs(den)>1e-14
            values=(base[:,j]-base[:,i])[ok]/den[ok]
            boundaries.extend(values[(values>0.)&(values<1.)].tolist())
    boundaries=np.unique(boundaries)
    nll=lambda b:-float(np.log(np.maximum((base+b*delta)[np.arange(len(y)),y],1e-12)).mean())
    optimum=minimize_scalar(nll,bounds=(0.,1.),method='bounded',options={'xatol':1e-10})
    star=float(min([(nll(0.),0.),(nll(1.),1.),(float(optimum.fun),float(optimum.x))])[1])
    points=list(BETAS)
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        midpoint=float((left+right)/2.)
        margin=min(1e-9,float(right-left)/4.)
        points.extend([midpoint,float(np.clip(star,left+margin,right-margin))])
    return np.unique(points)


def select(y, base_cv, probe_cv):
    y=np.asarray(y,dtype=int)
    base_cv,probe_cv=normalize(base_cv),normalize(probe_cv)
    base_acc=float(np.mean(base_cv.argmax(1)==y))
    best={'beta':0.,'accuracy':base_acc,'nll':-float(np.log(np.maximum(base_cv[np.arange(len(y)),y],1e-12)).mean())}
    trials=[]
    for beta in candidate_points(base_cv,probe_cv,y):
        p=normalize((1-beta)*base_cv+beta*probe_cv)
        accuracy=float(np.mean(p.argmax(1)==y))
        nll=-float(np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
        trials.append({'beta':float(beta),'accuracy':accuracy,'nll':nll})
        if accuracy>base_acc and (accuracy,-nll,-beta)>(best['accuracy'],-best['nll'],-best['beta']):
            best={'beta':float(beta),'accuracy':accuracy,'nll':nll}
    return dict(best,trials=trials,base_accuracy=base_acc,
                rule='strict calibration-accuracy improvement; all class-decision intervals searched; NLL then smaller beta tie-break')


def apply(policy,base,probe):
    return normalize((1-policy['beta'])*base+policy['beta']*probe)
