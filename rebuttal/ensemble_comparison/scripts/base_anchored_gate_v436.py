"""Calibration-only selection of a mixture with the original backbone anchor."""
import numpy as np
from run_tabicl import normalize

BETAS = (0., .25, .5, .75, 1.)


def select(y, base_cv, probe_cv):
    y=np.asarray(y,dtype=int)
    base_cv,probe_cv=normalize(base_cv),normalize(probe_cv)
    base_acc=float(np.mean(base_cv.argmax(1)==y))
    trials=[]
    best={'beta':0.,'accuracy':base_acc,'nll':-float(np.log(np.maximum(base_cv[np.arange(len(y)),y],1e-12)).mean())}
    for beta in BETAS[1:]:
        p=normalize((1-beta)*base_cv+beta*probe_cv)
        accuracy=float(np.mean(p.argmax(1)==y))
        nll=-float(np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
        trials.append({'beta':beta,'accuracy':accuracy,'nll':nll})
        # Original Base stays active unless the permitted calibration subset
        # has strictly fewer mistakes. The evaluation labels never enter here.
        if accuracy>base_acc and (accuracy,-nll,-beta)>(best['accuracy'],-best['nll'],-best['beta']):
            best={'beta':beta,'accuracy':accuracy,'nll':nll}
    return dict(best,trials=trials,base_accuracy=base_acc,
                rule='strict calibration-accuracy improvement over original Base; NLL then smaller beta tie-break')


def apply(policy, base, probe):
    beta=policy['beta']
    return normalize((1-beta)*base+beta*probe)
