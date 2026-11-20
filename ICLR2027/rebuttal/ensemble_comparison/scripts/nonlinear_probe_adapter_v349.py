"""Nonlinear readout on existing TFM features; unchanged target-label CV."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from run_tabicl import normalize
from linear_probe_adapter_v344 import score

LEAVES=(1,3,8)
TREES=128


def fit(X,y,leaf,weights,trees=TREES):
    model=ExtraTreesClassifier(n_estimators=trees,min_samples_leaf=leaf,
        max_features='sqrt',n_jobs=1,random_state=0,bootstrap=False)
    return model.fit(np.asarray(X,dtype=np.float32),np.asarray(y,dtype=int),sample_weight=weights)


def predict(model,X,K):
    p=model.predict_proba(np.asarray(X,dtype=np.float32))
    out=np.zeros((len(X),K),dtype=float)
    out[:,np.asarray(model.classes_,dtype=int)]=p
    return normalize(out)


def adapt(F,raw,reference_y,source_validation_y,adaptation_y,apos,epos,trees=TREES):
    """No evaluation labels accepted; only the readout family differs from v344."""
    F=np.asarray(F,dtype=np.float32);raw=normalize(raw);K=raw.shape[1]
    m,s=len(reference_y),len(source_validation_y);offset=m+s;folds=np.arange(len(apos))%4
    reference=np.arange(m);source_val=np.arange(m,offset)
    target=offset+apos;evaluation=offset+epos
    best={'kind':'raw','min_samples_leaf':None,'alpha':0.,
        'score':list(score(adaptation_y,raw[target],source_validation_y,raw[source_val]))}
    trials=[];oof={}
    for leaf in LEAVES:
        cv=np.full((len(apos),K),np.nan);cal=np.zeros((s,K))
        for fold in range(4):
            tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            rows=np.r_[reference,target[tr]];labels=np.r_[reference_y,adaptation_y[tr]]
            model=fit(F[rows],labels,leaf,np.r_[np.ones(m),np.full(len(tr),4.)],trees)
            cv[va]=predict(model,F[target[va]],K);cal+=predict(model,F[source_val],K)/4.
        assert np.isfinite(cv).all()
        oof[str(leaf)]={'target':cv,'source_validation':cal}
        for alpha in (.25,.5,.75,1.):
            pt=normalize((1-alpha)*raw[target]+alpha*cv)
            ps=normalize((1-alpha)*raw[source_val]+alpha*cal)
            value=score(adaptation_y,pt,source_validation_y,ps)
            trials.append({'min_samples_leaf':leaf,'alpha':alpha,'score':list(value)})
            if value>tuple(best['score']):best={'kind':'extra_trees_readout','min_samples_leaf':leaf,'alpha':alpha,'score':list(value)}
    model=None
    if best['kind']=='extra_trees_readout':
        rows=np.r_[reference,target];labels=np.r_[reference_y,adaptation_y]
        model=fit(F[rows],labels,best['min_samples_leaf'],np.r_[np.ones(m),np.full(len(target),4.)],trees)
        final=normalize((1-best['alpha'])*raw[evaluation]+best['alpha']*predict(model,F[evaluation],K))
    else:final=raw[evaluation].copy()
    best.update(trials=trials,model=model,reference_training_rows=m,source_validation_rows=s,
        target_adaptation_rows=len(target),target_training_weight=4.,trees=trees,max_features='sqrt',
        selection='target four-fold accuracy, then equal-domain NLL, then source accuracy')
    return final,best,oof
