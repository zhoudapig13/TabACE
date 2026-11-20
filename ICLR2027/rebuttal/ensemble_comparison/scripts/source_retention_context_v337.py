"""Keep a tunable number of source rows, then append target adaptation labels."""
import numpy as np
from run_tabicl import normalize

COUNTS=(0,32,96,224)


def context(name,A,b,T,AX,ay,requested,columns=None):
    assert requested in COUNTS
    if requested==0:rows=np.array([],dtype=int)
    else:
        rng=np.random.default_rng(0)
        mandatory=np.array([rng.choice(np.flatnonzero(b==label)) for label in np.unique(b)],dtype=int)
        n=min(len(b),256-len(ay),max(requested,len(mandatory)))
        assert len(mandatory)<=n
        rest=np.setdiff1d(np.arange(len(b)),mandatory)
        rows=np.r_[mandatory,rng.choice(rest,n-len(mandatory),replace=False)];rng.shuffle(rows)
    target=AX[:,columns] if name=='feature_subspace' else AX
    X,y=np.r_[A[rows],target],np.r_[b[rows],ay]
    assert len(y)<=256 and X.shape[1]==T.shape[1] and len(ay)>0
    return X,y,{'source_rows':len(rows),'target_rows':len(ay),'requested_source_rows':requested,'retained_candidate_positions':rows.tolist()}


def score(y,p):
    p=normalize(p)
    return (float(np.log(np.maximum(p[np.arange(len(y)),y.astype(int)],1e-12)).mean()),float(np.mean(p.argmax(axis=1)==y)))


def select(y,base,candidates):
    """Select by cross-fitted log loss, then accuracy, without evaluation labels."""
    best={'candidate':'Base','alpha':0.,'score':list(score(y,base))};trials=[]
    for name,p in candidates.items():
        for alpha in (.25,.5,.75,1.):
            prediction=normalize((1-alpha)*base+alpha*p);value=score(y,prediction)
            trials.append({'candidate':name,'alpha':alpha,'score':list(value)})
            if value>tuple(best['score']):best={'candidate':name,'alpha':alpha,'score':list(value)}
    return best|{'trials':trials,'selection':'four-fold adaptation-label cross-fitted log loss then accuracy','adaptation_rows':len(y)}


def apply(policy,base,candidate):
    return normalize((1-policy['alpha'])*base+policy['alpha']*candidate)
