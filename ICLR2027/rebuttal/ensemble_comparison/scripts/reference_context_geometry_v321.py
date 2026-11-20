"""Class-covering local membership without imposed source class proportions."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import robust_scale_shared_contexts_v290 as geometry_base


def select_rows(distance,y,task,budget=256):
    order=np.argsort(distance,kind='stable');budget=min(budget,len(y))
    if task=='regression':return order[:budget]
    mandatory=np.array([next(int(i) for i in order if y[i]==label) for label in np.unique(y)],dtype=int)
    assert len(mandatory)<=budget
    chosen=set(mandatory.tolist())
    rest=np.array([i for i in order if int(i) not in chosen],dtype=int)
    rows=np.r_[mandatory,rest[:budget-len(mandatory)]]
    return rows[np.argsort(distance[rows],kind='stable')]


def construct(X,y,Q,task):
    Z,V,center,scale=geometry_base.robust_geometry(X,Q)
    if Z.shape[1]>64:
        matrix=np.random.default_rng(0).normal(size=(Z.shape[1],32)).astype(np.float32)/np.sqrt(32.)
        Z,V=Z@matrix,V@matrix
    Z,V=Z.astype(np.float32),V.astype(np.float32)
    labels=KMeans(n_clusters=min(4,len(Q)),random_state=0,n_init=5).fit_predict(V)
    groups=[]
    for group in np.unique(labels):
        positions=np.flatnonzero(labels==group)
        D=pairwise_distances(Z,V[positions],metric='sqeuclidean')
        k=min(10,len(positions));distance=np.partition(D,k-1,axis=1)[:,:k].mean(axis=1)
        rows=select_rows(distance,y,task)
        assert len(set(rows.tolist()))==len(rows) and len(rows)<=256
        if task=='classification':np.testing.assert_array_equal(np.unique(y[rows]),np.unique(y))
        groups.append({'group':int(group),'query_positions':positions,'pool_positions':rows})
    assert sorted(np.concatenate([g['query_positions'] for g in groups]).tolist())==list(range(len(Q)))
    return groups
