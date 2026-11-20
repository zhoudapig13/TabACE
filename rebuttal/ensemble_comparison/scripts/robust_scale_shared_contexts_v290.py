"""Five fixed Shared roles on a larger permitted labeled pool.

Classification selection preserves observed-source class proportions; regression
retains the existing five rules with a target-local canonical256-row context.
The algorithm receives neither generator identity nor final evaluation labels.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import unseen_shared_robustness_v117 as source
import pooled_shared_contexts_v279 as parent


def robust_geometry(X,Q):
    center=np.median(X,axis=0)
    iqr=np.percentile(X,75,axis=0)-np.percentile(X,25,axis=0)
    sd=np.std(X,axis=0,dtype=np.float64)
    scale=np.where(iqr>1e-8,iqr,np.where(sd>1e-8,sd,1.))
    return (X-center)/scale,(Q-center)/scale,center,scale


def robust_regression_contexts(X,y,Q,canonical_X,canonical_y,budget,task):
    assert task=='regression'
    Z,V,center,scale=robust_geometry(X,Q)
    rng=np.random.default_rng(0)
    bootstrap=rng.choice(len(y),size=budget,replace=True)
    ordered=np.argsort(y,kind='stable');parts=[p for p in np.array_split(ordered,8) if len(p)]
    counts=np.full(len(parts),budget//len(parts));counts[:budget%len(parts)]+=1
    balanced=np.concatenate([rng.choice(p,int(n),replace=True) for p,n in zip(parts,counts)])
    rng.shuffle(balanced)
    matched=source.constructors.shared.matched_rows(Z,V,budget,0)
    target_center=np.median(Q,axis=0)
    target_scale=np.maximum(np.percentile(Q,75,axis=0)-np.percentile(Q,25,axis=0),1e-6)
    aligned=Z*target_scale+target_center
    canonical_aligned=(canonical_X-center)/scale*target_scale+target_center
    aligned_Z=(aligned-center)/scale
    aligned_matched=source.constructors.shared.matched_rows(aligned_Z,V,budget,0)
    return [('uniform_bootstrap',X[bootstrap],y[bootstrap],Q),
        ('response_balanced',X[balanced],y[balanced],Q),
        ('target_matched',X[matched],y[matched],Q),
        ('distribution_aligned',canonical_aligned,canonical_y,Q),
        ('aligned_target_matched',aligned[aligned_matched],y[aligned_matched],Q)]


def balanced_rows(score,y,source_y,n):
    counts=np.bincount(np.asarray(source_y,dtype=int))
    assert np.all(counts>0)
    n=min(n,len(y));assert n>=len(counts)
    desired=n*counts/counts.sum()
    quota=np.maximum(1,np.floor(desired).astype(int))
    while quota.sum()>n:
        permitted=np.flatnonzero(quota>1)
        j=permitted[np.argmax((quota-desired)[permitted])];quota[j]-=1
    while quota.sum()<n:
        j=int(np.argmax(desired-quota));quota[j]+=1
    chosen=[]
    for label,k in enumerate(quota):
        ids=np.flatnonzero(y==label)
        assert len(ids)>=k
        chosen.extend(ids[np.argsort(score[ids],kind='stable')[:k]].tolist())
    chosen=np.asarray(chosen,dtype=int)
    return chosen[np.argsort(score[chosen],kind='stable')]


def geometry(X,y,Q,C,task):
    Z,V,center,scale=robust_geometry(X,Q)
    W=(C-center)/scale
    matrix=None
    if Z.shape[1]>64:
        matrix=np.random.default_rng(0).normal(size=(Z.shape[1],32)).astype(np.float32)/np.sqrt(32.)
    A=Z.astype(np.float32) if matrix is None else np.asarray(Z@matrix,dtype=np.float32)
    T=V.astype(np.float32) if matrix is None else np.asarray(V@matrix,dtype=np.float32)
    U=W.astype(np.float32) if matrix is None else np.asarray(W@matrix,dtype=np.float32)
    clusters=KMeans(n_clusters=min(4,len(Q)),random_state=0,n_init=5).fit(T)
    out={'A':A,'T':T,'U':U,'query_groups':clusters.labels_,
         'reference_groups':clusters.predict(U),'centers':clusters.cluster_centers_}
    active=np.unique(out['query_groups'])
    missing=~np.isin(out['reference_groups'],active)
    if np.any(missing):
        dist=pairwise_distances(U[missing],out['centers'][active],metric='sqeuclidean')
        out['reference_groups'][missing]=active[np.argmin(dist,axis=1)]
    assert np.isin(out['reference_groups'],active).all()
    if task=='classification':
        nn=NearestNeighbors(n_neighbors=min(11,len(y)),n_jobs=1).fit(A)
        ids=nn.kneighbors(A,return_distance=False)
        neighbors=np.stack([row[row!=i][:min(10,len(y)-1)] for i,row in enumerate(ids)])
        consensus=np.max(np.stack([(y[neighbors]==label).mean(1) for label in np.unique(y)],axis=1),axis=1)
        out['disagreement']=1-consensus
        out['outlying']=(np.minimum(np.abs(A),10.)**2).mean(1)
        order=source.constructors.specific.importance(X,y.astype(float))
        columns=np.asarray([j for j in range(X.shape[1]) if j!=int(order[0])])
        out['columns']=columns if len(columns) else np.asarray([0])
    return out


def contexts(X,y,Q,C,source_y,task,model,group):
    qi=np.flatnonzero(model['query_groups']==group)
    ci=np.flatnonzero(model['reference_groups']==group)
    assert len(qi)>0
    Qg=Q[qi];predict_X=np.concatenate([Qg,C[ci]])
    A,T=model['A'],model['T'][qi]
    distances=pairwise_distances(A,T,metric='sqeuclidean')
    k=min(10,len(T));near=np.partition(distances,k-1,axis=1)[:,:k].mean(1)
    if task=='classification':
        canonical=balanced_rows(near,y,source_y,256)
        anchors=np.quantile(T,[.1,.3,.5,.7,.9],axis=0)
        anchor=pairwise_distances(A,anchors,metric='sqeuclidean').min(1)
        outlier,disagree=model['outlying'],model['disagreement']
        specs=[('target_neighbor',anchor,.8),('representative',near+.35*outlier,.7),
               ('locally_stable',disagree+.10*near,.8),('low_conflict',disagree+.20*outlier,.75)]
        views=[]
        for name,score,fraction in specs:
            n=min(256,max(len(np.unique(y)),int(np.ceil(len(y)*fraction))))
            rows=balanced_rows(score,y,source_y,n)
            views.append((name,X[rows],y[rows],predict_X))
        columns=model['columns']
        views.insert(3,('feature_subspace',X[canonical][:,columns],y[canonical],predict_X[:,columns]))
    else:
        canonical=np.argsort(near,kind='stable')[:256]
        # Construction uses target queries only. Unlabeled calibration features
        # are appended only to the prediction arrays, not constructor fitting.
        original=robust_regression_contexts(X,y,Qg,X[canonical],y[canonical],256,task)
        views=[]
        for name,V,labels,target in original:
            np.testing.assert_array_equal(target,Qg)
            views.append((name,V,labels,predict_X))
    return qi,ci,canonical,predict_X,views
