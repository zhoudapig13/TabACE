"""Label-free target preprocessing and source-only reliability diagnostics.

No function receives target labels, shift families, strengths or paper scores.
Six candidate roles are fixed before evaluating this adaptation attempt.
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import QuantileTransformer


def standardized(X,Q):
    mean=X.mean(0);scale=X.std(0);scale=np.where(scale>1e-8,scale,1.)
    return np.clip((X-mean)/scale,-20,20),np.clip((Q-mean)/scale,-20,20)


def density_and_proxy(X,y,Q):
    Z,V=standardized(X,Q);classes=np.unique(y)
    centers=np.stack([Z[y==label].mean(0) for label in classes])
    source=pairwise_distances(Z,centers,metric='sqeuclidean').min(1)/max(Z.shape[1],1)
    target=pairwise_distances(V,centers,metric='sqeuclidean').min(1)/max(Z.shape[1],1)
    bandwidth=max(float(np.median(source)),float(np.percentile(source,75))*.25,1e-4)
    density=np.exp(np.clip(-target/bandwidth,-50,0))+1e-8
    _,ids=np.unique(X,axis=0,return_index=True);ids=np.sort(ids)
    if len(ids)<3:return density,float(np.max(np.unique(y,return_counts=True)[1])/len(y)),source,target
    distance=pairwise_distances(Z[ids],metric='sqeuclidean');np.fill_diagonal(distance,np.inf)
    neighbors=distance.argmin(1);proxy=float(np.mean(y[ids][neighbors]==y[ids]))
    return density,proxy,source,target


def empirical_drift(X,Q):
    grid=np.linspace(.1,.9,9);threshold=np.quantile(X,grid,axis=0)
    source=(X[:,None,:]<=threshold[None,:,:]).mean(0)
    target=(Q[:,None,:]<=threshold[None,:,:]).mean(0)
    return np.max(np.abs(source-target),axis=0)


def recent_disagreement(history,recent):
    if history is None or recent is None:return 0.
    H,hy=history;R,ry=recent
    if H is None or R is None or len(hy)<12 or len(ry)<8:return 0.
    Z,V=standardized(H,R);d=pairwise_distances(Z,metric='sqeuclidean');np.fill_diagonal(d,np.inf)
    k=min(10,len(H)-1);source=np.argpartition(d,k-1,axis=1)[:,:k]
    target_d=pairwise_distances(V,Z,metric='sqeuclidean');target=np.argpartition(target_d,k-1,axis=1)[:,:k]
    classes=np.unique(hy)
    def vote(neighbors):return classes[np.stack([(hy[neighbors]==label).mean(1) for label in classes],axis=1).argmax(1)]
    return max(0.,float(np.mean(vote(target)!=ry)-np.mean(vote(source)!=hy)))


def construct(X,y,Q,*,history=None,recent=None,force_views=False):
    X=np.asarray(X,dtype=np.float64);Q=np.asarray(Q,dtype=np.float64)
    drift=empirical_drift(X,Q);Z,V=standardized(X,Q)
    _,_,source_distance,target_distance=density_and_proxy(X,y,Q)
    cutoff=max(float(np.percentile(source_distance,95))*1.5,1e-3)
    outlier_fraction=float(np.mean(target_distance>cutoff))
    excess_recent_error=recent_disagreement(history,recent)
    top_count=max(1,int(np.ceil(X.shape[1]/4)))
    shift_score=float(np.sort(drift)[-top_count:].mean())
    detected=(outlier_fraction>.15 and shift_score>.10) or excess_recent_error>.15
    details={'detected':bool(detected),'feature_shift_score':shift_score,'target_outlier_fraction':outlier_fraction,
        'source_only_outlier_cutoff':cutoff,'recent_source_excess_error':excess_recent_error,
        'target_labels_used':False,'family_or_strength_used':False}
    if not detected and not force_views:return [('identity',X,y,Q)]*6,details
    views=[('identity',X,y,Q)]
    transform=QuantileTransformer(n_quantiles=min(64,len(X)),output_distribution='normal',random_state=0,subsample=None)
    views.append(('source_gaussian_quantiles',transform.fit_transform(X),y,transform.transform(Q)))
    ranking=np.argsort(drift,kind='stable')
    stable=ranking[:max(1,int(np.ceil(X.shape[1]*.5)))]
    importance=np.zeros(X.shape[1])
    for label in np.unique(y):
        p=float(np.mean(y==label))
        if p<1:importance=np.maximum(importance,np.abs(Z[y==label].mean(0))*np.sqrt(p/max(1-p,1e-12)))
    anchors=stable[np.argsort(importance[stable],kind='stable')[-min(16,len(stable)):]]
    qcenter=Q.mean(0);qscale=Q.std(0);qscale=np.where(qscale>1e-8,qscale,1.)
    T=np.clip((Q-qcenter)/qscale,-20,20)
    source_cov=Z[:,anchors].T@Z/max(len(X),1);target_cov=T[:,anchors].T@T/max(len(Q),1)
    denominator=np.sqrt((source_cov**2).sum(0)*(target_cov**2).sum(0))
    agreement=np.sum(source_cov*target_cov,axis=0)/np.maximum(denominator,1e-8)
    critical=1.36*np.sqrt(1/len(X)+1/len(Q))
    flipped=(agreement<-.25)&(drift>critical)
    grid=np.linspace(.02,.98,33);source_grid=np.quantile(X,grid,axis=0);target_grid=np.quantile(Q,grid,axis=0)
    aligned=np.empty_like(Q)
    for j in range(X.shape[1]):
        values,first=np.unique(target_grid[:,j],return_index=True)
        if len(values)==1:aligned[:,j]=float(np.median(X[:,j]));continue
        quantile=np.interp(Q[:,j],values,grid[first])
        if flipped[j]:quantile=1-quantile
        aligned[:,j]=np.interp(quantile,grid,source_grid[:,j])
    views.append(('unlabelled_quantile_correlation_transport',X,y,aligned))
    for fraction in [.75,.5]:
        columns=np.sort(ranking[:max(1,int(np.ceil(X.shape[1]*fraction)))])
        views.append((f'least_drift_features_{fraction}',X[:,columns],y,Q[:,columns]))
    if recent is not None and recent[0] is not None:
        R,ry=recent;rng=np.random.default_rng(0)
        mandatory=[int(rng.choice(np.flatnonzero(y==label))) for label in np.unique(y)]
        remaining=len(y)-len(mandatory);ids=rng.choice(len(ry),remaining,replace=True)
        A=np.vstack([X[mandatory],R[ids]]);b=np.r_[y[mandatory],ry[ids]]
        views.append(('recent_source_bootstrap',A,b,Q))
    else:
        distances=pairwise_distances(V[:,anchors],Z[:,anchors],metric='sqeuclidean')
        count=min(10,len(X));neighbors=np.argpartition(distances,count-1,axis=1)[:,:count]
        repaired=Q.copy();bad=np.flatnonzero(drift>critical)
        if len(bad):repaired[:,bad]=np.median(X[neighbors][:,:,bad],axis=1)
        views.append(('source_neighbour_feature_repair',X,y,repaired))
    assert len(views)==6
    assert all(np.isfinite(A).all() and np.isfinite(T).all() and len(b)<=len(y) for _,A,b,T in views)
    details.update(flipped_feature_count=int(flipped.sum()),critical_feature_distance=float(critical))
    return views,details
