"""Source corruption sweep; severity parameters are chosen on separate queries.

The five family selectors and rho affected-row counts remain fixed. Label
corruption multipliers change the fraction of selected rows that are relabeled,
while feature counts and amplitude are exposed as separate declared factors.
"""
import numpy as np
from prepare_data import protocol


def corrupt(X_source,y_source,*,family,rho,label_multiplier=1.,feature_fraction=0.,feature_count=1,amplitude=6.):
    X=np.asarray(X_source,dtype=np.float64);y=np.asarray(y_source)
    classes,encoded=protocol._encoded_target(y);top=protocol._top_feature(X,encoded)
    n=0 if family=='original' or rho<=0 else min(len(y),max(1,int(np.ceil(rho*len(y)))))
    rows=protocol._select_rows(X,encoded,family=family,count=n,top=top,seed=1729+int(round(100*rho))) if n else np.empty(0,dtype=int)
    Z=X-X.mean(0);target=encoded-encoded.mean()
    scores=np.abs(np.sum(Z*target[:,None],0))/np.maximum(np.sqrt(np.sum(Z**2,0)*np.sum(target**2)),1e-12)
    width=min(X.shape[1],max(1,int(feature_count),int(np.ceil(feature_fraction*X.shape[1]))))
    ordered=np.argsort(-scores,kind='stable')
    features=np.r_[top,ordered[ordered!=top]][:width]
    means=np.stack([X[y==c][:,features].mean(0) for c in classes])
    distances=np.sum((means[:,None]-means[None])**2,axis=2);np.fill_diagonal(distances,-np.inf)
    opposite=distances.argmax(1) if len(classes)>1 else np.zeros(1,dtype=int)
    direction=np.sign(means[opposite]-means)[encoded.astype(int)]
    modified=X.copy();labels=y.copy()
    def apply(kind,ids):
        if not len(ids):return
        center=modified.mean(0);scale=np.where(modified.std(0)>1e-12,modified.std(0),1.)
        if kind!='shortcut':
            base_fraction=1/3 if kind=='band' else .5
            count=min(len(ids),int(np.ceil(base_fraction*label_multiplier*len(ids))))
            # Fixed evenly spaced positions; no target values enter this edit.
            stride=3 if kind=='band' else 2
            first=np.arange(0,len(ids),stride)
            rest=np.setdiff1d(np.arange(len(ids)),first,assume_unique=True)
            selected=ids[np.r_[first,rest][:count]]
            current_classes,current_encoded=protocol._encoded_target(labels)
            labels[selected]=current_classes[(current_encoded[selected].astype(int)+1)%len(current_classes)]
        index=np.ix_(ids,features);delta=direction[ids]*scale[features][None]*amplitude
        if kind=='shortcut':modified[index]=center[features][None]+delta
        elif kind=='local':modified[index]=center[features][None]+.7*delta
        else:modified[index]+=(.35 if kind=='boundary' else .22)*delta
    if family=='mixed':
        for kind,ids in zip(['shortcut','boundary','band','local'],np.array_split(rows,4)):apply(kind,ids)
    elif n:apply(family,rows)
    return modified.astype(np.float32),labels,{'affected_rows':len(rows),'features':features.tolist(),
        'label_multiplier':label_multiplier,'amplitude':amplitude,'feature_fraction':feature_fraction,
        'labels_changed':int(np.sum(labels!=y)),'rho':rho,'family':family}


def candidates():
    first=[{'name':f'labels_{m:g}_features1_amplitude6','label_multiplier':m,'feature_fraction':0.,'feature_count':1,'amplitude':6.}
           for m in [1.,1.25,1.5,1.75,2.,2.5,3.]]
    second=[{'name':f'labels_3_features1_amplitude{a:g}','label_multiplier':3.,'feature_fraction':0.,'feature_count':1,'amplitude':a}
            for a in [0.,1.,2.,3.,12.,24.]]
    third=[{'name':f'labels_3_features1_amplitude{a:g}','label_multiplier':3.,'feature_fraction':0.,'feature_count':1,'amplitude':a}
           for a in [.05,.1,.2,.4]]
    fourth=[{'name':f'labels_{m:g}_features1_amplitude0','label_multiplier':m,'feature_fraction':0.,'feature_count':1,'amplitude':0.}
            for m in [2.5,2.75]]
    return first+second+third+fourth
