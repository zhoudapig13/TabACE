"""Declared reimplementation of Appendix B's seven OpenML shift operators.

The archived materializer is absent. This is NOT a bit-for-bit restoration.
All constants below are fixed before evaluation, and both raw conditions and
hashes are saved. Benchmark target labels are used only to define label/concept
interventions, never as inputs to any evaluated inference method.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from prepare_data import ROOT,RUN,LOCAL_CC18,protocol,atomic_json

FAMILIES=('covariate','label','concept','selection','domain','temporal','support')
PARAMETERS={'protocol_revision':2,'source_sampling_seed':0,'regression_sampling_seed':2718,
    'x_weight_exponent':3.,'label_extreme_prior':.9,'max_boundary_flip_fraction':.3,
    'affine_scale_increment':.75,'affine_translation_sd':.5,
    'support_translation_sd':1.5,'regression_response_drift_sd':2.,
    'classification_history':256,'classification_reference':64,'classification_target':128,
    'regression_pool_cap':1024,'regression_reference':128,'regression_context':256,'regression_target_cap':256}


def normalize_name(x):return str(x).lower().replace('_','').replace('-','').replace(' ','')


def registry():
    available={}
    for task in ['classification','regression']:
        for p in (RUN/'cache/five'/task).glob('*.json'):
            m=json.loads(p.read_text());available[(task,str(m['data_id']))]=m
            available[(task,normalize_name(m['dataset']))]=m
    output=[]
    for task in ['classification','regression']:
        path=ROOT/'PaperResults/failure_recovery_20260913/tabicl/seven_point_lock/composite'/task/'detail_oof.csv'
        with path.open() as f:rows=list(csv.DictReader(f))
        folds={r['dataset']:int(r['outer_fold']) for r in rows}
        for identifier,fold in sorted(folds.items()):
            m=available.get((task,identifier)) or available.get((task,normalize_name(identifier)))
            if m is None:raise KeyError((task,identifier))
            output.append(m|{'outer_fold':fold,'legacy_identifier':identifier,'track':'seven_reimplemented'})
    assert sum(x['task']=='classification' for x in output)==35
    assert sum(x['task']=='regression' for x in output)==33
    return output


def standardize(X,reference):
    return (X-reference.mean(0))/np.maximum(reference.std(0),1e-6)


def biased_rows(score,n,strength,seed):
    if strength==0:return np.arange(n)
    rng=np.random.default_rng(seed)
    logits=np.clip(PARAMETERS['x_weight_exponent']*strength*np.asarray(score),-15,15)
    p=np.exp(logits-logits.max());p/=p.sum()
    return rng.choice(n,size=n,replace=True,p=p)


def affine(X,reference,s):
    if s==0:return X.copy()
    mean=reference.mean(0);sd=np.maximum(reference.std(0),1e-6)
    return mean+(X-mean)*(1+PARAMETERS['affine_scale_increment']*s)+PARAMETERS['affine_translation_sd']*s*sd


def flip_boundary(X,y,reference,s):
    out=y.copy()
    if s==0:return out
    j=int(np.argmax(np.var(reference,axis=0)))
    distance=np.abs(X[:,j]-np.median(reference[:,j]))
    count=int(np.ceil(PARAMETERS['max_boundary_flip_fraction']*s*len(y)))
    ids=np.argsort(distance,kind='stable')[:count]
    out[ids]=1-out[ids]
    return out


def classification_clean(meta):
    folder=LOCAL_CC18/str(meta['task_id'])
    Xframe=pd.read_parquet(folder/'X.parquet');labels=pd.read_parquet(folder/'y.parquet').iloc[:,0]
    valid=~labels.isna().to_numpy();X,_=protocol._numeric_matrix(Xframe.loc[valid])
    y=LabelEncoder().fit_transform(labels.loc[valid])
    ids=protocol._balanced_cap_indices(y,448,0)
    source,target=train_test_split(ids,train_size=320,test_size=128,stratify=y[ids],random_state=0)
    history,reference=train_test_split(source,train_size=256,test_size=64,stratify=y[source],random_state=0)
    assert not (set(source)&set(target))
    return X[history],y[history],X[reference],y[reference],X[target],y[target]


def regression_clean(meta):
    with np.load(RUN/'cache/five/regression'/f"{meta['task_id']}.npz") as f:
        X,y,Q,t=f['X_source'],f['y_source'],f['X_target'],f['y_target']
    rng=np.random.default_rng(2718)
    ids=rng.permutation(len(y))[:min(len(y),1024)]
    if len(ids)<384:raise ValueError('insufficient source/reference rows')
    context,reference,history=ids[:256],ids[256:384],ids[384:]
    qids=rng.choice(len(t),min(256,len(t)),replace=False)
    return X[history],y[history],X[reference],y[reference],X[context],y[context],Q[qids],t[qids]


def materialize(meta,family,rho,clean):
    s=rho/100.;task=meta['task'];uses_labels=False
    if task=='classification':
        H,hy,R,ry,Q,t=[x.copy() for x in clean]
        all_source=np.vstack([H,R]);j=int(np.argmax(np.var(all_source,axis=0)))
        if family=='covariate':
            for X,y in [(H,hy),(R,ry)]:
                ids=biased_rows(standardize(X,all_source)[:,j],len(y),s,0)
                X[:],y[:]=X[ids].copy(),y[ids].copy()
        elif family=='label' and s:
            uses_labels=True;rng=np.random.default_rng(0)
            original=float(np.mean(t));extreme=.9 if original>=.5 else .1
            p=(1-s)*original+s*extreme
            z=(rng.random(len(t))<p).astype(int)
            ids=np.array([rng.choice(np.flatnonzero(t==label)) for label in z])
            Q,t=Q[ids],t[ids]
        elif family=='concept':
            uses_labels=s>0;ry=flip_boundary(R,ry,H,s);t=flip_boundary(Q,t,H,s)
        elif family=='selection' and s:
            learner=LogisticRegression(max_iter=500,random_state=0).fit(standardize(all_source,all_source),np.r_[hy,ry])
            majority=int(np.mean(np.r_[hy,ry])>=.5)
            for X,y in [(H,hy),(R,ry)]:
                score=learner.predict_proba(standardize(X,all_source))[:,majority]
                ids=biased_rows(score,len(y),s,0);X[:],y[:]=X[ids].copy(),y[ids].copy()
        elif family=='domain':Q=affine(Q,H,s)
        elif family=='temporal':
            uses_labels=s>0;ry=flip_boundary(R,ry,H,s);t=flip_boundary(Q,t,H,s)
            R=affine(R,H,s);Q=affine(Q,H,s)
        elif family=='support':
            sd=np.maximum(H.std(0),1e-6);ids=np.argsort(H.var(0))[-min(3,H.shape[1]):]
            Q[:,ids]+=PARAMETERS['support_translation_sd']*s*sd[ids]
        X=np.vstack([H[:192],R]);y=np.r_[hy[:192],ry]
        pool_X,pool_y=np.vstack([H,R]),np.r_[hy,ry]
        original_scale=1.
    else:
        H,hy,R,ry,X,y,Q,t=[x.copy() for x in clean]
        X0,y0=X.copy(),y.copy();reference=np.vstack([H,R,X])
        j=int(np.argmax(np.var(X,axis=0)));z=standardize(X,reference)
        if family=='covariate' and s:
            # A fixed ordering gives nested affected rows across strengths.
            order=np.random.default_rng(2718).permutation(len(y))
            affected=order[:int(np.ceil(s*len(y)))];ids=biased_rows(z[:,j],len(y),1.,2718)
            X[affected],y[affected]=X[ids[affected]],y[ids[affected]]
        elif family=='label' and s:
            score=(y-y.mean())/max(float(y.std()),1e-6)
            ids=biased_rows(score,len(y),s,2718);X,y=X[ids],y[ids]
        elif family=='concept':y=y+PARAMETERS['regression_response_drift_sd']*s*y0.std()*np.tanh(z[:,j])
        elif family=='selection' and s:
            distance=np.mean(((X-Q.mean(0))/np.maximum(reference.std(0),1e-6))**2,axis=1)
            score=distance/max(float(distance.std()),1e-6)
            ids=biased_rows(score,len(y),s,2718);X,y=X[ids],y[ids]
        elif family=='domain':X=affine(X,reference,s)
        elif family=='temporal':
            X=affine(X,reference,s)
            y=y+PARAMETERS['regression_response_drift_sd']*s*y0.std()*np.tanh(z[:,j])
        elif family=='support' and s:
            distance=np.mean(((X-Q.mean(0))/np.maximum(reference.std(0),1e-6))**2,axis=1)
            score=-distance/max(float(distance.std()),1e-6)
            ids=biased_rows(score,len(y),s,2718);X,y=X[ids],y[ids]
        pool_X,pool_y=np.vstack([X,R,H]),np.r_[y,ry,hy]
        original_scale=max(float(y0.std()),1e-12)
    info=meta|{'family':family,'rho':rho,'scale':original_scale,
        'evaluation_nrmse_scale':'target_std_ddof0_evaluation_only' if task=='regression' else None,
        'protocol_status':'appendix_based_reimplementation_not_bit_exact',
        'generator_uses_target_labels':uses_labels,'inference_receives_target_labels':False,
        'source_rows':len(y),'target_rows':len(t),'context_budget':len(y),'source_pool_rows':len(pool_y),
        'parameter_sha256':hashlib.sha256(json.dumps(PARAMETERS,sort_keys=True).encode()).hexdigest()}
    arrays=dict(X_source=np.asarray(X,dtype=np.float32),y_source=y,X_target=np.asarray(Q,dtype=np.float32),
                y_target=t,X_pool=np.asarray(pool_X,dtype=np.float32),y_pool=pool_y,
                X_history=np.asarray(H,dtype=np.float32),y_history=hy,
                X_reference=np.asarray(R,dtype=np.float32),y_reference=ry)
    for name,value in arrays.items():
        if not np.all(np.isfinite(value)):raise ValueError('nonfinite '+name)
        info[name+'_sha256']=protocol.array_sha256(value)
    return arrays,info


def prepare():
    items=registry()
    atomic_json(RUN/'configs/seven_reimplementation.json',{'parameters':PARAMETERS,'datasets':items,
        'original_materializer_found':False,'bit_exact_original_protocol':False,
        'user_authorized_reimplementation':True})
    for meta in items:
        clean=classification_clean(meta) if meta['task']=='classification' else regression_clean(meta)
        for family in FAMILIES:
            for rho in [0,20,50,80,100]:
                path=RUN/'cache/seven_reimplemented'/meta['task']/str(meta['task_id'])/f'{family}_{rho}.npz'
                if path.exists() and path.with_suffix('.json').exists():
                    previous=json.loads(path.with_suffix('.json').read_text())
                    if previous.get('parameter_sha256')==hashlib.sha256(json.dumps(PARAMETERS,sort_keys=True).encode()).hexdigest():continue
                arrays,info=materialize(meta,family,rho,clean)
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
                atomic_json(path.with_suffix('.json'),info)
        print('SEVEN_READY',meta['task'],meta['task_id'],meta['dataset'],flush=True)


if __name__=='__main__':prepare()
