"""Exact source-only memoization; no TFM representation-cache approximation.

Never cache query-dependent statistics or selected columns. Repeated source
quantiles, source atom counts and identical source augmentations are reusable.
The verified installer is enabled only by a matching array-equivalence audit.
"""
from collections import OrderedDict
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from prepare_data import RUN,protocol

AUDIT=RUN/'results/tabace_context_adaptation_v29/source_constructor_memo_v35_audit.json'


def file_sha():return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_sha(module):
    text='\n'.join(inspect.getsource(getattr(module,name)) for name in ['robust','stable_columns','augmented_source'])
    return hashlib.sha256(text.encode()).hexdigest()


def matrix_key(X):return X.shape,X.dtype.str,protocol.array_sha256(X)


def get(cache,key,build,limit):
    if key not in cache:
        cache[key]=build()
        if len(cache)>limit:cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]


def install(module):
    if getattr(module,'_source_memo_v35',False):return
    augment=module.augmented_source
    statistics=OrderedDict();stability=OrderedDict();augmentations=OrderedDict()
    def stats(X):
        def build():
            center=np.median(X,axis=0)
            scale=np.percentile(X,75,axis=0)-np.percentile(X,25,axis=0)
            scale=np.where(scale>1e-6,scale,np.maximum(X.std(0),1e-6))
            return center,scale
        return get(statistics,matrix_key(X),build,16)
    def robust(X,Q):
        center,scale=stats(X)
        return (X-center)/scale,(Q-center)/scale
    def columns(X,Q):
        def build():
            Z,_=robust(X,X[:1]);penalty=np.zeros(X.shape[1])
            if len(X)>=32:
                for j in range(X.shape[1]):
                    _,count=np.unique(X[:,j],return_counts=True)
                    if len(count)>2 and count.max()/len(X)>.20:penalty[j]=2*count.max()/len(X)
            return Z.std(0),penalty
        std,penalty=get(stability,matrix_key(X),build,16)
        center,scale=stats(X);V=(Q-center)/scale
        ratio=np.maximum(V.std(0),1e-6)/np.maximum(std,1e-6)
        score=np.abs(np.median(V,axis=0))+np.abs(np.log(np.clip(ratio,1e-6,1e6)))
        score+=penalty
        keep=np.flatnonzero(score<=1.5)
        if len(keep)<max(1,X.shape[1]//4):keep=np.argsort(score,kind='stable')[:max(1,X.shape[1]//2)]
        return np.sort(keep)
    def augmented(meta,family,strength,X,y):
        cfg=module.raw.frozen.configuration()['classification_five']['family_settings'][family]
        key=(int(meta['task_id']),int(meta['data_id']),family,int(strength),matrix_key(X),matrix_key(y),json.dumps(cfg,sort_keys=True))
        template=get(augmentations,key,lambda:augment(meta,family,strength,X,y),4)
        # The cached template is not exposed to downstream model code.
        return template.copy()
    module.robust=robust;module.stable_columns=columns;module.augmented_source=augmented
    module._source_memo_v35=True


def install_if_verified(module):
    if not AUDIT.exists():return False
    record=json.loads(AUDIT.read_text())
    if record.get('status')!='PASS' or record.get('memo_code_sha256')!=file_sha() or record.get('original_functions_sha256')!=source_sha(module):return False
    install(module);return True
