"""Frozen feature-only reconstruction; never reads query labels to construct shifts."""
import json, fcntl
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from prepare_data import RUN, atomic_json, protocol
from scaled_historical_feature_protocol import apply_feature_scaled_shift


def materialize(meta, family, rho, *, query=None):
    frozen=json.loads((RUN/'configs/feature_protocol_frozen.json').read_text())
    assert frozen['status']=='FROZEN'
    with np.load(RUN/'cache/five/classification'/f"{meta['task_id']}.npz") as f:
        data={k:f[k] for k in f.files}
    if query is not None:
        data['X_target'],data['y_target']=query
    X,y,Q=data['X_source'],data['y_source'],data['X_target']
    if rho:
        teacher=ExtraTreesClassifier(n_estimators=128,random_state=0,n_jobs=1).fit(X,y)
        pseudo=teacher.predict(Q)
        kind='aucband' if family=='band' else family
        name=f"publicshift_d{meta['data_id']}_{kind}_rho{rho}_beta{int(100*frozen['beta'])}"
        changed,details=apply_feature_scaled_shift(name,X,y,Q,pseudo,0,
            feature_multiplier=frozen['feature_multiplier'])
        data['X_target']=changed
    else:details={}
    info=meta|{'track':'five_feature_calibrated','family':family,'rho':rho,
        'n_classes':len(np.unique(y)),'condition_materialized':True,
        'protocol_variant':'historical feature-only reconstruction; not exact paper restoration',
        'feature_protocol_sha256':frozen['protocol_sha256'],
        'source_labels_changed':False,'generator_reads_query_labels':False,
        'generator_teacher':'ExtraTrees128 fitted only to source',
        'severity_parameters':{'beta':frozen['beta'],'feature_multiplier':frozen['feature_multiplier']},
        'operator_details':details,'X_context_sha256':protocol.array_sha256(X),
        'y_context_sha256':protocol.array_sha256(y),
        'X_shifted_target_sha256':protocol.array_sha256(data['X_target']),
        'y_target_sha256':protocol.array_sha256(data['y_target'])}
    return data,info


def load(meta,family,rho):
    path=RUN/'cache/five_feature_calibrated/classification'/str(meta['task_id'])/f'{family}_{rho}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if not path.with_suffix('.json').exists():
        data,info=materialize(meta,family,rho)
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(path)
        atomic_json(path.with_suffix('.json'),info)
    info=json.loads(path.with_suffix('.json').read_text())
    frozen=json.loads((RUN/'configs/feature_protocol_frozen.json').read_text())
    assert info['feature_protocol_sha256']==frozen['protocol_sha256']
    with np.load(path) as f:data={k:f[k] for k in f.files}
    return data,info
