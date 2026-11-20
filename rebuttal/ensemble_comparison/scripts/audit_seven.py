"""Check the rebuilt benchmark's invariants before any GPU evaluation."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from prepare_data import RUN,atomic_json
from seven_shifts import FAMILIES,PARAMETERS,registry,classification_clean,regression_clean,materialize


def main():
    manifest=registry();checked=0
    digest=hashlib.sha256(json.dumps(PARAMETERS,sort_keys=True).encode()).hexdigest()
    for meta in manifest:
        root=RUN/'cache/seven_reimplemented'/meta['task']/str(meta['task_id'])
        clean=None
        for family in FAMILIES:
            for rho in [0,20,50,80,100]:
                p=root/f'{family}_{rho}.npz'
                info=json.loads(p.with_suffix('.json').read_text())
                assert info['parameter_sha256']==digest,(p,'stale parameter revision')
                with np.load(p) as f:
                    a={k:f[k] for k in f.files}
                assert a['X_source'].shape[0]==len(a['y_source'])==256
                assert a['X_target'].shape[0]==len(a['y_target'])
                assert len(a['y_target'])==128 if meta['task']=='classification' else len(a['y_target'])<=256
                assert all(np.all(np.isfinite(v)) for v in a.values())
                if rho==0:
                    if clean is None:clean=a
                    for key in a:assert np.array_equal(a[key],clean[key]),(p,key,'nonidentity at zero')
                if family in ['covariate','selection'] or meta['task']=='regression':
                    assert np.array_equal(a['X_target'],clean['X_target']),(p,'target features changed')
                    assert np.array_equal(a['y_target'],clean['y_target']),(p,'target labels changed')
                if family=='domain':assert np.array_equal(a['y_target'],clean['y_target'])
                if family=='concept':
                    assert np.array_equal(a['X_source'],clean['X_source'])
                    assert np.array_equal(a['X_target'],clean['X_target'])
                checked+=1
    # Independently materialize one dataset per task twice, checking determinism.
    determinism=[]
    for task in ['classification','regression']:
        meta=next(x for x in manifest if x['task']==task)
        original=classification_clean(meta) if task=='classification' else regression_clean(meta)
        for family in FAMILIES:
            a,_=materialize(meta,family,80,original);b,_=materialize(meta,family,80,original)
            assert all(np.array_equal(a[k],b[k]) for k in a)
        determinism.append(task)
    result={'status':'PASS','conditions_checked':checked,'datasets':len(manifest),
            'parameter_sha256':digest,'determinism_checked':determinism,
            'zero_strength_identity':True,'fixed_target_where_required':True,
            'finite_arrays':True,'original_bit_exact_restoration':False,
            'metric_scale':'seven-shift regression uses target std only after prediction',
            'limitations':'Original missing constants/sampling cannot be verified; source row disjointness is asserted at construction.'}
    atomic_json(RUN/'results/seven_protocol_audit.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
