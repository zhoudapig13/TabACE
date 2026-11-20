"""Reconstruct inputs and recompute metrics for a declared development run."""
import argparse
import importlib
import json
import numpy as np
import pandas as pd
import label_extension_recovery_v38 as base
from prepare_data import atomic_json,protocol
from run_tabicl import metrics

MODULES={39:'nested_response_recovery_v39',40:'coupled_shift_recovery_v40',
         41:'offset_response_recovery_v41',42:'half_coupled_recovery_v42'}


def audit(version):
    mod=importlib.import_module(MODULES[version]);root=mod.OUT
    context=mod.experiment.context if version==42 else mod.context
    rows=[];verified=0;blocks=[]
    for block in ['five','seven']:
        frozen=root/f'parameters_{block}.json'
        if not frozen.exists():continue
        policy=json.loads(frozen.read_text());blocks.append(block)
        for track,meta in base.prior.v.prior.jobs():
            if track!=block:continue
            pool=base.prior.v.prior.original.prepare('regression',block,meta)
            assert np.intersect1d(pool['source_ids'],pool['query_ids']).size==0
            assert len(np.unique(pool['source_ids']))==len(pool['source_ids'])
            for family,rho in base.prior.v.grid(block):
                env=base.prior.v.environment(pool,meta,block,family,rho,np.arange(len(pool['y'])))
                original,info=base.prior.v.prior.original.bank('regression',block,meta,family,rho)
                previous=None
                for budget in [256,1024,4096,8192]:
                    path=root/'query'/str(budget)/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                    record=json.loads(path.with_suffix('.json').read_text())
                    X,y=context(env,block,family,policy['policies'][family],budget)
                    assert len(y)==min(budget,len(pool['y']))
                    assert np.array_equal(X[:256],original['X_source']) and np.array_equal(y[:256],original['y_source'])
                    if previous is not None:
                        A,b=previous;assert np.array_equal(X[:len(b)],A) and np.array_equal(y[:len(b)],b)
                    previous=X,y
                    measured_hashes=dict(X_source=protocol.array_sha256(X),y_source=protocol.array_sha256(y),
                        X_target=protocol.array_sha256(original['X_target']),y_target=protocol.array_sha256(original['y_target']))
                    assert record['input_hashes']==measured_hashes
                    assert record['parameter_sha256']==base.prior.v.prior.original.sha(frozen)
                    assert record['scale']==info['scale']
                    with np.load(path) as z:
                        p=z['prediction'];assert np.isfinite(p).all() and np.array_equal(z['y_target'],original['y_target'])
                        measured=metrics(z['y_target'],p,'regression',record['scale'])
                        assert all(abs(measured[k]-record['metrics'][k])<1e-12 for k in measured)
                    rows.append(dict(block=block,context=budget,task_id=meta['task_id'],actual_context=len(y),**measured))
                    verified+=1
    frame=pd.DataFrame(rows);avg=frame.groupby(['block','context'],as_index=False).agg(datasets=('task_id','nunique'),conditions=('task_id','size'),
        r2=('r2','mean'),nrmse=('nrmse','mean'),actual_context_mean=('actual_context','mean'))
    avg.to_csv(root/'audited_averages.csv',index=False)
    expected=sum((35*20 if b=='five' else 33*28)*4 for b in blocks)
    assert verified==expected
    checks=[]
    for block,g in avg.groupby('block'):
        g=g.sort_values('context');r,n=g.r2.values,g.nrmse.values
        checks.append(dict(block=block,R2_monotone=bool(np.all(np.diff(r)>0)),nRMSE_monotone=bool(np.all(np.diff(n)<=0)),
            final_R2_gain=float(r[-1]-r[0]),within_predeclared_008_query_gain=bool(.005<=r[-1]-r[0]<=.08)))
    atomic_json(root/'query_reconstruction_audit.json',dict(status='PASS',verified_conditions=verified,expected_conditions=expected,
        fixed_core256_unchanged=True,nested_context_prefixes=True,source_query_disjoint=True,parameters_immutable=True,
        all_metrics_recomputed=True,checks=checks,development_only=True,independent_test_claim=False,
        original_paper_replication=False,extra_clean_TRAIN_supervision=True,
        query_feedback_used_for_protocol_design=bool(json.loads((root/'plan.json').read_text()).get('query_feedback_used_for_protocol_design',False))))
    print('DEVELOPMENT_RECONSTRUCTION_AUDIT',version,verified,checks,flush=True)
    print(avg.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,choices=MODULES,required=True);a=p.parse_args();audit(a.version)
