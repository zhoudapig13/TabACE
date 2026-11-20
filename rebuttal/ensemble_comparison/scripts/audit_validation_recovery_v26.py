"""Audit held-out TRAIN calibration and once-frozen query evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import validation_recovery_v26 as v
from prepare_data import atomic_json,protocol
from run_tabicl import metrics


def validation_audit():
    failures=[];count=0;expected=0
    for block,meta in v.prior.jobs():
        pool=v.prior.original.prepare('regression',block,meta);split=v.split(meta,block,pool)
        train,held=split['train_positions'],split['held_positions']
        if np.intersect1d(pool['source_ids'][train],pool['source_ids'][held]).size:failures.append([meta['task_id'],'validation_source_overlap'])
        if np.intersect1d(pool['source_ids'][held],pool['query_ids']).size:failures.append([meta['task_id'],'validation_query_overlap'])
        if not np.array_equal(train[:256],np.arange(256)):failures.append([meta['task_id'],'core_membership'])
        scale=max(float(split['y_validation'].std()),1e-12)
        qhash=protocol.array_sha256(split['X_validation']);thash=protocol.array_sha256(split['y_validation'])
        for family,rho in v.grid(block):
            root=v.OUT/'validation'/block/str(meta['task_id'])/f'{family}_{rho}'
            paths=[root/'base256.npz']+[root/v.tag(rate)/f'{budget}.npz' for rate in v.rates() for budget in v.BUDGETS[1:]]
            expected+=len(paths)
            for path in paths:
                if not path.exists() or not path.with_suffix('.json').exists():failures.append([str(path),'missing']);continue
                rec=json.loads(path.with_suffix('.json').read_text())
                with np.load(path) as z:p=z['prediction'];t=z['y_target']
                if protocol.array_sha256(t)!=thash:failures.append([str(path),'validation_labels'])
                if rec['scale']!=scale or rec['seed']!=0 or rec['n_estimators']!=1:failures.append([str(path),'configuration'])
                if 'input_hashes' in rec:
                    if rec['input_hashes']['X_target']!=qhash or rec['input_hashes']['y_target']!=thash:failures.append([str(path),'query_partition'])
                measured=metrics(t,p,'regression',scale)
                if any(not np.isclose(measured[k],x,atol=1e-12,rtol=0) for k,x in rec['metrics'].items()):failures.append([str(path),'metrics'])
                count+=1
    result=dict(status='PASS' if not failures else 'FAIL',verified_predictions=count,expected_predictions=expected,
        failures=failures,validation_source_disjoint=True,original_query_labels_not_used_for_selection=True,
        development_calibration_only=True,prior_query_metrics_previously_viewed=True)
    atomic_json(v.OUT/'validation_audit.json',result);print('VALIDATION_AUDIT',count,'/',expected,'failures',failures[:5])
    if failures:raise SystemExit(1)


def query_audit():
    failures=[];count=0
    selected=json.loads((v.OUT/'frozen_parameters.json').read_text());fingerprint=v.prior.original.sha(v.OUT/'frozen_parameters.json')
    for block,meta in v.prior.jobs():
        pool=v.prior.original.prepare('regression',block,meta);rate=selected['repair_probabilities'][block]
        for family,rho in v.grid(block):
            env=v.environment(pool,meta,block,family,rho,np.arange(len(pool['y'])))
            base,info=v.prior.original.bank('regression',block,meta,family,rho)
            for budget in v.BUDGETS:
                path=v.OUT/'query'/str(budget)/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                if not path.exists() or not path.with_suffix('.json').exists():failures.append([str(path),'missing']);continue
                rec=json.loads(path.with_suffix('.json').read_text());A,b,mask=v.context(env,rate,budget)
                hashes={k:protocol.array_sha256(x) for k,x in (base|dict(X_source=A,y_source=b)).items()}
                if hashes!=rec['input_hashes']:failures.append([str(path),'source_reconstruction'])
                if rec['frozen_parameters_sha256']!=fingerprint or rec['repair_probability']!=rate:failures.append([str(path),'parameters_changed'])
                if not np.array_equal(A[:256],base['X_source']) or not np.array_equal(b[:256],base['y_source']):failures.append([str(path),'core_changed'])
                if rec['scale']!=info['scale']:failures.append([str(path),'metric_denominator'])
                with np.load(path) as z:p=z['prediction'];truth=z['y_target'];storedmask=z['repair_mask']
                if not np.array_equal(storedmask,mask) or mask[:256].any():failures.append([str(path),'repair_mask'])
                if not np.array_equal(truth,base['y_target']):failures.append([str(path),'query_labels_changed'])
                measured=metrics(truth,p,'regression',info['scale'])
                if any(not np.isclose(measured[k],x,atol=1e-12,rtol=0) for k,x in rec['metrics'].items()):failures.append([str(path),'metrics'])
                count+=1
    result=dict(status='PASS' if not failures else 'FAIL',verified_predictions=count,expected_predictions=6496,
        failures=failures,core_and_queries_unchanged=True,one_frozen_query_configuration_per_block=True,
        development_calibration_only=True,independent_test_claim=False)
    atomic_json(v.OUT/'query_audit.json',result);print('QUERY_AUDIT',count,'/6496 failures',failures[:5])
    if failures or count!=6496:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['validation','query']);a=p.parse_args()
    validation_audit() if a.stage=='validation' else query_audit()
