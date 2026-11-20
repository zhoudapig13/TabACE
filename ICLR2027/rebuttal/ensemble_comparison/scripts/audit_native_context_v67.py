"""Audit all native banks/fusions and freshly replay both named image tasks."""
import json
import time
import hashlib
import numpy as np
import torch
import native_context_specific_v67 as experiment
from prepare_data import atomic_json,protocol
from run_tabicl import Predictor,metrics,CHECKPOINTS,normalize


def audit():
    torch.set_num_threads(1);torch.manual_seed(0);predictor=Predictor('classification','cuda:0')
    checkpoint=CHECKPOINTS/'tabicl-classifier-v2-20260212.ckpt';plan=json.loads((experiment.OUT/'plan.json').read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==plan['checkpoint_sha256']
    bank_cases=0;query_cases=0;fresh=0;maximum=0.;fresh_maximum=0.;deploy_counts={}
    for budget in experiment.BUDGETS:
        source=experiment.parent.load('validation',budget);queries=experiment.parent.load('query',budget)
        for role,cells in [('validation',source),('query',queries)]:
            for c in cells:
                tid=str(c['meta']['task_id']);path=experiment.OUT/'banks'/role/str(budget)/f'{tid}.npz';row=json.loads(path.with_suffix('.json').read_text())
                assert row['n_estimators']==8 and row['seed']==0 and row['actual_context_rows']<=budget
                assert row['input_hashes']==c['meta']['source_input_hashes']
                with np.load(path) as z:
                    p=z['prediction'];assert p.shape==c['base'].shape and np.isfinite(p).all()
                    assert np.allclose(p.sum(1),1.,atol=1e-12)
                    if role=='query':assert 'y_target' not in z.files
                    else:assert np.array_equal(z['y_validation'],c['y'])
                if int(tid) in [167121,167124]:
                    meta=c['meta'];pool=experiment.RAW.scaling.prepare('classification','five',meta)
                    if role=='validation':
                        import tabace_validation_banks_v28 as validation
                        split=validation.validation_split('classification','five',meta,pool)
                        data,_=validation.materialize('classification','clean',meta,'original',0,budget,pool,split)
                    else:data,*_=experiment.RAW.query_input('classification','clean',meta,'original',0,budget,pool)
                    assert {k:protocol.array_sha256(v) for k,v in data.items()}==row['input_hashes']
                    value,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=8,random_state=0,n_classes=c['base'].shape[1])
                    error=float(np.max(np.abs(p-value)));fresh_maximum=max(fresh_maximum,error);assert error<1e-5,(tid,budget,role,error)
                    fresh+=1
                bank_cases+=1
        root=experiment.OUT/'policies'/str(budget)/'classification/clean';freeze=root/'all_policies_frozen.json';policy=json.loads(freeze.read_text());choice=policy['Specific']
        for c in queries:
            old=experiment.parent.OUT/'predictions'/str(budget)/c['key']
            with np.load(old) as z:base=z['Base'];sh=z['Shared'];previous=z['Specific'];truth=z['y_target']
            sp=sh
            if choice is not None:
                with np.load(experiment.OUT/'banks/query'/str(budget)/f'{c["meta"]["task_id"]}.npz') as z:
                    value=experiment.parent.transform(c,z['prediction'],choice['prior_power'],choice['temperature'])
                if choice['source']=='union_mean':value=normalize(.5*(value+previous))
                elif choice['source']=='previous_specific':value=previous
                a=choice['blend'];sp=normalize((1-a)*sh+a*value)
            out=experiment.OUT/'predictions'/str(budget)/c['key'];record=json.loads(out.with_suffix('.json').read_text())
            assert freeze.stat().st_mtime<=out.stat().st_mtime
            with np.load(out) as z:
                assert np.array_equal(z['Base'],base) and np.array_equal(base,c['base']) and np.array_equal(z['Shared'],sh)
                assert np.array_equal(z['y_target'],truth) and np.array_equal(truth,c['y'])
                error=float(np.max(np.abs(z['Specific']-sp)));maximum=max(maximum,error);assert error<1e-12
                for name in ['Base','Shared','Specific']:
                    values=metrics(truth,z[name],'classification',1.)
                    assert all((v is None and record['methods'][name][k] is None) or abs(v-record['methods'][name][k])<1e-12 for k,v in values.items())
            uses=choice is not None and choice['source']!='previous_specific'
            count=11 if choice is None else 19 if choice['source']=='native' else 30 if choice['source']=='union_mean' else 22
            # Clarify candidate-generation compute versus the selected recipe;
            # no probability, metric, or trained parameter is changed.
            record.update(selected_recipe_uses_added_native_context=uses,selected_recipe_added_native_estimators=8 if uses else 0,
                selected_recipe_max_configured_estimators=count,native_candidate_bank_estimators_executed=8)
            atomic_json(out.with_suffix('.json'),record);deploy_counts[str(budget)]=count;query_cases+=1
    assert bank_cases==432 and query_cases==216 and fresh==12
    atomic_json(experiment.OUT/'raw_audit.json',dict(status='PASS',native_banks_verified=bank_cases,query_conditions_replayed=query_cases,
        fresh_Native8_cases=fresh,fresh_tasks=['CIFAR-10','Devnagari-Script'],fresh_max_probability_error=fresh_maximum,
        maximum_fusion_probability_error=maximum,checkpoint_hash_verified=True,source_input_hashes_match_frozen_banks=True,
        Base_and_Shared_bitwise_unchanged=True,all_metrics_recomputed=True,selected_configured_estimator_upper_bounds=deploy_counts,
        additional_inference_budget_disclosed=True,matched_inference_budget_claim=False,policies_frozen_before_fusion_outputs=True,time_unix=time.time()))
    print('NATIVE_CONTEXT_AUDIT_PASS',bank_cases,query_cases,fresh,fresh_maximum,deploy_counts,flush=True)


if __name__=='__main__':audit()
