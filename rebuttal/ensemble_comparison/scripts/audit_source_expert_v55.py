"""Replay every new v53 model and verify source caps, identities, and metrics."""
import argparse
import json
import time
import joblib
import numpy as np
import source_expert_v53 as experiment
from prepare_data import atomic_json,protocol


def audit(budget):
    h=experiment.h;verified=0;source_models=0;query_models=0;max_error=0.;max_normalized_error=0.;warnings=[]
    h.source.verify_registry()
    for block in ['clean','seven']:
        root=experiment.OUT/'policies'/str(budget)/'regression'/block
        path=root/'all_policies_frozen.json';policy=json.loads(path.read_text());assert (root/'completion.json').exists()
        source_cache={};model_cache={}
        for meta in h.source.adaptation.raw.frozen.cohort('regression',block):
            tid=str(meta['task_id']);pool,partition,pos=experiment.source_data(meta,block,budget)
            base=experiment.OUT/'experts'/str(budget)/block/f'{tid}.npz';info=json.loads(base.with_suffix('.json').read_text())
            assert info['source_X_sha256']==protocol.array_sha256(pool['X'][pos])
            assert info['source_y_sha256']==protocol.array_sha256(pool['y'][pos])
            assert info['validation_X_sha256']==protocol.array_sha256(partition['X_validation'])
            assert info['validation_y_sha256']==protocol.array_sha256(partition['y_validation'])
            assert info['validation_training_rows']==len(pos)<=budget
            assert info['query_refit_rows']==min(budget,len(pool['y']))
            with np.load(base) as z:
                assert np.array_equal(z['training_original_ids'],pool['source_ids'][pos])
                assert np.array_equal(z['validation_original_ids'],partition['validation_original_ids'])
                assert not np.intersect1d(z['training_original_ids'],z['validation_original_ids']).size
                assert np.array_equal(z['y_validation'],partition['y_validation'])
                for index in range(len(experiment.PROFILES)):
                    mpath=base.parent/'models'/tid/f'validation_{index}.joblib';m=joblib.load(mpath)
                    p=h.head_predict(partition['X_validation'],m)
                    error=float(np.max(np.abs(p-z['predictions'][index])));max_error=max(max_error,error)
                    normalized=error/max(1.,float(np.std(partition['y_validation'])))
                    max_normalized_error=max(max_normalized_error,normalized)
                    assert normalized<1e-10,(budget,block,tid,index,error,normalized)
                    estimator=m['estimator']
                    iterations=getattr(estimator,'n_iter_',0)
                    limit=getattr(estimator,'max_iter',None) or float('inf')
                    if iterations is not None and np.max(np.atleast_1d(iterations))>=limit:
                        warnings.append(dict(block=block,task_id=meta['task_id'],profile_index=index,issue='source_solver_reached_iteration_limit'))
                    source_models+=1
            source_cache[tid]=info
        cells=h.risk.load(h.source.OUT,'query',budget,block)
        for c in cells:
            meta=c['meta'];tid=str(meta['task_id']);raw,_=h.source.adaptation.raw.scaling.bank('regression',block,meta,meta['family'],meta['rho']);X=raw['X_target']
            assert protocol.array_sha256(X)==meta['source_input_hashes']['X_target']
            def predict(p,anchor):
                chosen=p['chosen']
                if chosen is None:return anchor
                index=chosen['profile_index'];key=(tid,index)
                if key not in model_cache:model_cache[key]=joblib.load(source_cache[tid]['query_model_paths'][index])
                value=h.head_predict(X,model_cache[key]);a=chosen['blend'];return (1-a)*anchor+a*value
            sh=predict(policy['Shared'],c['base']);sp=predict(policy['Specific'][meta['family']],sh)
            out=experiment.OUT/'predictions'/str(budget)/c['key'];record=json.loads(out.with_suffix('.json').read_text())
            assert path.stat().st_mtime<=out.stat().st_mtime
            assert record['input_hashes']==meta['source_input_hashes']
            with np.load(out) as z:
                assert np.array_equal(z['Base'],c['base']) and np.array_equal(z['y_target'],c['y'])
                for name,value in [('Shared',sh),('Specific',sp)]:
                    error=float(np.max(np.abs(z[name]-value)));max_error=max(max_error,error)
                    normalized=error/max(1.,float(np.std(c['y'])))
                    max_normalized_error=max(max_normalized_error,normalized)
                    assert normalized<1e-10,(str(out),name,error,normalized)
                for name in ['Base','Shared','Specific']:
                    values=h.risk.metrics(z['y_target'],z[name],'regression',meta['scale'])
                    assert all(abs(v-record['methods'][name][k])<1e-12 for k,v in values.items())
            verified+=1
        query_models+=len(model_cache)
    assert verified==959
    atomic_json(experiment.OUT/'raw_audits'/f'{budget}.json',dict(status='PASS',budget=budget,verified_cases=verified,
        source_models_replayed=source_models,selected_query_models_replayed=query_models,max_absolute_prediction_error=max_error,
        max_prediction_error_normalized_by_max1_label_std=max_normalized_error,normalized_tolerance=1e-10,
        all_new_metrics_recomputed=True,source_validation_disjoint=True,unique_TRAIN_row_caps_verified=True,
        Base_and_truth_exactly_match_frozen_banks=True,policies_frozen_before_predictions=True,
        solver_iteration_limit_warnings=warnings,solver_convergence_claim=False,time_unix=time.time()))
    print('SOURCE_EXPERT_AUDIT_PASS',budget,verified,source_models,len(warnings),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);a=p.parse_args();audit(a.budget)
