"""Reconstruct all v56 TRAIN subsets and replay every saved source model."""
import argparse
import json
import time
import joblib
import numpy as np
import bagged_specific_v56 as experiment
from prepare_data import atomic_json,protocol


def audit(budget):
    parent=experiment.parent;h=parent.h;verified=0;source_models=0;max_normalized_error=0.;solver_flags=[]
    h.source.verify_registry()
    for block in ['clean','seven']:
        root=experiment.OUT/'policies'/str(budget)/'regression'/block
        freeze=root/'all_policies_frozen.json';policy=json.loads(freeze.read_text());assert (root/'completion.json').exists()
        banks={}
        for meta in h.source.adaptation.raw.frozen.cohort('regression',block):
            tid=str(meta['task_id']);pool,cal,_=parent.source_data(meta,block,budget)
            path=experiment.OUT/'experts'/str(budget)/block/f'{tid}.npz';info=json.loads(path.with_suffix('.json').read_text())
            assert len(info['records'])==len(experiment.PROFILES)*experiment.MEMBERS
            with np.load(path) as z:
                assert np.array_equal(z['y_validation'],cal['y_validation']);banks[tid]=z['predictions']
                scale=max(1.,float(np.std(cal['y_validation'])))
                for record in info['records']:
                    i,k=record['profile_index'],record['member']
                    source_pos=experiment.subset(pool,cal,budget,k,tid,'validation')
                    query_pos=experiment.subset(pool,cal,budget,k,tid,'query')
                    assert record['source_original_ids']==pool['source_ids'][source_pos].tolist()
                    assert record['query_original_ids']==pool['source_ids'][query_pos].tolist()
                    assert record['source_rows']==len(source_pos)<=budget and record['query_rows']==len(query_pos)<=budget
                    assert not np.intersect1d(pool['source_ids'][source_pos],cal['validation_original_ids']).size
                    assert not np.intersect1d(pool['source_ids'][query_pos],pool['query_ids']).size
                    assert record['source_X_sha256']==protocol.array_sha256(pool['X'][source_pos])
                    assert record['source_y_sha256']==protocol.array_sha256(pool['y'][source_pos])
                    model=joblib.load(path.parent/'models'/tid/f'validation_{i}_{k}.joblib')
                    value=h.head_predict(cal['X_validation'],model);error=float(np.max(np.abs(value-z['predictions'][i,k])))/scale
                    max_normalized_error=max(max_normalized_error,error);assert error<1e-10,(budget,block,tid,i,k,error)
                    estimator=model['estimator'];iterations=getattr(estimator,'n_iter_',0);limit=getattr(estimator,'max_iter',None) or float('inf')
                    if iterations is not None and np.max(np.atleast_1d(iterations))>=limit:solver_flags.append(dict(task_id=meta['task_id'],profile=i,member=k,block=block))
                    source_models+=1
        cells=h.risk.load(h.source.OUT,'validation',budget,block);baseline=experiment.parent_source(cells,budget,block)
        for family in sorted({c['meta']['family'] for c in cells}):
            group=[c for c in cells if c['meta']['family']==family]
            choice=experiment.select(group,baseline,banks)
            assert choice['chosen']==policy['Specific'][family]['chosen']
        queries=h.risk.load(h.source.OUT,'query',budget,block);cache={}
        for c in queries:
            m=c['meta'];original=parent.OUT/'predictions'/str(budget)/c['key']
            with np.load(original) as z:base=z['Base'];sh=z['Shared'];sp=z['Specific'];truth=z['y_target']
            choice=policy['Specific'][m['family']]['chosen']
            if choice is not None:
                data,_=h.source.adaptation.raw.scaling.bank('regression',block,m,m['family'],m['rho']);X=data['X_target']
                assert protocol.array_sha256(X)==m['source_input_hashes']['X_target']
                values=[]
                for member in range(choice['members']):
                    path=experiment.OUT/'experts'/str(budget)/block/'models'/str(m['task_id'])/f'query_{choice["profile_index"]}_{member}.joblib'
                    if str(path) not in cache:cache[str(path)]=joblib.load(path)
                    values.append(h.head_predict(X,cache[str(path)]))
                a=choice['blend'];sp=(1-a)*sp+a*np.mean(values,axis=0)
            out=experiment.OUT/'predictions'/str(budget)/c['key'];record=json.loads(out.with_suffix('.json').read_text())
            assert freeze.stat().st_mtime<=out.stat().st_mtime
            assert record['input_hashes']==m['source_input_hashes']
            with np.load(out) as z:
                assert np.array_equal(z['Base'],base) and np.array_equal(z['Base'],c['base'])
                assert np.array_equal(z['Shared'],sh) and np.array_equal(z['y_target'],truth) and np.array_equal(truth,c['y'])
                error=float(np.max(np.abs(z['Specific']-sp)))/max(1.,float(np.std(truth)))
                max_normalized_error=max(max_normalized_error,error);assert error<1e-10,(str(out),error)
                for name in ['Base','Shared','Specific']:
                    measured=h.risk.metrics(truth,z[name],'regression',m['scale'])
                    assert all(abs(v-record['methods'][name][k])<1e-12 for k,v in measured.items())
            verified+=1
    assert verified==959 and source_models==2720
    atomic_json(experiment.OUT/'raw_audits'/f'{budget}.json',dict(status='PASS',budget=budget,verified_conditions=verified,
        source_models_replayed=source_models,all_selected_query_ensembles_replayed=True,
        max_prediction_error_normalized_by_max1_label_std=max_normalized_error,normalized_tolerance=1e-10,
        source_subset_IDs_and_hashes_reconstructed=True,source_calibration_disjoint=True,query_overlap=0,
        per_expert_unique_TRAIN_row_caps_verified=True,Shared_bitwise_unchanged=True,Base_and_truth_exactly_match_frozen_banks=True,
        source_selection_choices_reproduced=True,all_metrics_recomputed=True,policies_frozen_before_query_predictions=True,
        solver_iteration_limit_flags=solver_flags,solver_convergence_claim=False,time_unix=time.time()))
    print('BAGGED_SPECIFIC_RAW_AUDIT_PASS',budget,verified,source_models,max_normalized_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);a=p.parse_args();audit(a.budget)
