"""Replay every v54 expert and query fusion, without fitting any model."""
import argparse
import json
import time
import joblib
import numpy as np
import augmented_classifier_v54 as experiment
from prepare_data import atomic_json,protocol
from run_tabicl import metrics


def audit(budget):
    raw=experiment.nn.source.adaptation.raw;source_models=0;conditions=0;max_error=0.
    for block in ['clean','seven']:
        root=experiment.OUT/'policies'/str(budget)/'classification'/block
        freeze=root/'all_policies_frozen.json';policy=json.loads(freeze.read_text())
        assert (root/'completion.json').exists()
        source_cells=experiment.nn.load('validation',budget,block);by_task={}
        for c in source_cells:by_task.setdefault(str(c['meta']['task_id']),[]).append(c)
        for meta in raw.frozen.cohort('classification',block):
            tid=str(meta['task_id']);track='five' if block=='clean' else 'seven'
            pool=raw.scaling.prepare('classification',track,meta)
            p=raw.OUT/'validation_splits/classification'/track/f'{tid}.npz'
            with np.load(p) as z:partition={k:z[k] for k in z.files}
            X,y,ids,H=experiment.training_arrays(pool,partition,budget,block,'validation')
            Xf,yf,idsf,Hf=experiment.training_arrays(pool,partition,budget,block,'query')
            assert len(np.unique(ids))==len(ids)<=budget and len(np.unique(idsf))==len(idsf)<=budget
            assert not np.intersect1d(ids,partition['validation_original_ids']).size
            assert not np.intersect1d(idsf,pool['query_ids']).size
            group=by_task[tid];data={}
            for c in group:
                m=c['meta'];arrays,_=experiment.nn.source.adaptation.validation.materialize('classification',block,meta,m['family'],m['rho'],budget,pool,partition)
                assert protocol.array_sha256(arrays['X_target'])==m['source_input_hashes']['X_target']
                assert np.array_equal(arrays['y_target'],c['y']);data[c['key']]=arrays['X_target']
            roles=['Shared']+[f'Specific_{f}' for f in (['original'] if block=='clean' else experiment.FAMILIES)]
            for role in roles:
                folder=experiment.OUT/'experts'/str(budget)/block/tid/role
                bank=folder/'source_predictions.npz';info=json.loads(bank.with_suffix('.json').read_text())
                assert info['source_unique_rows']==len(ids) and info['final_unique_rows']==len(idsf)
                expected_hashes={k:protocol.array_sha256(a) for k,a in [('X',X),('y',y),('H',H if H is not None else X[:0])]}
                assert info['source_hashes']==expected_hashes
                scope=group if role=='Shared' else [c for c in group if c['meta']['family']==role.removeprefix('Specific_')]
                with np.load(bank) as z:
                    assert np.array_equal(z['training_original_ids'],ids)
                    assert np.array_equal(z['validation_original_ids'],partition['validation_original_ids'])
                    for index in ([0] if role=='Shared' else range(len(experiment.PROFILES))):
                        model=joblib.load(folder/f'validation_{index}.joblib')
                        for c in scope:
                            value=experiment.predict(data[c['key']],model,c['base'].shape[1])
                            saved=z[f'{index}_{c["meta"]["family"]}_{c["meta"]["rho"]}']
                            error=float(np.max(np.abs(value-saved)));max_error=max(max_error,error)
                            assert error<1e-10,(budget,block,tid,role,index,error)
                        source_models+=1
        cache={};queries=experiment.nn.load('query',budget,block)
        for c in queries:
            m=c['meta'];data,_=raw.scaling.bank('classification',block,m,m['family'],m['rho']);X=data['X_target']
            assert protocol.array_sha256(X)==m['source_input_hashes']['X_target']
            def predict(p,role,anchor):
                choice=p['chosen']
                if choice is None:return anchor
                path=experiment.OUT/'experts'/str(budget)/block/str(m['task_id'])/role/f'query_{choice["profile_index"]}.joblib'
                if str(path) not in cache:cache[str(path)]=joblib.load(path)
                value=experiment.predict(X,cache[str(path)],c['base'].shape[1]);a=choice['blend']
                return experiment.normalize((1-a)*anchor+a*value)
            sh=predict(policy['Shared'],'Shared',c['base']);sp=predict(policy['Specific'][m['family']],f'Specific_{m["family"]}',sh)
            out=experiment.OUT/'predictions'/str(budget)/c['key'];record=json.loads(out.with_suffix('.json').read_text())
            assert freeze.stat().st_mtime<=out.stat().st_mtime
            assert record['input_hashes']==m['source_input_hashes']
            with np.load(out) as z:
                assert np.array_equal(z['Base'],c['base']) and np.array_equal(z['y_target'],c['y'])
                for name,value in [('Shared',sh),('Specific',sp)]:
                    error=float(np.max(np.abs(z[name]-value)));max_error=max(max_error,error)
                    assert error<1e-10,(str(out),name,error)
                for name in ['Base','Shared','Specific']:
                    measured=metrics(z['y_target'],z[name],'classification',1.)
                    for key,value in measured.items():
                        expected=record['methods'][name][key]
                        assert (value is None and expected is None) or abs(value-expected)<1e-12
            conditions+=1
    assert conditions==1052 and source_models==1058
    atomic_json(experiment.OUT/'raw_audits'/f'{budget}.json',dict(status='PASS',budget=budget,verified_conditions=conditions,
        source_models_replayed=source_models,all_selected_query_models_replayed=True,max_probability_error=max_error,
        source_training_and_calibration_disjoint=True,unique_original_TRAIN_row_caps_verified=True,
        Base_and_query_truth_exact_match=True,all_metrics_recomputed=True,policies_frozen_before_query_predictions=True,time_unix=time.time()))
    print('AUGMENTED_CLASSIFIER_RAW_AUDIT_PASS',budget,conditions,source_models,max_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);a=p.parse_args();audit(a.budget)
