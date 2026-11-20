"""Replay the expanded source calibration and all35 new Clean8192 outputs."""
import json
import time
import joblib
import numpy as np
import torch
import clean_calibration_v59 as experiment
from prepare_data import atomic_json,protocol
from run_tabicl import Predictor,metrics


def audit():
    torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.set_per_process_memory_fraction(.2)
    predictor=Predictor('regression','cuda:0');parent=experiment.bag.parent;bag=experiment.bag
    root=experiment.OUT/'policies/8192/regression/clean';freeze=root/'all_policies_frozen.json';policy=json.loads(freeze.read_text())
    assert (root/'completion.json').exists();conditions=0;source_models=0;native_cases=0;max_native_error=0.;max_head_error=0.;solver_flags=[]
    for meta in parent.h.source.adaptation.raw.frozen.cohort('regression','clean'):
        tid=str(meta['task_id']);pool=parent.h.source.adaptation.raw.scaling.prepare('regression','five',meta);cal=experiment.partition(meta)
        assert not np.intersect1d(cal['validation_original_ids'],cal['source_original_ids']).size
        assert not np.intersect1d(cal['validation_original_ids'],pool['query_ids']).size
        pos=cal['train_positions'][:8192];assert len(pos)<=8192 and np.array_equal(pool['X'][pos][:256],pool['X'][:256])
        scale=max(1.,float(np.std(cal['y_validation'])));native=experiment.OUT/'source_native'/f'{tid}.npz'
        info=json.loads(native.with_suffix('.json').read_text())
        assert info['hashes']=={k:protocol.array_sha256(v) for k,v in [('X_source',pool['X'][pos]),('y_source',pool['y'][pos]),('X_validation',cal['X_validation'])]}
        value,_=predictor(pool['X'][pos],pool['y'][pos],cal['X_validation'],members=1,random_state=0)
        with np.load(native) as z:
            error=float(np.max(np.abs(value-z['prediction'])))/scale;max_native_error=max(max_native_error,error)
            assert error<1e-5,(tid,error);assert np.array_equal(z['y_validation'],cal['y_validation'])
        native_cases+=1
        experts=experiment.OUT/'source_experts'/f'{tid}.npz';info=json.loads(experts.with_suffix('.json').read_text())
        assert info['single_source_ids']==pool['source_ids'][pos].tolist()
        with np.load(experts) as z:
            assert np.array_equal(z['y_validation'],cal['y_validation'])
            records=[('single',i,0) for i in range(len(parent.PROFILES))]+[('bag',i,k) for i in range(len(bag.PROFILES)) for k in range(bag.MEMBERS)]
            for kind,i,k in records:
                path=experts.parent/'models'/tid/(f'single_{i}.joblib' if kind=='single' else f'bag_{i}_{k}.joblib');model=joblib.load(path)
                if kind=='bag':
                    selected=bag.subset(pool,cal,8192,k,tid,'validation')
                    row=next(r for r in info['bagged_records'] if r['profile_index']==i and r['member']==k)
                    assert row['source_original_ids']==pool['source_ids'][selected].tolist()
                    assert row['source_X_sha256']==protocol.array_sha256(pool['X'][selected]) and row['source_y_sha256']==protocol.array_sha256(pool['y'][selected])
                value=parent.h.head_predict(cal['X_validation'],model);saved=z['single'][i] if kind=='single' else z['bagged'][i,k]
                error=float(np.max(np.abs(value-saved)))/scale;max_head_error=max(max_head_error,error);assert error<1e-10,(tid,kind,i,k,error)
                e=model['estimator'];n=getattr(e,'n_iter_',0);limit=getattr(e,'max_iter',None) or float('inf')
                if n is not None and np.max(np.atleast_1d(n))>=limit:solver_flags.append(dict(task_id=meta['task_id'],kind=kind,profile=i,member=k))
                source_models+=1
        old=bag.OUT/'predictions/8192/regression/clean'/tid/'original_0.npz'
        with np.load(old) as z:base=z['Base'];sh=z['Shared'];truth=z['y_target']
        choice=policy['Specific'];sp=sh
        if choice is not None:
            recipe=choice['recipe'];data,_=parent.h.source.adaptation.raw.scaling.bank('regression','clean',meta,'original',0);values=[]
            if recipe['kind']=='single':paths=[parent.OUT/'experts/8192/clean/models'/tid/f'query_{recipe["profile_index"]}.joblib']
            else:paths=[bag.OUT/'experts/8192/clean/models'/tid/f'query_{recipe["profile_index"]}_{k}.joblib' for k in range(recipe['members'])]
            for path in paths:values.append(parent.h.head_predict(data['X_target'],joblib.load(path)))
            a=choice['blend'];sp=(1-a)*sh+a*np.mean(values,axis=0)
        out=experiment.OUT/'predictions/8192/regression/clean'/tid/'original_0.npz';record=json.loads(out.with_suffix('.json').read_text())
        assert freeze.stat().st_mtime<=out.stat().st_mtime
        with np.load(out) as z:
            assert np.array_equal(z['Base'],base) and np.array_equal(z['Shared'],sh) and np.array_equal(z['y_target'],truth)
            error=float(np.max(np.abs(z['Specific']-sp)))/max(1.,float(np.std(truth)));max_head_error=max(max_head_error,error);assert error<1e-10
            for name in ['Base','Shared','Specific']:
                measured=metrics(truth,z[name],'regression',record['scale']);assert all(abs(v-record['methods'][name][k])<1e-12 for k,v in measured.items())
        conditions+=1
    assert conditions==35 and source_models==1750 and native_cases==35
    atomic_json(experiment.OUT/'raw_audit.json',dict(status='PASS',verified_conditions=conditions,source_models_replayed=source_models,
        source_native_cases_replayed=native_cases,max_normalized_native_error=max_native_error,max_normalized_head_error=max_head_error,
        all_source_calibration_rows_disjoint=True,source_core256_unchanged=True,per_learner_row_cap8192=True,
        Base_and_Shared_bitwise_unchanged=True,all_metrics_recomputed=True,policies_frozen_before_query_predictions=True,
        solver_iteration_limit_flags=solver_flags,solver_convergence_claim=False,time_unix=time.time()))
    print('EXPANDED_CALIBRATION_RAW_AUDIT_PASS',conditions,source_models,native_cases,max_native_error,max_head_error,flush=True)


if __name__=='__main__':audit()
