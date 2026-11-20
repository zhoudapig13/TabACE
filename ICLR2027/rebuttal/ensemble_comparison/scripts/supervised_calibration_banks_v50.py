"""REJECTED exploratory generator alternative; never use for goal results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from collections import OrderedDict
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import supervised_calibration_v50 as experiment
import source_constructor_memo_v35 as memo
from prepare_data import atomic_json,protocol
from run_tabicl import Predictor


def worker(index,count,limit=0,task_id=None):
    status=json.loads((experiment.OUT/'plan.json').read_text())['status']
    if status.startswith('REJECTED'):
        raise RuntimeError('v50 was rejected: frozen queries and existing calibration already use the same pseudo-label generator.')
    import torch,warnings
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    source=experiment.source;memo_enabled=memo.install_if_verified(source.adaptation)
    predictor=Predictor('classification','cuda:0');completed=0;failures=[]
    cohort=source.adaptation.raw.frozen.cohort('classification','five')
    if task_id is not None:cohort=[m for m in cohort if int(m['task_id'])==task_id]
    cohort=sorted(cohort,key=lambda m:(m['task_id'] not in [167124,167121],int(m['task_id'])))
    for meta in cohort:
        pool=None;partition=None;cache=OrderedDict()
        for budget in [8192,4096,1024,256]:
            for family,rho in source.adaptation.raw.case_grid('five'):
                relative=Path(str(budget))/'classification/five'/str(meta['task_id'])/f'{family}_{rho}.npz'
                owner=int.from_bytes(hashlib.sha256(str(relative).encode()).digest()[:8],'big')%count
                if owner!=index:continue
                out=experiment.OUT/'banks/validation'/relative
                if out.exists() and out.with_suffix('.json').exists():continue
                try:
                    if pool is None:
                        pool=source.adaptation.raw.scaling.prepare('classification','five',meta)
                        partition=source.full.validation.validation_split('classification','five',meta,pool)
                    payload=source.full.materialize('validation','classification','five',meta,family,rho,budget,pool,partition)
                    data=payload['canonical']
                    Q=experiment.shifted_features(meta,family,rho,pool['X'][:256],pool['y'][:256],partition['X_validation'],partition['y_validation'])
                    data=data|dict(X_target=np.asarray(Q,dtype=np.float32))
                    assert np.array_equal(data['y_target'],partition['y_validation'])
                    old=source.OUT/'banks/validation'/relative;oldmeta=json.loads(old.with_suffix('.json').read_text())
                    with np.load(old) as z:
                        oldbase=z['base'];oldmembers=z['members']
                        assert np.array_equal(z['y_target'],data['y_target'])
                    for key,p in zip(oldmeta['member_input_hashes'],oldmembers):cache[tuple(key)]=p
                    oldkey=tuple(oldmeta['source_input_hashes'][k] for k in ['X_source','y_source','X_target']);cache[oldkey]=oldbase
                    key=tuple(protocol.array_sha256(data[k]) for k in ['X_source','y_source','X_target'])
                    base_reused=key in cache
                    if base_reused:base=cache[key]
                    else:base,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=1,random_state=0,n_classes=oldbase.shape[1])
                    cache[key]=base
                    payload=payload|dict(canonical=data,base=base,canonical_input_hashes={k:protocol.array_sha256(v) for k,v in data.items()})
                    safe=source.full.constructor_input(payload)
                    assert 'y_target' not in safe['canonical']
                    common,additional=source.full.contexts(safe,'classification','five',family)
                    c=safe['canonical'];extra=source.adaptation.new_views(c['X_source'],c['y_source'],c['X_target'],'classification','five',family,meta)
                    views=common+extra[:5]+additional+extra[5:];assert len(views)==21
                    predictions=[];names=[];rows=[];hashes=[];aliases=[]
                    for name,A,b,T in views:
                        A,T=np.asarray(A,dtype=np.float32),np.asarray(T,dtype=np.float32)
                        assert 0<len(b)<=payload['actual_Base_context']<=budget
                        key=tuple(protocol.array_sha256(v) for v in [A,b,T]);alias=key in cache
                        if alias:p=cache[key]
                        else:p,_=predictor(A,b,T,members=1,random_state=0,n_classes=oldbase.shape[1])
                        assert p.shape==oldbase.shape and np.isfinite(p).all();cache[key]=p;cache.move_to_end(key)
                        while len(cache)>1024:cache.popitem(last=False)
                        predictions.append(p);names.append(name);rows.append(len(b));hashes.append(key);aliases.append(alias)
                    assert payload['canonical_input_hashes']=={k:protocol.array_sha256(v) for k,v in data.items()}
                    assert payload['pool_input_hashes']==dict(X_pool=protocol.array_sha256(payload['X_pool']),y_pool=protocol.array_sha256(payload['y_pool']))
                    source.adaptation.raw.scaling.npwrite(out,base=base,members=np.asarray(predictions),y_target=data['y_target'],
                        source_mean=np.asarray(float(payload['y_pool'].mean())),source_scale=np.asarray(max(float(payload['y_pool'].std()),1e-8)))
                    record=oldmeta|dict(status='SUPERVISED_TRAIN_CALIBRATION_BANK_COMPLETE',source_input_hashes=payload['canonical_input_hashes'],
                        shared_roles=names[:10],specific_additional_roles=names[10:],context_rows=rows,member_input_hashes=hashes,aliases=aliases,
                        new_TFM_calls=sum(not a for a in aliases)+(not base_reused),
                        Base_calibration_recomputed=not base_reused,original_validation_bank=str(old),
                        TRAIN_labels_used_by_calibration_generator=True,final_query_labels_used_for_generator=False,
                        calibration_truth_given_to_context_constructors=False,source_constructor_memo_enabled=memo_enabled,
                        classification_query_benchmark_unchanged=True,Base_query_predictions_unchanged=True,adaptation_version='v50')
                    atomic_json(out.with_suffix('.json'),record);completed+=1
                    print('SUPERVISED_CAL_CASE_COMPLETE',str(relative),flush=True)
                    if limit and completed>=limit:return
                except Exception:
                    error=traceback.format_exc();print(error,flush=True);failures.append(dict(case=str(relative),error=error))
                    atomic_json(experiment.OUT/'errors'/f'{index}_{meta["task_id"]}_{family}_{rho}_{budget}.json',failures[-1])
    atomic_json(experiment.OUT/'workers'/f'{index}.json',dict(status='COMPLETE' if not failures else 'FAILED',completed=completed,failures=failures))
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--count',type=int,default=8);p.add_argument('--limit',type=int,default=0);p.add_argument('--task-id',type=int);a=p.parse_args();worker(a.index,a.count,a.limit,a.task_id)
