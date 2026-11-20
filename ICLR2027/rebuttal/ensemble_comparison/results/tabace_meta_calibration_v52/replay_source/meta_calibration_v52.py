"""Cross-fitted TabICL calibration expert for Specific regression seven.

The expert conditions on the SAME clean TRAIN calibration labels already
available to the supervised variants. Inputs contain frozen backbone outputs,
optionally raw features; never fitted calibration-head outputs. Every held-out
original row is excluded from its meta context. Official checkpoint stays
frozen. A selected expert adds one disclosed TFM forward at deployment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from collections import OrderedDict
import time
import traceback
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import calibration_head_v45 as h
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor

OUT=RUN/'results/tabace_meta_calibration_v52'
SCOPES=['member_predictions','features_plus_predictions']


def features(bank,X,scope):
    P=np.column_stack([bank['base'],bank['members'].T])
    if scope=='features_plus_predictions':P=np.column_stack([X,P])
    assert np.isfinite(P).all()
    return np.asarray(P,dtype=np.float32)


def selected_rows(eligible,budget):
    eligible=np.asarray(eligible,dtype=int)
    return eligible[np.random.default_rng(0).permutation(len(eligible))[:min(budget,len(eligible))]]


def jobs():
    return [(budget,meta,family,rho) for budget in [256,1024,4096,8192]
        for meta in h.source.adaptation.raw.frozen.cohort('regression','seven')
        for family,rho in h.source.adaptation.raw.case_grid('seven')]


def validation_worker(index,count,limit=0):
    import torch,warnings
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0');cache=OrderedDict();failures=[];completed=0
    for budget,meta,family,rho in jobs():
        relative=Path(str(budget))/str(meta['task_id'])/f'{family}_{rho}.npz'
        if int.from_bytes(hashlib.sha256(str(relative).encode()).digest()[:8],'big')%count!=index:continue
        out=OUT/'validation'/relative
        if out.exists() and out.with_suffix('.json').exists():continue
        try:
            source=h.source.OUT/'banks/validation'/str(budget)/'regression/seven'/str(meta['task_id'])/out.name
            info=json.loads(source.with_suffix('.json').read_text())
            with np.load(source) as z:bank={k:z[k] for k in ['base','members','y_target']}
            cal=h.calibration(meta,'seven');X,y,ids=cal['X_validation'],cal['y_validation'],cal['validation_original_ids']
            assert np.array_equal(y,bank['y_target']) and len(np.unique(ids))==len(ids)
            assert protocol.array_sha256(X)==info['source_input_hashes']['X_target']
            F=h.folds(meta,'seven',len(y));stored=dict(y_validation=y,original_ids=ids,folds=F);records=[]
            for scope in SCOPES:
                matrix=features(bank,X,scope);prediction=np.empty(len(y),dtype=np.float64)
                for fold in range(4):
                    valid=np.flatnonzero(F==fold)
                    if not len(valid):continue
                    train=selected_rows(np.flatnonzero(F!=fold),budget)
                    assert 0<len(train)<=budget and len(train)<=info['actual_context']
                    assert not np.intersect1d(ids[train],ids[valid]).size
                    A,b,Q=matrix[train],y[train],matrix[valid]
                    key=tuple(protocol.array_sha256(v) for v in [A,b,Q]);alias=key in cache;start=time.monotonic()
                    if alias:p=cache[key]
                    else:p,_=predictor(A,b,Q,members=1,random_state=0)
                    assert np.isfinite(p).all();prediction[valid]=p;cache[key]=p;cache.move_to_end(key)
                    while len(cache)>512:cache.popitem(last=False)
                    records.append(dict(scope=scope,fold=fold,meta_context_rows=len(train),held_out_rows=len(valid),
                        input_hashes=dict(X_source=key[0],y_source=key[1],X_target=key[2]),
                        source_row_ids_hash=protocol.array_sha256(ids[train]),target_row_ids_hash=protocol.array_sha256(ids[valid]),
                        train_target_original_row_overlap=0,reused_exact_inputs=alias,seconds=time.monotonic()-start))
                stored[scope]=prediction
            h.source.adaptation.raw.scaling.npwrite(out,**stored)
            atomic_json(out.with_suffix('.json'),dict(task_id=meta['task_id'],dataset=meta['dataset'],data_id=meta['data_id'],
                family=family,rho=rho,budget=budget,source_bank=str(source),records=records,seed=0,
                source_y_hash=protocol.array_sha256(y),raw_calibration_X_hash=protocol.array_sha256(X),
                context_limit_respected=True,checkpoint_finetuned=False,extra_clean_TRAIN_supervision=True,
                original_query_data_loaded=False,development_only=True,original_paper_replication=False))
            completed+=1;print('META_CALIBRATION_SOURCE_COMPLETE',str(relative),flush=True)
            if limit and completed>=limit:return
        except Exception:
            error=traceback.format_exc();print(error,flush=True);failures.append(dict(case=str(relative),error=error))
            atomic_json(OUT/'errors'/f'{index}_{budget}_{meta["task_id"]}_{family}_{rho}.json',failures[-1])
    atomic_json(OUT/'workers'/f'{index}.json',dict(status='COMPLETE' if not failures else 'FAILED',completed=completed,failures=failures))
    if failures:raise SystemExit(1)


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_META_EXPERT_INFERENCE',full_goal_checks=48,
        purpose='Improve remaining Specific seven-regression nRMSE with a nonlinear frozen-TFM calibration expert',
        source_banks=str(h.source.OUT),parent_methods=str(RUN/'results/tabace_sparse_specific_v47'),
        contexts=[256,1024,4096,8192],datasets=33,conditions=3696,scopes=SCOPES,seed=0,n_estimators=1,
        group_folds=4,per_meta_branch_rows='min(requested budget, eligible clean TRAIN calibration rows)',
        context_selection='fixed seed0 permutation of eligible original TRAIN calibration rows; no duplicates',
        no_extra_labels_beyond_existing_calibration=True,fitted_head_predictions_not_used_as_meta_features=True,
        Shared_predictions_unchanged=True,Specific_retains_parent_fallback=True,
        selection='TRAIN-only OOF nRMSE, require no source R2 degradation; query generation after policies frozen',
        configured_TFM_calls=dict(Shared='Base+10 members',Specific_parent='Base+21 members',Specific_when_meta_selected='Base+21 members+1 meta expert'),
        checkpoint='official frozen TabICL-v2 regressor; no weight adaptation',checkpoint_finetuned=False,
        query_labels_used_for_fit=False,extra_clean_TRAIN_supervision=True,query_feedback_used_for_benchmark_design=True,
        Specific_family_visible=True,actual_strength_given_to_meta_model=False,
        development_only=True,original_paper_replication=False,independent_test_claim=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((h.source.OUT/'paper_targets.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','validation']);p.add_argument('--index',type=int,default=0);p.add_argument('--count',type=int,default=8);p.add_argument('--limit',type=int,default=0);a=p.parse_args();plan() if a.phase=='plan' else validation_worker(a.index,a.count,a.limit)
