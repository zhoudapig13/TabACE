"""Add one native-permutation context type to the frozen Specific candidate set.

Shared is fixed v62. Native8 uses the same canonical TRAIN context and row cap
as Base, with the official frozen backbone's label/feature ensemble. This is
an explicit compute expansion: up to30 configured estimators for Specific
versus11 for Shared. Source-only calibration selects its blend; query metrics
are never used for selection. No claim of matched inference budget is made.
"""
import argparse
import hashlib
import json
import os
import time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import weighted_gate_v62 as parent
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,normalize,metrics,CHECKPOINTS

OUT=RUN/'results/tabace_native_context_specific_v67'
BUDGETS=[1024,4096,8192]
POWERS=[0.,.25,.5,.75,1.]
TEMPERATURES=[.7,1.,1.4,2.]
BLENDS=[.1,.25,.5,.75,1.]
RAW=parent.prior.risk.head.source.adaptation.raw


def generate():
    import torch,warnings
    torch.set_num_threads(1);torch.manual_seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('classification','cuda:0')
    for meta in RAW.frozen.cohort('classification','clean'):
        tid=str(meta['task_id']);pool=RAW.scaling.prepare('classification','five',meta)
        split=RAW.validation.validation_split('classification','five',meta,pool) if hasattr(RAW,'validation') else None
        if split is None:
            import tabace_validation_banks_v28 as validation
            split=validation.validation_split('classification','five',meta,pool)
        for budget in BUDGETS:
            for role in ['validation','query']:
                out=OUT/'banks'/role/str(budget)/f'{tid}.npz'
                if out.exists() and out.with_suffix('.json').exists():continue
                if role=='validation':
                    data,_=parent.prior.risk.head.source.adaptation.validation.materialize('classification','clean',meta,'original',0,budget,pool,split)
                else:data,*_=RAW.query_input('classification','clean',meta,'original',0,budget,pool)
                archive=parent.prior.risk.head.source.OUT/'banks'/role/str(budget)/'classification/clean'/tid/'original_0.json'
                info=json.loads(archive.read_text());hashes={k:protocol.array_sha256(v) for k,v in data.items()}
                assert hashes==info['source_input_hashes']
                assert len(data['y_source'])<=budget
                started=time.monotonic();k=int(pool['y'].max())+1
                pred,_=predictor(data['X_source'],data['y_source'],data['X_target'],members=8,random_state=0,n_classes=k)
                assert np.isfinite(pred).all()
                # Query labels are deliberately omitted from the native bank.
                arrays=dict(prediction=pred)
                if role=='validation':arrays['y_validation']=data['y_target']
                RAW.scaling.npwrite(out,**arrays)
                atomic_json(out.with_suffix('.json'),dict(status='NATIVE_BANK_COMPLETE',task_id=meta['task_id'],dataset=meta['dataset'],
                    budget=budget,role=role,actual_context_rows=len(data['y_source']),n_estimators=8,seed=0,
                    input_hashes=hashes,query_labels_used_for_training=False,query_labels_stored=False,
                    checkpoint_finetuned=False,seconds=time.monotonic()-started))
                print('NATIVE_CONTEXT_BANK_COMPLETE',role,budget,tid,flush=True)


def source_anchor(cells,budget):
    root=parent.OUT/'policies'/str(budget)/'classification/clean';policy=json.loads((root/'all_policies_frozen.json').read_text())
    ids=sorted({int(c['meta']['data_id']) for c in cells});folds={d:i%4 for i,d in enumerate(ids)}
    models={f:json.loads((root/'Shared'/f'fold_{f}.json').read_text()) for f in range(4)}
    other={f:json.loads((root/'Specific_original'/f'fold_{f}.json').read_text()) for f in range(4)}
    result={};specific={}
    for c in cells:
        f=folds[int(c['meta']['data_id'])];assert int(c['meta']['data_id']) in models[f]['held_datasets']
        assert int(c['meta']['data_id']) not in models[f]['train_datasets']
        result[c['key']]=parent.apply(c,models[f]['model'],policy['Shared'],c['base'])
        assert int(c['meta']['data_id']) not in other[f]['train_datasets']
        specific[c['key']]=parent.apply(c,other[f]['model'],policy['Specific'],result[c['key']])
    return result,specific


def run(budget):
    cells=parent.load('validation',budget);baseline,previous=source_anchor(cells,budget);native={}
    for c in cells:
        p=OUT/'banks/validation'/str(budget)/f'{c["meta"]["task_id"]}.npz'
        info=json.loads(p.with_suffix('.json').read_text());assert info['input_hashes']==c['meta']['source_input_hashes']
        with np.load(p) as z:
            assert np.array_equal(z['y_validation'],c['y']);native[c['key']]=z['prediction']
    initial,best_value=parent.prior.score(cells,baseline);best=None;trials=[]
    for power in POWERS:
        for temperature in TEMPERATURES:
            values={c['key']:parent.transform(c,native[c['key']],power,temperature) for c in cells}
            for source in ['native','union_mean','previous_specific']:
                candidate=values if source=='native' else {key:normalize(.5*(values[key]+previous[key])) for key in values} if source=='union_mean' else previous
                for blend in BLENDS:
                    predictions={c['key']:normalize((1-blend)*baseline[c['key']]+blend*candidate[c['key']]) for c in cells}
                    scores,utility=parent.prior.score(cells,predictions)
                    eligible=scores['accuracy']>=initial['accuracy']-1e-10 and scores['auc']>=initial['auc']-1e-10
                    trials.append(dict(source=source,prior_power=power,temperature=temperature,blend=blend,source_scores=scores,utility=utility,eligible=eligible))
                    if eligible and utility>best_value+1e-10:best_value=utility;best=dict(source=source,prior_power=power,temperature=temperature,blend=blend)
    root=OUT/'policies'/str(budget)/'classification/clean'
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY_BANKS_READ',Specific=best,source_baseline=initial,trials=trials,
        Shared_frozen_from_v62=True,native_estimators_added=8,Shared_configured_estimators=11,Specific_max_configured_estimators=30,
        same_per_context_row_cap=True,matched_inference_budget_claim=False,checkpoint_finetuned=False,
        query_labels_used_for_selection=False,seed=0))
    rows=[]
    for c in parent.load('query',budget):
        original=parent.OUT/'predictions'/str(budget)/c['key']
        with np.load(original) as z:base=z['Base'];shared=z['Shared'];truth=z['y_target'];previous=z['Specific']
        specific=shared
        if best is not None:
            p=OUT/'banks/query'/str(budget)/f'{c["meta"]["task_id"]}.npz';info=json.loads(p.with_suffix('.json').read_text())
            assert info['input_hashes']==c['meta']['source_input_hashes']
            with np.load(p) as z:value=parent.transform(c,z['prediction'],best['prior_power'],best['temperature'])
            if best['source']=='union_mean':value=normalize(.5*(value+previous))
            elif best['source']=='previous_specific':value=previous
            a=best['blend'];specific=normalize((1-a)*shared+a*value)
        out=OUT/'predictions'/str(budget)/c['key'];RAW.scaling.npwrite(out,Base=base,Shared=shared,Specific=specific,y_target=truth)
        record=json.loads(original.with_suffix('.json').read_text());record.update(adaptation_version='v67',Shared_bitwise_unchanged_from_v62=True,
            native_context_type_added=True,native_estimators_added=8,Shared_configured_estimators=11,Specific_max_configured_estimators=30,
            matched_inference_budget_claim=False,
            methods={name:metrics(truth,p,'classification',1.) for name,p in [('Base',base),('Shared',shared),('Specific',specific)]})
        atomic_json(out.with_suffix('.json'),record)
        for name,values in record['methods'].items():rows.append(dict(context=budget,block='clean',method=name,task_id=c['meta']['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=72,Shared_unchanged=True))
    print('NATIVE_CONTEXT_SPECIFIC_COMPLETE',budget,flush=True)


def plan():
    assert not (OUT/'plan.json').exists();parent.prior.risk.head.source.verify_registry()
    checkpoint=CHECKPOINTS/'tabicl-classifier-v2-20260212.ckpt'
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_NEW_NATIVE_INFERENCE',seed=0,budgets=BUDGETS,
        full_goal_checks=48,refit='Specific Clean1024/4096/8192 only',Shared_frozen_from_v62=True,Clean256_frozen_from_v62=True,
        new_context_type='canonical context with official Native8 feature/label permutation ensemble',
        original_TFM_contexts_preserved_as_candidates=True,Specific_source_candidates=['native','union_mean','previous_specific'],
        additional_native_estimators=8,Shared_configured_estimators=11,Specific_max_configured_estimators=30,
        per_estimator_row_cap_unchanged=True,matched_inference_budget_claim=False,checkpoint_finetuned=False,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        prior_powers=POWERS,temperatures=TEMPERATURES,blends=BLENDS,
        source_calibration='old disjoint TRAIN calibration; whole-dataset OOF parent Shared',
        query_native_banks='inferred without label use; no query labels stored; no query metrics before source selection',
        query_labels_used_for_fit_or_selection=False,query_feedback_used_for_development_design=True,
        development_only=True,original_paper_replication=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((parent.OUT/'paper_targets.json').read_text())
    for b in [256,1024,4096,8192]:
        for task in ['classification','regression']:
            for block in ['clean','five','seven']:
                if task=='classification' and block=='clean' and b!=256:continue
                origin=parent.OUT if task=='classification' and block=='clean' else RUN/'results/tabace_balanced_seven_v66' if task=='classification' else RUN/'results/tabace_clean_calibration_v59'
                for name in ['policies','predictions']:
                    dest=OUT/name/str(b)/task/block;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(origin/name/str(b)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','banks','fit']);p.add_argument('--budget',type=int);a=p.parse_args()
    if a.phase=='plan':plan()
    elif a.phase=='banks':generate()
    else:run(a.budget)
