"""Specific residual gates with exact per-context TRAIN label priors.

Freeze v62 Shared. Train Specific on top of that complete policy, rather than
anchoring an independent gate to Native1. Context prior correction uses v63's
hash-verified per-member labels. Entire validation datasets are excluded from
both the parent gate and the residual gate used for their source predictions.
Only TRAIN-calibration predictions select recipes; query ranking is audited.
"""
import argparse
import json
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import weighted_gate_v62 as parent
import member_priors_v63 as histograms
from prepare_data import RUN,atomic_json
from run_tabicl import normalize,metrics

OUT=RUN/'results/tabace_hierarchical_specific_v64'
RECIPES=[dict(gain=g,regularization=r,member_prior_power=p) for g in [1.,2.] for r in [.01,.1] for p in [0.,1.]]
TEMPERATURES=[.7,1.,1.4,2.]
BLENDS=[.1,.25,.5,1.]


def load(role,budget):
    cells=parent.load(role,budget)
    for c in cells:
        tid=str(c['meta']['task_id']);p=histograms.OUT/role/str(budget)/f'{tid}.json';info=json.loads(p.read_text())
        assert info['status']=='PASS' and info['canonical_input_hashes']==c['meta']['source_input_hashes']
        assert info['member_input_hashes']==c['meta']['member_input_hashes']
        counts=np.asarray(info['member_label_counts'],dtype=float)[c['order']]+.5
        c['member_priors']=counts/counts.sum(1,keepdims=True)
        native=np.asarray(info['canonical_label_counts'],dtype=float)+.5
        assert np.allclose(native/native.sum(),c['context_prior'],atol=1e-15,rtol=0)
    return cells


def bind(c,anchor,power):
    ratio=np.power(c['core_prior'][None]/c['member_priors'],power)
    bank=np.asarray([normalize(p*r) for p,r in zip(c['bank'],ratio)])
    return c|dict(base=anchor,bank=bank)


def shared_source(cells,budget):
    root=parent.OUT/'policies'/str(budget)/'classification/clean';policy=json.loads((root/'all_policies_frozen.json').read_text())
    ids=sorted({int(c['meta']['data_id']) for c in cells});folds={d:i%4 for i,d in enumerate(ids)}
    anchors={};models={}
    for fold in range(4):
        info=json.loads((root/'Shared'/f'fold_{fold}.json').read_text());models[fold]=info['model']
        held={int(c['meta']['data_id']) for c in cells if folds[int(c['meta']['data_id'])]==fold}
        assert set(info['held_datasets'])==held and not held&set(info['train_datasets'])
        for c in cells:anchors[fold,c['key']]=parent.apply(c,models[fold],policy['Shared'],c['base'])
    baseline={c['key']:anchors[folds[int(c['meta']['data_id'])],c['key']] for c in cells}
    return policy,folds,anchors,baseline


def sharpen(p,temperature):
    z=np.log(np.clip(p,1e-12,1.))/temperature
    return normalize(np.exp(z-z.max(1,keepdims=True)))


def run(budget):
    assert budget in [1024,4096,8192]
    assert json.loads((histograms.OUT/'reconstruction_audit.json').read_text())['status']=='PASS'
    parent.configure();cells=load('validation',budget);policy,folds,anchors,baseline=shared_source(cells,budget)
    initial,best_value=parent.prior.score(cells,baseline);best=None;trials=[]
    root=OUT/'policies'/str(budget)/'classification/clean'
    for index,recipe in enumerate(RECIPES):
        predictions={}
        for fold in range(4):
            train=[bind(c,anchors[fold,c['key']],recipe['member_prior_power']) for c in cells if folds[int(c['meta']['data_id'])]!=fold]
            valid=[c for c in cells if folds[int(c['meta']['data_id'])]==fold]
            assert not {c['meta']['data_id'] for c in train}&{c['meta']['data_id'] for c in valid}
            model=parent.gate.fit(train,'classification',21,recipe['gain'],recipe['regularization'])
            atomic_json(root/'cv'/f'{index}_{fold}.json',dict(model=model,recipe=recipe,
                train_datasets=sorted({int(c['meta']['data_id']) for c in train}),held_datasets=sorted({int(c['meta']['data_id']) for c in valid})))
            for c in valid:predictions[c['key']]=parent.gate.predict(bind(c,anchors[fold,c['key']],recipe['member_prior_power']),model)
        for temperature in TEMPERATURES:
            values={key:sharpen(p,temperature) for key,p in predictions.items()}
            for blend in BLENDS:
                mixed={c['key']:normalize((1-blend)*baseline[c['key']]+blend*values[c['key']]) for c in cells}
                scores,utility=parent.prior.score(cells,mixed)
                eligible=scores['accuracy']>=initial['accuracy']-1e-10 and scores['auc']>=initial['auc']-1e-10
                trials.append(dict(recipe_index=index,temperature=temperature,blend=blend,source_scores=scores,utility=utility,eligible=eligible))
                if eligible and utility>best_value+1e-10:
                    best_value=utility;best=dict(recipe_index=index,temperature=temperature,blend=blend)
        print('HIERARCHICAL_RECIPE_COMPLETE',budget,index,flush=True)
    model=None
    if best is not None:
        recipe=RECIPES[best['recipe_index']]
        train=[bind(c,parent.apply(c,policy['Shared_raw'],policy['Shared'],c['base']),recipe['member_prior_power']) for c in cells]
        model=parent.gate.fit(train,'classification',21,recipe['gain'],recipe['regularization'])
    atomic_json(root/'all_policies_frozen.json',dict(status='FROZEN_BEFORE_QUERY',Specific=best,Specific_raw=model,recipes=RECIPES,
        source_baseline=initial,trials=trials,Shared_source=str(parent.OUT),Shared_unchanged=True,
        parent_and_residual_dataset_folds=4,exact_per_member_label_priors=True,query_labels_used_for_fit_or_selection=False,
        Specific_complete_Shared_fallback=True,seed=0))
    rows=[]
    for c in load('query',budget):
        original=parent.OUT/'predictions'/str(budget)/c['key']
        with np.load(original) as z:base=z['Base'];shared=z['Shared'];truth=z['y_target']
        specific=shared
        if best is not None:
            recipe=RECIPES[best['recipe_index']];value=parent.gate.predict(bind(c,shared,recipe['member_prior_power']),model)
            value=sharpen(value,best['temperature']);a=best['blend'];specific=normalize((1-a)*shared+a*value)
        out=OUT/'predictions'/str(budget)/c['key'];parent.prior.risk.head.source.adaptation.raw.scaling.npwrite(out,Base=base,Shared=shared,Specific=specific,y_target=truth)
        record=json.loads(original.with_suffix('.json').read_text());record.update(adaptation_version='v64',Shared_bitwise_unchanged_from_v62=True,
            exact_per_member_label_priors=True,nested_dataset_fold_parent=True,
            methods={name:metrics(truth,p,'classification',1.) for name,p in [('Base',base),('Shared',shared),('Specific',specific)]})
        atomic_json(out.with_suffix('.json'),record)
        for name,values in record['methods'].items():rows.append(dict(context=budget,block='clean',method=name,task_id=c['meta']['task_id'],**values))
    pd.DataFrame(rows).to_csv(root/'query_detail.csv',index=False)
    atomic_json(root/'completion.json',dict(status='COMPLETE',conditions=72,Shared_unchanged=True))
    print('HIERARCHICAL_SPECIFIC_COMPLETE',budget,flush=True)


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_HIERARCHICAL_FITS',seed=0,refit='Specific classification Clean1024/4096/8192 only',
        full_goal_checks=48,Shared_bitwise_unchanged_from_v62=True,accepted_Clean256_unchanged=True,
        recipes=RECIPES,temperatures=TEMPERATURES,blends=BLENDS,
        calibration='four dataset folds; parent and residual gate exclude same held datasets',
        new_candidate='residual confidence/agreement/Base gate anchored to the entire accepted Shared policy',
        context_priors='exact v63 histograms, all21 input hashes verified; no canonical proxy in residual members',
        clean_TRAIN_calibration_supervision=True,extra_TFM_calls=0,query_labels_used_for_fit_or_selection=False,
        query_feedback_used_for_development_design=True,development_only=True,original_paper_replication=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((parent.OUT/'paper_targets.json').read_text())
    for b in [256,1024,4096,8192]:
        for task in ['classification','regression']:
            for block in ['clean','five','seven']:
                if task=='classification' and block=='clean' and b!=256:continue
                origin=parent.OUT if task=='classification' and block=='clean' else RUN/'results/tabace_augmented_classifier_v54' if task=='classification' else RUN/'results/tabace_clean_calibration_v59'
                for name in ['policies','predictions']:
                    dest=OUT/name/str(b)/task/block;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(origin/name/str(b)/task/block,target_is_directory=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);a=p.parse_args();plan() if a.phase=='plan' else run(a.budget)
