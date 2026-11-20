"""Replay the new Clean fusion variants from immutable banks and policies."""
import argparse
import json
import time
import numpy as np
import weighted_gate_v62 as gate
import hierarchical_specific_v64 as hierarchical
import equivariant_specific_v65 as equivariant
from prepare_data import atomic_json
from run_tabicl import metrics,normalize


def audit(stage,budget):
    module={'v62':gate,'v64':hierarchical,'v65':equivariant}[stage]
    root=module.OUT/'policies'/str(budget)/'classification/clean';freeze=root/'all_policies_frozen.json';policy=json.loads(freeze.read_text())
    assert (root/'completion.json').exists()
    if stage=='v62':
        source=gate.load('validation',budget);ids={int(c['meta']['data_id']) for c in source}
        for name in ['Shared','Specific_original']:
            for fold in range(4):
                row=json.loads((root/name/f'fold_{fold}.json').read_text())
                assert not set(row['train_datasets'])&set(row['held_datasets'])
                assert set(row['train_datasets'])|set(row['held_datasets'])==ids
    if stage=='v64':
        for path in (root/'cv').glob('*.json'):
            row=json.loads(path.read_text());assert not set(row['train_datasets'])&set(row['held_datasets'])
    source_replayed=False
    if stage=='v65' and policy['Specific'] is not None:
        cells=hierarchical.load('validation',budget);p,folds,_,baseline=hierarchical.shared_source(cells,budget)
        extras=equivariant.source_extras(cells,budget,p,folds,baseline);chosen=policy['Specific'];profile=equivariant.PROFILES[chosen['profile_index']]
        predictions={}
        for c in cells:
            anchor=baseline[c['key']];D=equivariant.features(c,anchor,extras[c['key']],profile['member_prior_power']);value=np.empty_like(anchor)
            for fold in range(4):
                train=c['folds']!=fold;valid=~train
                if not valid.any():continue
                assert not np.intersect1d(c['ids'][train],c['ids'][valid]).size
                model=equivariant.fit(D[train],c['y'][train],anchor[train],c['metric_core_prior'],profile['regularization'])
                value[valid]=equivariant.predict(D[valid],anchor[valid],model)
            a=chosen['blend'];predictions[c['key']]=equivariant.normalize((1-a)*anchor+a*value)
        observed,_=gate.prior.score(cells,predictions)
        trial=next(r for r in policy['trials'] if r['profile_index']==chosen['profile_index'] and r['blend']==chosen['blend'])
        assert all(abs(observed[k]-trial['source_OOF_scores'][k])<1e-8 for k in observed)
        source_replayed=True
    maximum=0.;verified=0
    for c in (gate.load('query',budget) if stage=='v62' else hierarchical.load('query',budget)):
        if stage=='v62':
            base=c['base'];sh=gate.apply(c,policy['Shared_raw'],policy['Shared'],base);sp=gate.apply(c,policy['Specific_raw'],policy['Specific'],sh)
        else:
            original=gate.OUT/'predictions'/str(budget)/c['key']
            with np.load(original) as z:base=z['Base'];sh=z['Shared'];old=z['Specific']
            chosen=policy['Specific'];sp=sh
            if stage=='v64' and chosen is not None:
                recipe=hierarchical.RECIPES[chosen['recipe_index']]
                value=gate.gate.predict(hierarchical.bind(c,sh,recipe['member_prior_power']),policy['Specific_raw'])
                a=chosen['blend'];sp=normalize((1-a)*sh+a*hierarchical.sharpen(value,chosen['temperature']))
            if stage=='v65' and chosen is not None:
                with np.load(hierarchical.OUT/'predictions'/str(budget)/c['key']) as z:previous=z['Specific']
                profile=equivariant.PROFILES[chosen['profile_index']];D=equivariant.features(c,sh,[old,previous],profile['member_prior_power'])
                value=equivariant.predict(D,sh,policy['models'][str(c['meta']['task_id'])]);a=chosen['blend'];sp=normalize((1-a)*sh+a*value)
        out=module.OUT/'predictions'/str(budget)/c['key'];record=json.loads(out.with_suffix('.json').read_text())
        assert freeze.stat().st_mtime<=out.stat().st_mtime
        assert record['input_hashes']==c['meta']['source_input_hashes'] and record['role_permutation']==c['order']
        with np.load(out) as z:
            assert np.array_equal(z['Base'],base) and np.array_equal(z['Base'],c['base']) and np.array_equal(z['y_target'],c['y'])
            if stage!='v62':assert np.array_equal(z['Shared'],sh)
            for name,value in [('Shared',sh),('Specific',sp)]:
                error=float(np.max(np.abs(z[name]-value)));maximum=max(maximum,error);assert error<1e-12
            for name in ['Base','Shared','Specific']:
                measured=metrics(z['y_target'],z[name],'classification',1.)
                for key,value in measured.items():
                    target=record['methods'][name][key]
                    assert (value is None and target is None) or abs(value-target)<1e-12
        verified+=1
    assert verified==72
    atomic_json(module.OUT/'raw_audits'/f'{budget}.json',dict(status='PASS',stage=stage,budget=budget,verified_conditions=verified,
        all_saved_predictions_replayed=True,maximum_probability_error=maximum,source_dataset_folds_disjoint=True,
        selected_original_row_OOF_scores_reproduced=source_replayed,Base_and_truth_match_frozen_banks=True,
        all_metrics_recomputed=True,Shared_bitwise_unchanged_from_v62=stage!='v62',policies_frozen_before_predictions=True,time_unix=time.time()))
    print('CLEAN_FUSION_AUDIT_PASS',stage,budget,verified,maximum,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['v62','v64','v65'],required=True);p.add_argument('--budget',type=int,required=True);a=p.parse_args();audit(a.stage,a.budget)
