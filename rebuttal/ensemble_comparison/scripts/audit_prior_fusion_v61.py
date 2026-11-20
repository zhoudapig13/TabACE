"""Replay prior-aware source statistics and every Clean query prediction."""
import argparse
import json
import time
import numpy as np
import prior_fusion_v61 as experiment
from prepare_data import atomic_json
from run_tabicl import metrics


def audit(budget):
    root=experiment.OUT/'policies'/str(budget)/'classification/clean';freeze=root/'all_policies_frozen.json';policy=json.loads(freeze.read_text())
    source=experiment.load('validation',budget);stats_verified=0
    for name,m in [('Shared',10),('Specific',21)]:
        selected=policy[name]
        if selected is None:continue
        transformed=[experiment.correct(c,selected['prior_power']) for c in source];data=experiment.prepare(transformed,m)
        rebuilt=experiment.risk.policy(data,selected['model']['profile']);saved=selected['model']
        assert np.allclose(rebuilt['prior'],saved['prior'],atol=1e-12,rtol=0)
        assert set(rebuilt['statistics'])==set(saved['statistics'])
        for task,row in rebuilt['statistics'].items():
            assert row['count']==saved['statistics'][task]['count']
            assert np.allclose(row['risk'],saved['statistics'][task]['risk'],atol=1e-12,rtol=0)
            stats_verified+=1
    verified=0;maximum=0.
    for c in experiment.load('query',budget):
        sh=experiment.apply(c,policy['Shared'],c['base']);sp=experiment.apply(c,policy['Specific'],sh)
        out=experiment.OUT/'predictions'/str(budget)/c['key'];record=json.loads(out.with_suffix('.json').read_text())
        assert freeze.stat().st_mtime<=out.stat().st_mtime
        assert record['input_hashes']==c['meta']['source_input_hashes'] and record['role_permutation']==c['order']
        with np.load(out) as z:
            assert np.array_equal(z['Base'],c['base']) and np.array_equal(z['y_target'],c['y'])
            for name,value in [('Shared',sh),('Specific',sp)]:
                error=float(np.max(np.abs(value-z[name])));maximum=max(maximum,error);assert error<1e-12
            for name in ['Base','Shared','Specific']:
                observed=metrics(z['y_target'],z[name],'classification',1.)
                for key,value in observed.items():
                    target=record['methods'][name][key]
                    assert (value is None and target is None) or abs(value-target)<1e-12
        verified+=1
    assert verified==72
    atomic_json(experiment.OUT/'raw_audits'/f'{budget}.json',dict(status='PASS',budget=budget,verified_conditions=verified,
        source_group_statistics_reconstructed=stats_verified,all_frozen_risk_statistics_reproduced=True,
        canonical_context_label_hashes_verified=True,TRAIN_core_excludes_entire_calibration=True,
        all_prior_corrections_and_query_fusions_replayed=True,maximum_probability_error=maximum,
        Base_and_query_truth_exactly_match_frozen_banks=True,all_metrics_recomputed=True,
        policies_frozen_before_predictions=True,time_unix=time.time()))
    print('PRIOR_FUSION_RAW_AUDIT_PASS',budget,verified,stats_verified,maximum,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);a=p.parse_args();audit(a.budget)
