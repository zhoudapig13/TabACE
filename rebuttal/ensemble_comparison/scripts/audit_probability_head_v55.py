"""Replay frozen classifier heads and recompute metrics for all v51 cases."""
import argparse
import json
import time
import numpy as np
import torch
import probability_head_v51 as head
from prepare_data import atomic_json
from run_tabicl import metrics


def audit(budget):
    torch.set_num_threads(1);verified=0;max_error=0.;blocks=[]
    for block in ['clean','five','seven']:
        root=head.OUT/'policies'/str(budget)/'classification'/block
        frozen=root/'all_policies_frozen.json';assert frozen.exists()
        shared=json.loads((root/'Shared/policy.json').read_text())['model']
        specific={};cache={};n=0
        for c in head.load('query',budget,block):
            family=c['meta']['family']
            if family not in specific:specific[family]=json.loads((root/f'Specific_{family}/policy.json').read_text())['model']
            sh=head.predict_cell(c,shared,c['base'],'cpu',cache)
            sp=head.predict_cell(c,specific[family],sh,'cpu',cache)
            path=head.OUT/'predictions'/str(budget)/c['key'];record=json.loads(path.with_suffix('.json').read_text())
            assert frozen.stat().st_mtime<=path.stat().st_mtime
            assert record['role_permutation']==c['order']
            assert record['input_hashes']==c['meta']['source_input_hashes']
            with np.load(path) as z:
                assert np.array_equal(z['Base'],c['base']) and np.array_equal(z['y_target'],c['y'])
                for name,value in [('Shared',sh),('Specific',sp)]:
                    error=float(np.max(np.abs(z[name]-value)));max_error=max(max_error,error)
                    assert error<1e-5,(str(path),name,error)
                for name in ['Base','Shared','Specific']:
                    observed=metrics(z['y_target'],z[name],'classification',1.)
                    for key,value in observed.items():
                        expected=record['methods'][name][key]
                        assert (expected is None and value is None) or abs(expected-value)<1e-12
            verified+=1;n+=1
        expected=len(head.source.adaptation.raw.frozen.cohort('classification',block))*dict(clean=1,five=20,seven=28)[block]
        assert n==expected;blocks.append(dict(block=block,cases=n))
    atomic_json(head.OUT/'raw_audits'/f'{budget}.json',dict(status='PASS',context=budget,verified_cases=verified,
        blocks=blocks,all_frozen_heads_replayed=True,CPU_GPU_max_probability_error=max_error,tolerance=1e-5,
        all_metrics_recomputed=True,Base_and_truth_exactly_match_frozen_banks=True,
        policies_frozen_before_predictions=True,time_unix=time.time()))
    print('PROBABILITY_RAW_AUDIT_PASS',budget,verified,max_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);a=p.parse_args();audit(a.budget)
