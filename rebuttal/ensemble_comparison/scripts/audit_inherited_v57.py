"""Verify all immutable five-perturbation outputs inherited by the v56 suite."""
import json
import time
import numpy as np
import bagged_specific_v56 as latest
import probability_head_v51 as classifier
import sparse_specific_v47 as regression
from prepare_data import atomic_json
from run_tabicl import metrics


def audit():
    main=latest.parent.h.source.adaptation.raw;rows=[];verified=0
    for budget in [256,1024,4096,8192]:
        for task,parent in [('classification',classifier.OUT),('regression',regression.OUT)]:
            root=latest.OUT/'predictions'/str(budget)/task/'five'
            proof=classifier.OUT/'raw_audits'/f'{budget}.json' if task=='classification' else regression.OUT/'saved_prediction_audit.json'
            assert json.loads(proof.read_text())['status']=='PASS'
            expected={(str(m['task_id']),f'{family}_{rho}') for m in main.frozen.cohort(task,'five') for family,rho in main.case_grid('five')}
            observed=set();count=0
            for path in sorted(root.glob('*/*.npz')):
                original=parent/'predictions'/str(budget)/task/'five'/path.parent.name/path.name
                assert path.resolve()==original.resolve()
                assert path.with_suffix('.json').resolve()==original.with_suffix('.json').resolve()
                assert path.stat().st_mtime<=proof.stat().st_mtime
                assert path.with_suffix('.json').stat().st_mtime<=proof.stat().st_mtime
                record=json.loads(path.with_suffix('.json').read_text())
                source=latest.parent.h.source.OUT/'banks/query'/str(budget)/task/'five'/path.parent.name/path.name
                with np.load(source) as bank,np.load(path) as z:
                    assert np.array_equal(z['Base'],bank['base']) and np.array_equal(z['y_target'],bank['y_target'])
                    for name in ['Base','Shared','Specific']:
                        assert np.isfinite(z[name]).all()
                        measured=metrics(z['y_target'],z[name],task,record['scale'])
                        for key,value in measured.items():
                            target=record['methods'][name][key]
                            assert (value is None and target is None) or abs(value-target)<1e-12
                observed.add((path.parent.name,path.stem));verified+=1;count+=1
            assert observed==expected and count==len(expected)
            rows.append(dict(budget=budget,task=task,conditions=count,original_model_replay_audit=str(proof)))
    assert verified==8560
    atomic_json(latest.OUT/'inherited_prediction_audit_v57.json',dict(status='PASS',verified_conditions=verified,
        groups=rows,exact_same_immutable_parent_files=True,parents_have_frozen_model_replay_audits=True,
        no_parent_prediction_changed_since_replay=True,Base_and_truth_exactly_match_frozen_banks=True,
        all_metrics_recomputed=True,time_unix=time.time()))
    print('INHERITED_RAW_AUDIT_PASS',verified,flush=True)


if __name__=='__main__':audit()
