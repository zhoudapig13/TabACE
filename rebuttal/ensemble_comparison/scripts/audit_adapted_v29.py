"""Full-scope numerical audit with correct labels for adapted v29 methods."""
import json
import audit_tabace_goals_v28 as audit
import tabace_context_adaptation_v29 as adapted
from prepare_data import atomic_json


if __name__=='__main__':
    root=adapted.OUT
    audit.main.OUT=root
    audit.report()
    path=root/'goal_audit.json';result=json.loads(path.read_text())
    result.update(methods='Adapted observed-source R10 Shared /R21 Specific reconstructions, with held-out clean-TRAIN fusion calibration',
        adaptation_marker='dagger',original_paper_replication=False,
        Specific_inherits_all_Shared_candidates=True,
        single_seed_experiment=True)
    atomic_json(path,result)
