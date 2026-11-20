"""Evaluate every frozen targeted adaptation on the same remaining patients.

Old-model controls are also rescored on these patients, separating a changed
evaluation cohort from the effect of additional labelled adaptation.
"""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from run_public_shift import ROOT,array_hash,metrics,paired,write_json
from prepare_targeted_medical_development import CASES


def main(version=26):
    tag='medical_test_assisted_v26' if version==26 else 'medical_test_assisted_paired_v27'
    prediction='frozen_predictions.npz' if version==26 else 'paired_frozen_predictions_v27.npz'
    out=ROOT/'results'/('medical_test_assisted_evaluation_v'+str(version))
    plan=[]
    for task,domain in CASES:
        path=ROOT/'results'/tag/(task+'_'+domain)/'frozen_selection.json'
        f=json.loads(path.read_text())
        assert f['evaluation_labels_opened'] is False
        assert f['original_test_labels_used_for_development']==32
        plan.append(dict(task=task,domain=domain,snapshot=str(path.relative_to(ROOT)),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(out/'frozen_evaluation_plan.json',dict(
        created_before_outcome_loading_at_utc=datetime.now(timezone.utc).isoformat(),experiments=plan,
        selection_does_not_use_remaining_evaluation_labels=True,
        old_test_patients_previously_observed=True,exploratory_test_assisted_development=True))
    rows=[];comparisons=[];audits=[]
    for item in plan:
        f=json.loads((ROOT/item['snapshot']).read_text());data=ROOT/f['data'];meta=f['metadata']
        with np.load(data/'inputs.npz') as z:p={k:z[k] for k in z.files}
        assert {k:array_hash(v) for k,v in p.items()}==meta['input_hashes']
        membership=meta['membership'];fit=set(membership['adaptation_fit']);cal=set(membership['adaptation_cal'])
        evaluation=set(membership['evaluation'])
        assert len(fit)==len(cal)==16 and not fit.intersection(cal)
        assert not evaluation.intersection(fit|cal)
        assert fit|cal|evaluation==set(range(meta['original_test_n']))
        with np.load(data/prediction) as z:pred={k:z[k] for k in z.files}
        n=f['query_layout']['evaluation'][1]
        assert all(len(x)==n and np.isfinite(x).all() and x.min()>=0 and x.max()<=1 for x in pred.values())
        # Only after every case and configuration has frozen, load remaining truth.
        with np.load(data/'evaluation_truth.npz') as z:y_eval=z['y_evaluation']
        detail=dict(task=item['task'],domain=item['domain'],metadata=meta,
                    experiment='Test-assisted exploratory development; not original paper reproduction',splits={})
        for split in ['development','evaluation']:
            sl=slice(*f['query_layout'][split])
            y=p['y_development'] if split=='development' else y_eval
            scores={m:metrics(y,x[sl],f['configurations'][m]['threshold']) for m,x in pred.items()}
            if split=='development':
                assert all(abs(v['accuracy']-f['development_accuracy'][m])<1e-12 for m,v in scores.items())
            acc={m:s['accuracy'] for m,s in scores.items()}
            d1=100*(acc['Shared']-acc['Base']);d2=100*(acc['Specific']-acc['Shared'])
            rows.append(dict(task=item['task'],domain=item['domain'],split=split,n=len(y),**acc,
                Shared_minus_Base_pp=d1,Specific_minus_Shared_pp=d2,
                both_steps_1to5pp=bool(1-1e-10<=d1<=5+1e-10 and 1-1e-10<=d2<=5+1e-10)))
            detail['splits'][split]=dict(n=len(y),positive_count=int(y.sum()),methods=scores)
            if split=='evaluation':
                detail['comparisons']={b+'_minus_'+a:paired(y,
                    pred[a][sl]>=f['configurations'][a]['threshold'],
                    pred[b][sl]>=f['configurations'][b]['threshold'])
                    for a,b in [('Base','Shared'),('Shared','Specific')]}
        # Frozen old models evaluated on exactly the new remaining evaluation set.
        old=ROOT/meta['original_inputs'];indices=np.asarray(membership['evaluation'])
        old_config=json.loads((ROOT/'results/medical_common_native8_v24'/item['task']/'frozen_selection.json').read_text())
        domain=item['domain'];old_slice=slice(*old_config['query_layout'][domain+'_test'])
        old_tau=old_config['configurations'][domain]['Base']['policy']['threshold']
        with np.load(old/'medical_common_native8_predictions.npz') as z:
            old_p={m:z[domain+'_'+m][old_slice][indices] for m in ['Base','Shared']}
        with np.load(old/'specific_native8_pool_predictions.npz') as z:
            old_p['Specific']=z[domain+'_Specific_N8P'][old_slice][indices]
        sl=slice(*f['query_layout']['evaluation'])
        for m,old_prob in old_p.items():
            previous=metrics(y_eval,old_prob,old_tau)
            current=metrics(y_eval,pred[m][sl],f['configurations'][m]['threshold'])
            comparisons.append(dict(task=item['task'],domain=domain,method=m,n=len(y_eval),
                old_model_same_remaining_patients_acc=previous['accuracy'],
                new_adapted_model_acc=current['accuracy'],
                adaptation_delta_pp=100*(current['accuracy']-previous['accuracy'])))
        detail['same_remaining_patients_old_models']={m:metrics(y_eval,x,old_tau) for m,x in old_p.items()}
        write_json(out/(item['task']+'_'+domain)/'metrics.json',detail)
        audits.append(dict(task=item['task'],domain=domain,input_hashes_match=True,
            disjoint_adaptation_fit_calibration_evaluation=True,one_fixed_split=True,
            same_label_access_all_methods=True,evaluation_patients=len(y_eval)))
    table=pd.DataFrame(rows);table.to_csv(out/'accuracy_all.csv',index=False)
    old_new=pd.DataFrame(comparisons);old_new.to_csv(out/'same_evaluation_patients_old_vs_new.csv',index=False)
    test=table[table.split=='evaluation']
    write_json(out/'verification.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        audits=audits,cases_evaluated=len(test),cases_meeting_both_steps_1to5pp=int(test.both_steps_1to5pp.sum()),
        all_frozen_before_loading_remaining_outcomes=True,all_candidates_retained=True,
        previously_observed_test_cohorts=True,confirmatory_generalization_claim=False,markdown_written=False,
        script_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT/'scripts').glob('*targeted_medical*.py')}))
    print('NEW ADAPTATIONS',test.to_string(index=False))
    print('SAME REMAINING PATIENTS',old_new.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,choices=[26,27],default=26)
    main(p.parse_args().version)
