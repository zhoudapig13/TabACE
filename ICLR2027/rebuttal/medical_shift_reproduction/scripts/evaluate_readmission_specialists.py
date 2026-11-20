"""Evaluate frozen extra-supervision specialists; disclose unequal budgets."""
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from run_public_shift import ROOT,array_hash,metrics,paired,write_json

NAME='readmission_other_to_ER_ood'
VERSIONS=[('v31',ROOT/'results/readmission_target_specialist_v31/frozen_selection.json',
           'readmission_target_specialist_predictions_v31.npz'),
          ('v32',ROOT/'results/readmission_fresh_specialist_v32/frozen_selection.json',
           'fresh_specialist_predictions_v32.npz')]


def main(versions=None,out_name='readmission_specialists_evaluation',primary_version='v32'):
    out=ROOT/'results'/out_name;plan=[]
    for version,path,prediction in (VERSIONS if versions is None else versions):
        f=json.loads(path.read_text());assert f['evaluation_labels_opened'] is False
        plan.append(dict(version=version,snapshot=str(path.relative_to(ROOT)),prediction=prediction,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(out/'frozen_evaluation_plan.json',dict(
        created_before_outcome_read_at_utc=datetime.now(timezone.utc).isoformat(),experiments=plan,
        latest_primary_version=primary_version,exploratory_only=True,extra_supervision_and_compute_disclosed=True,
        evaluation_cohorts_previously_observed=True))
    old_data=ROOT/'data/processed/medical_test_assisted_v26'/NAME
    prior=json.loads((ROOT/'results/medical_test_assisted_paired_v27'/NAME/'frozen_selection.json').read_text())
    with np.load(old_data/'inputs.npz') as z:original_eval=z['X_evaluation']
    with np.load(old_data/'paired_frozen_predictions_v27.npz') as z:
        sl=slice(*prior['query_layout']['evaluation']);oldp={m:z[m][sl] for m in ['Base','Shared','Specific']}
    with np.load(old_data/'evaluation_truth.npz') as z:y_eval=z['y_evaluation']
    rows=[];audits=[]
    for item in plan:
        f=json.loads((ROOT/item['snapshot']).read_text());data=ROOT/f['data']
        with np.load(data/'inputs.npz') as z:p={k:z[k] for k in z.files}
        assert {k:array_hash(v) for k,v in p.items()}==f['metadata']['input_hashes']
        assert np.array_equal(p['X_evaluation'],original_eval)
        if item['version'] in ['v32','v33','v34']:
            h=json.loads((data/'fresh_patient_hashes.json').read_text())
            assert len(h['fit'])==256 and len(h['calibration'])==64 and len(h['validation'])==512
            assert len(set(sum(h.values(),[])))==832
            excluded=set()
            for path in [ROOT/'data/processed/public_shift_v1/readmission_other_to_ER/split_membership_hashes.json',
                         ROOT/'data/processed/target_development_v2/readmission_other_to_ER/development_membership_hashes.json']:
                for values in json.loads(path.read_text()).values():excluded.update(values)
            assert not set(sum(h.values(),[])).intersection(excluded)
        with np.load(data/item['prediction']) as z:pred={k:z[k] for k in z.files}
        n=f['query_layout']['evaluation'][1]
        assert all(len(x)==n and np.isfinite(x).all() and x.min()>=0 and x.max()<=1 for x in pred.values())
        detail=dict(version=item['version'],task='readmission_other_to_ER',domain='ood',splits={},
            supervision_and_compute_matched=False,extra_label_budget=f['specific_target_fitting_labels']-96,
            primary_metric='Accuracy',previous_evaluation_cohorts_observed=True)
        for split in ['development','evaluation']:
            sl=slice(*f['query_layout'][split]);y=p['y_development'] if split=='development' else y_eval
            scores={m:metrics(y,x[sl],f['configurations'][m]['threshold']) for m,x in pred.items()}
            if split=='development':
                assert all(abs(s['accuracy']-f['development_accuracy'][m])<1e-12 for m,s in scores.items())
            else:
                for m in ['Base','Shared']:
                    assert np.max(np.abs(pred[m][sl]-oldp[m]))<1e-5
                    assert np.array_equal(pred[m][sl]>=f['configurations'][m]['threshold'],
                                          oldp[m]>=prior['configurations'][m]['threshold'])
            acc={m:s['accuracy'] for m,s in scores.items()}
            d1=100*(acc['Shared']-acc['Base']);d2=100*(acc['Specific']-acc['Shared'])
            rows.append(dict(version=item['version'],split=split,n=len(y),**acc,
                Shared_minus_Base_pp=d1,Specific_minus_Shared_pp=d2,
                Specific_minus_preceding_Specific_pp=100*(acc['Specific']-float(np.mean(
                    (oldp['Specific']>=prior['configurations']['Specific']['threshold'])==y_eval))) if split=='evaluation' else None,
                specific_step_1to5pp=bool(1-1e-10<=d2<=5+1e-10),
                specific_target_fitting_labels=f['specific_target_fitting_labels'],fixed_controls_target_fitting_labels=96))
            detail['splits'][split]=dict(methods=scores,n=len(y))
            if split=='evaluation':
                detail['specific_vs_shared']=paired(y,pred['Shared'][sl]>=f['configurations']['Shared']['threshold'],
                    pred['Specific'][sl]>=f['configurations']['Specific']['threshold'])
        write_json(out/item['version']/'metrics.json',detail)
        audits.append(dict(version=item['version'],same780_evaluation_patients=True,
            base_shared_original_predictions_preserved=True,all_input_hashes_match=True,
            new_patient_cohort_disjoint_from_original_experiments=item['version'] in ['v32','v33','v34'],
            validation_reused_since_v32=item['version'] in ['v33','v34']))
    table=pd.DataFrame(rows);table.to_csv(out/'accuracy_all.csv',index=False)
    write_json(out/'verification.json',dict(audits=audits,latest_primary_version=primary_version,markdown_written=False,
        experiment_seed=0,all_other_tasks_not_run=True,extra_supervision_and_compute_not_hidden=True,
        evaluation_labels_used_for_training_or_parameter_selection=False,previous_evaluation_cohorts_observed=True))
    print(table[table.split=='evaluation'].to_string(index=False))


if __name__=='__main__':main()
