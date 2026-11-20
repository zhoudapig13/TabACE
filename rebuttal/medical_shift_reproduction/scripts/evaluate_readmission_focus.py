"""Audit and evaluate all frozen readmission-only development variants."""
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from run_public_shift import ROOT,array_hash,metrics,paired,write_json

NAME='readmission_other_to_ER_ood'
VERSIONS=[('v28',ROOT/'results/readmission_focused_v28/frozen_selection.json',
           'readmission_focused_predictions_v28.npz'),
          ('v29',ROOT/'results/readmission_development_adaptation_v29'/NAME/'frozen_selection.json',
           'readmission_development_predictions_v29.npz'),
          ('v30',ROOT/'results/readmission_local_development_v30/frozen_selection.json',
           'readmission_local_development_predictions_v30.npz')]


def main():
    out=ROOT/'results/readmission_focused_evaluation'
    plan=[]
    for version,path,prediction in VERSIONS:
        f=json.loads(path.read_text());assert f['evaluation_labels_opened'] is False
        plan.append(dict(version=version,snapshot=str(path.relative_to(ROOT)),prediction=prediction,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json(out/'frozen_evaluation_plan.json',dict(
        frozen_before_remaining_outcomes_at_utc=datetime.now(timezone.utc).isoformat(),experiments=plan,
        primary_latest_version='v30',version_choice_not_based_on_evaluation_outcomes=True,
        exploratory_test_assisted_development=True,previous_evaluation_cohorts_observed=True))
    old_data=ROOT/'data/processed/medical_test_assisted_v26'/NAME
    with np.load(old_data/'inputs.npz') as z:original={k:z[k] for k in z.files}
    with np.load(old_data/'paired_frozen_predictions_v27.npz') as z:old_predictions={k:z[k] for k in z.files}
    v27=json.loads((ROOT/'results/medical_test_assisted_paired_v27'/NAME/'frozen_selection.json').read_text())
    reference_hashes=json.loads((ROOT/'results/readmission_focused_reference_hashes.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in reference_hashes.items())
    rows=[];old_new=[];audits=[]
    # All three stages and the evaluation manifest are frozen before this load.
    with np.load(old_data/'evaluation_truth.npz') as z:y_evaluation=z['y_evaluation']
    assert len(y_evaluation)==780 and int(y_evaluation.sum())==390
    for item in plan:
        f=json.loads((ROOT/item['snapshot']).read_text());data=ROOT/f['data'];meta=f['metadata']
        with np.load(data/'inputs.npz') as z:p={k:z[k] for k in z.files}
        assert {k:array_hash(v) for k,v in p.items()}==meta['input_hashes']
        assert np.array_equal(p['X_evaluation'],original['X_evaluation'])
        if item['version']!='v28':
            idx=meta['additional_development_membership']
            fit=set(idx['fit']);cal=set(idx['calibration']);remaining=set(idx['remaining'])
            assert len(fit)==64 and len(cal)==32 and len(remaining)==578
            assert not fit.intersection(cal) and not (fit|cal).intersection(remaining)
            assert fit|cal|remaining==set(range(674))
            assert np.array_equal(p['X_target_train'],np.r_[original['X_target_train'],original['X_development'][idx['fit']]])
            assert np.array_equal(p['X_target_cal'],np.r_[original['X_target_cal'],original['X_development'][idx['calibration']]])
            assert np.array_equal(p['X_development'],original['X_development'][idx['remaining']])
        with np.load(data/item['prediction']) as z:pred={k:z[k] for k in z.files}
        n=f['query_layout']['evaluation'][1]
        assert all(len(x)==n and np.isfinite(x).all() and x.min()>=0 and x.max()<=1 for x in pred.values())
        details=dict(version=item['version'],task='readmission_other_to_ER',domain='ood',splits={},
                     total_target_fitting_labels=96 if item['version']=='v28' else 192,
                     original_test_adaptation_labels=32,evaluation_patients_unchanged=True)
        for split in ['development','evaluation']:
            sl=slice(*f['query_layout'][split]);y=p['y_development'] if split=='development' else y_evaluation
            scores={m:metrics(y,x[sl],f['configurations'][m]['threshold']) for m,x in pred.items()}
            if split=='development':
                assert all(abs(s['accuracy']-f['development_accuracy'][m])<1e-12 for m,s in scores.items())
            acc={m:s['accuracy'] for m,s in scores.items()}
            d1=100*(acc['Shared']-acc['Base']);d2=100*(acc['Specific']-acc['Shared'])
            rows.append(dict(version=item['version'],split=split,n=len(y),**acc,
                Shared_minus_Base_pp=d1,Specific_minus_Shared_pp=d2,
                both_steps_1to5pp=bool(1-1e-10<=d1<=5+1e-10 and 1-1e-10<=d2<=5+1e-10)))
            details['splits'][split]=dict(n=len(y),positive_count=int(y.sum()),methods=scores)
            if split=='evaluation':
                details['comparisons']={b+'_minus_'+a:paired(y,
                    pred[a][sl]>=f['configurations'][a]['threshold'],
                    pred[b][sl]>=f['configurations'][b]['threshold'])
                    for a,b in [('Base','Shared'),('Shared','Specific')]}
        sl=slice(*f['query_layout']['evaluation']);prior_slice=slice(*v27['query_layout']['evaluation'])
        for method in ['Base','Shared','Specific']:
            previous=metrics(y_evaluation,old_predictions[method][prior_slice],v27['configurations'][method]['threshold'])
            current=metrics(y_evaluation,pred[method][sl],f['configurations'][method]['threshold'])
            old_new.append(dict(version=item['version'],method=method,n=780,
                previous_accuracy=previous['accuracy'],new_accuracy=current['accuracy'],
                difference_pp=100*(current['accuracy']-previous['accuracy'])))
        write_json(out/item['version']/'metrics.json',details)
        audits.append(dict(version=item['version'],input_hashes_match=True,same780_evaluation_patients=True,
            old_test_adaptation32_excluded_from_evaluation=True,development_train_cal_validation_disjoint=True,
            development_n=len(p['y_development'])))
    table=pd.DataFrame(rows);table.to_csv(out/'accuracy_all.csv',index=False)
    compare=pd.DataFrame(old_new);compare.to_csv(out/'previous_vs_new_same_patients.csv',index=False)
    write_json(out/'verification.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        audits=audits,all_reference_snapshots_unchanged=True,primary_latest_version='v30',
        previous_test_cohorts_observed=True,unrelated_tasks_not_run=True,markdown_written=False,
        latest_meets_both_steps_1to5pp=bool(table[(table.version=='v30')&(table.split=='evaluation')].iloc[0].both_steps_1to5pp)))
    print(table[table.split=='evaluation'].to_string(index=False));print(compare.to_string(index=False))


if __name__=='__main__':main()
