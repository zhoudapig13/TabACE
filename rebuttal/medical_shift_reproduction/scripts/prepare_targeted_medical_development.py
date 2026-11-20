"""Move a fixed, disclosed subset of old test patients into adaptation.

Only this preparation step opens old test truth. The tuner receives the selected
32 labels and the original development labels; remaining evaluation truth is
written separately and is not part of the tuner's inputs.
"""
from datetime import datetime, timezone
import json
import numpy as np
from sklearn.model_selection import train_test_split
from run_public_shift import ROOT, array_hash, write_json

CASES=[('sepsis_A_to_B','ood'),('sepsis_B_to_A','id'),('readmission_other_to_ER','ood')]
DEST=ROOT/'data/processed/medical_test_assisted_v26'


def main():
    protocol=dict(seed=0, cases=[dict(task=t,domain=d) for t,d in CASES],
        original_test_patients_moved_to_adaptation=32, new_context_labels=16, new_calibration_labels=16,
        split='Stratified seed=0; one fixed split, never resampled by outcome or method ranking',
        original_target_fitting_budget=64, same_label_access_for_all_methods=True,
        all_methods_refit_or_recalibrate_using_same_available_labels=True,
        original_source_preprocessor_frozen=True, context_cap=256, native_estimators_per_branch=8,
        development_only_parameter_selection=True, original_test_cohorts_previously_observed=True,
        exploratory_test_assisted_development=True, original_paper_reproduction=False,
        other_table_rows_frozen=True, per_case_max_auxiliary_learners=12,
        methods='Base native8; Shared five-context fusion; Specific additional auxiliary residual fusion',
        stages=[dict(purpose='Add fixed adaptation labels without evaluating the same patients',
            hypothesis='Additional domain-matched supervision can improve calibration and expert correction',
            setting='32 moved patients, existing development cohorts unchanged',
            results='pending',analysis='pending',next_steps='Select all configurations on original development only, then evaluate once')])
    write_json(ROOT/'configs/medical_test_assisted_v26.json',protocol)
    for task,domain in CASES:
        old=ROOT/'data/processed/expanded_medical_development_v18'/task
        meta=json.loads((old/'metadata.json').read_text())
        with np.load(old/'inputs.npz') as z:p={k:z[k] for k in z.files}
        assert {k:array_hash(v) for k,v in p.items()}==meta['input_hashes']
        with np.load(ROOT/meta['original_inputs']/'test_truth.npz') as z:
            y=z['y_'+domain]
        x=p['X_'+domain];indices=np.arange(len(y))
        moved,remaining=train_test_split(indices,train_size=32,stratify=y,random_state=0)
        fit,cal=train_test_split(moved,train_size=16,stratify=y[moved],random_state=0)
        fit,cal,remaining=map(np.sort,[fit,cal,remaining])
        assert not set(fit).intersection(cal) and not set(moved).intersection(remaining)
        assert sorted(np.r_[fit,cal,remaining].tolist())==indices.tolist()
        original=json.loads((ROOT/'results/medical_common_native8_v24'/task/'frozen_selection.json').read_text())
        ti=np.asarray(original['target_context_indices']);ci=np.asarray(original['target_calibration_indices'])
        inputs=dict(X_source_train=p['X_train'],y_source_train=p['y_train'],
            X_target_train=p['X_dev_fit'][ti],y_target_train=p['y_dev_fit'][ti],
            X_source_cal=p['X_cal'],y_source_cal=p['y_cal'],
            X_target_cal=p['X_dev_fit'][ci],y_target_cal=p['y_dev_fit'][ci],
            X_development=p['X_'+domain+'_dev'],y_development=p['y_'+domain+'_dev'],
            X_evaluation=x[remaining],X_adaptation_fit=x[fit],y_adaptation_fit=y[fit],
            X_adaptation_cal=x[cal],y_adaptation_cal=y[cal])
        dest=DEST/(task+'_'+domain);dest.mkdir(parents=True,exist_ok=True)
        path=dest/'inputs.npz'
        if path.exists():
            oldmeta=json.loads((dest/'metadata.json').read_text())
            assert oldmeta['input_hashes']=={k:array_hash(v) for k,v in inputs.items()},'Frozen split cannot be replaced'
        else:
            np.savez_compressed(path,**inputs)
            np.savez_compressed(dest/'evaluation_truth.npz',y_evaluation=y[remaining])
        membership=dict(adaptation_fit=fit.tolist(),adaptation_cal=cal.tolist(),evaluation=remaining.tolist())
        write_json(dest/'metadata.json',dict(task=task,domain=domain,seed=0,protocol=protocol,
            original_inputs=str(old.relative_to(ROOT)),original_metadata=meta,
            membership=membership,original_test_n=len(y),evaluation_n=len(remaining),
            development_n=len(inputs['y_development']),original_target_context_indices=ti.tolist(),
            original_target_calibration_indices=ci.tolist(),
            input_hashes={k:array_hash(v) for k,v in inputs.items()},
            prepared_at_utc=datetime.now(timezone.utc).isoformat()))
        print(task,domain,'adaptation fit/cal',len(fit),len(cal),'remaining evaluation',len(remaining),
              'development',len(inputs['y_development']),flush=True)


if __name__=='__main__':main()
