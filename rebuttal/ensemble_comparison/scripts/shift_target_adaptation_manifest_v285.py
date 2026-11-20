"""Independent source references and a fixed target-adaptation/evaluation split.

Only this preparation step opens the original target-label archive, to allocate
the authorized adaptation labels and write a separate held-out evaluation file.
All methods are scored on the identical held-out subset. Split seed0 is fixed
without observing outcomes, and is reused across every generator/strength.
"""
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import unseen_shared_robustness_v117 as source
from prepare_data import LOCAL_CC18,RUN,ROOT,atomic_json,protocol

OUT=RUN/'results/shift_target_adaptation_manifest_v285'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    contract={'scope':{'perturbations':['cell_heavy_tail','column_desynchronization','feature_quantization'],
        'shifts':['nonlinear_sensor','cross_feature_rotation','interaction_sampling']},
        'classification_primary':'accuracy','regression_primary':'r2',
        'required_absolute_gain_strictly_greater_than':{'perturbation':.05,'shift':.02},
        'each_family_and_task_must_pass':True,'averaging':'four positive strengths per data_id, then equal dataset mean',
        'seed':0,'TFM_context_cap':256,'new_supervision_and_target_adaptation_must_be_marked':True,
        'no_adaptation_rows_in_final_evaluation':True,'Base_and_Shared_same_evaluation_rows':True,
        'test_adaptation_fraction':.25,'target_adaptation_labels_cap':64,
        'fixed_split_not_selected_using_metrics':True}
    atomic_json(OUT/'acceptance_contract.json',contract)
    plan={'status':'SUPERVISED_TARGET_CONTEXT_ADAPTATION_PREPARATION',
        'seed':0,'target_adaptation_count':'min(64,max(16,floor(original_query_rows*.25)))',
        'target_split':'seed0 uniform permutation, independent of labels or predictions',
        'external_source_reference_training_cap':1984,'source_calibration_reserved':64,
        'source_reference_calibration_and_entire_original_query_row_ids_disjoint':True,
        'test_adaptation_rows_removed_from_final_eval':True,
        'same_split_all_generators_and_strengths':True,
        'Base_recomputed_on_remaining_query_rows_from_cached_forecasts':True,
        'generator_and_original_source_context_unchanged':True,
        'script_sha256':source.sha(__file__),'Markdown_written':False}
    path=OUT/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:atomic_json(path,plan)
    manifest=[]
    for c in json.loads((source.OUT/'case_manifest.json').read_text()):
        if c['track']!='shift':continue
        task,tid=c['task'],c['task_id']
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X,y,Q,sid,qid=(z[k] for k in ('X_source','y_source','X_target','source_indices','target_indices'))
        n=min(64,max(16,len(Q)//4));order=np.random.default_rng(0).permutation(len(Q))
        adapt,evalpos=order[:n],order[n:]
        assert len(evalpos)>16 and set(adapt).isdisjoint(evalpos)
        assert source.sha(c['truth_file'])==c['truth_sha256']
        with np.load(c['truth_file']) as z:
            truth=z['y_target'];np.testing.assert_array_equal(z['target_indices'],qid)
        if task=='classification':
            folder=LOCAL_CC18/str(tid)
            frame=pd.read_parquet(folder/'X.parquet');labels=pd.read_parquet(folder/'y.parquet').iloc[:,0]
            valid=labels.notna().to_numpy();ids=np.flatnonzero(valid)
            allX,_=protocol._numeric_matrix(frame.loc[valid]);pos={int(row):i for i,row in enumerate(ids)}
            encoder=LabelEncoder().fit(labels.iloc[sid].to_numpy())
            np.testing.assert_array_equal(encoder.transform(labels.iloc[sid].to_numpy()),y)
            candidates=np.setdiff1d(ids,np.r_[sid,qid])
            candidates=candidates[labels.iloc[candidates].isin(encoder.classes_).to_numpy()]
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in sid]],X)
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in qid]],Q)
            get_y=lambda selected:encoder.transform(labels.iloc[selected].to_numpy()).astype(y.dtype)
        else:
            assert source.sha(c['original_raw_cache'])==c['original_cache_sha256']
            with np.load(c['original_raw_cache']) as z:ids,allX,ally=z['source_indices'],z['X_source'],z['y_source']
            pos={int(row):i for i,row in enumerate(ids)}
            candidates=np.setdiff1d(ids,np.r_[sid,qid])
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in sid]],X)
            np.testing.assert_array_equal(ally[[pos[int(i)] for i in sid]],y)
            get_y=lambda selected:ally[[pos[int(i)] for i in selected]]
        assert len(candidates)>64
        cal=np.random.default_rng(0).choice(candidates,64,replace=False)
        rest=np.setdiff1d(candidates,cal)
        train=np.random.default_rng(0).choice(rest,min(1984,len(rest)),replace=False)
        TX,Ty=allX[[pos[int(i)] for i in train]],get_y(train)
        CX,Cy=allX[[pos[int(i)] for i in cal]],get_y(cal)
        assert set(map(int,train)).isdisjoint(map(int,np.r_[sid,qid,cal]))
        assert set(map(int,cal)).isdisjoint(map(int,np.r_[sid,qid]))
        dest=OUT/'adaptation_inputs'/task/f'{tid}.npz'
        source.npwrite(dest,X_train_reference=TX,y_train_reference=Ty,train_reference_indices=train,
            X_calibration=CX,y_calibration=Cy,calibration_indices=cal,
            target_adaptation_positions=adapt,target_adaptation_y=truth[adapt],
            evaluation_positions=evalpos,target_adaptation_indices=qid[adapt],evaluation_indices=qid[evalpos])
        truthdest=OUT/'evaluation_truth'/task/f'{tid}.npz'
        source.npwrite(truthdest,y_target=truth[evalpos],target_indices=qid[evalpos])
        meta=dict(c)
        meta.update(adaptation_input_file=str(dest),adaptation_input_sha256=source.sha(dest),
            evaluation_truth_file=str(truthdest),evaluation_truth_sha256=source.sha(truthdest),
            target_adaptation_rows=n,evaluation_rows=len(evalpos),additional_source_training_labels=len(Ty),
            source_calibration_labels_reserved=64,adaptation_labels_come_from_original_test_subset=True,
            original_eval_rows_removed_for_adaptation=n,
            supervision_type='EXTERNAL_LABELED_REFERENCE_PLUS_LABELED_TARGET_ADAPTATION',
            no_final_eval_label_in_adaptation_archive=True)
        manifest.append(meta)
        print('SHIFT_ADAPTATION_READY',task,tid,'source_extra',len(Ty),'target_labels',n,'eval',len(evalpos),flush=True)
    assert len(manifest)==68
    atomic_json(OUT/'case_manifest.json',manifest)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(OUT/'audit.json',{'status':'PASS','cases':68,'conditions':848,
        'target_adaptation_disjoint_from_final_evaluation':True,
        'source_reference_rows_disjoint_from_entire_original_query':True,
        'original_source_features_and_labels_replayed':True,
        'target_adaptation_labels_in_model_input_but_final_evaluation_labels_separate':True,
        'Base_rescored_on_same_remaining_rows_required':True,'Markdown_bytes':0})


if __name__=='__main__':main()
