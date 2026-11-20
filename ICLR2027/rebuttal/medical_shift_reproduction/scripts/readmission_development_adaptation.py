"""Readmission-only additional development supervision, with unchanged evaluation.

Fix64 development patients for fitting and32 for calibration once, seed0. Leave
the remaining578 development patients for tuning. Keep all780 evaluation patients
and their outcomes entirely outside the tuner.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
from tune_targeted_medical import TargetedWorker

NAME='readmission_other_to_ER_ood'
OLD=exp.ROOT/'data/processed/medical_test_assisted_v26'/NAME
NEWROOT=exp.ROOT/'data/processed/readmission_development_adaptation_v29'


def prepare():
    meta=json.loads((OLD/'metadata.json').read_text())
    with np.load(OLD/'inputs.npz') as z:p={k:z[k] for k in z.files}
    assert {k:exp.array_hash(v) for k,v in p.items()}==meta['input_hashes']
    moved,remaining=train_test_split(np.arange(len(p['y_development'])),train_size=96,
                                     stratify=p['y_development'],random_state=0)
    fit,cal=train_test_split(moved,train_size=64,stratify=p['y_development'][moved],random_state=0)
    fit,cal,remaining=map(np.sort,[fit,cal,remaining])
    assert not set(fit).intersection(cal) and not set(moved).intersection(remaining)
    inputs=dict(p)
    for role,indices in [('train',fit),('cal',cal)]:
        inputs['X_target_'+role]=np.r_[p['X_target_'+role],p['X_development'][indices]]
        inputs['y_target_'+role]=np.r_[p['y_target_'+role],p['y_development'][indices]]
    inputs['X_development']=p['X_development'][remaining]
    inputs['y_development']=p['y_development'][remaining]
    assert np.array_equal(inputs['X_evaluation'],p['X_evaluation'])
    dest=NEWROOT/NAME;dest.mkdir(parents=True,exist_ok=True)
    updated=dict(meta)
    updated.update(input_hashes={k:exp.array_hash(v) for k,v in inputs.items()},
        additional_development_membership=dict(fit=fit.tolist(),calibration=cal.tolist(),remaining=remaining.tolist()),
        extra_development_fit_labels=64,extra_development_calibration_labels=32,development_n=len(remaining),
        preceding_input_data=str(OLD.relative_to(exp.ROOT)),
        evaluation_truth_file=str((OLD/'evaluation_truth.npz').relative_to(exp.ROOT)),
        evaluation_patients_unchanged=True,total_target_fitting_labels=192,
        target_context_pool=112,target_calibration_pool=80,
        prepared_at_utc=datetime.now(timezone.utc).isoformat())
    if (dest/'inputs.npz').exists():
        assert json.loads((dest/'metadata.json').read_text())['input_hashes']==updated['input_hashes']
    else:np.savez_compressed(dest/'inputs.npz',**inputs)
    exp.write_json(dest/'metadata.json',updated)
    exp.write_json(exp.ROOT/'configs/readmission_development_adaptation_v29.json',dict(
        task='UCI Diabetes readmission, other admission sources to emergency room',seed=0,
        original_source_preprocessing_frozen=True,new_development_fit_labels=64,new_development_calibration_labels=32,
        original_test_adaptation32_unchanged=True,no_more_test_patients_reassigned=True,
        target_context_pool=112,target_calibration_pool=80,total_target_fitting_labels=192,
        source_training_labels=len(inputs['y_source_train']),remaining_development_labels=len(remaining),
        evaluation_patients=len(inputs['X_evaluation']),label_access_matched_all_methods=True,
        context_cap=256,native_estimators_per_branch=8,
        original_test_cohorts_previously_observed=True,exploratory_development_not_paper_reproduction=True))
    print('READMISSION_ADDITIONAL_DEVELOPMENT',dict(extra_fit=64,extra_calibration=32,
        remaining_development=len(remaining),unchanged_evaluation=len(inputs['X_evaluation'])),flush=True)


class DevelopmentWorker(TargetedWorker):
    def __init__(self,gpu):
        super().__init__('readmission_other_to_ER','ood',gpu,data_root=NEWROOT)
        self.results_tag='readmission_development_adaptation_v29'
        self.predictions_filename='readmission_development_predictions_v29.npz'
        self.calibration_shares=[.5,.8,1.]
        self.auxiliary_specs=[dict(kind='logistic',C=.003),dict(kind='logistic',C=.03),
            dict(kind='forest',depth=4),dict(kind='forest',depth=8),
            dict(kind='lightgbm',depth=3),dict(kind='lightgbm',depth=4)]
        self.previous=json.loads((exp.ROOT/'results/medical_test_assisted_paired_v27'/NAME/'frozen_selection.json').read_text())
        with np.load(OLD/'paired_frozen_predictions_v27.npz') as z:self.previous_predictions={k:z[k] for k in z.files}
        with np.load(OLD/'inputs.npz') as z:
            oldq=np.r_[z['X_source_cal'],z['X_target_cal'],z['X_adaptation_cal'],z['X_development'],z['X_evaluation']]
        assert np.array_equal(self.remap(oldq),self.queries)
        self.replacement_protocol='Fixed64 development fitting and32 calibration patients added to original pools; evaluation unchanged.'

    def remap(self,p):
        f=self.previous;nsc=self.nsc;old_ntc=32;old_ncal=f['query_layout']['calibration'][1]
        d0=f['query_layout']['development'][0];e0=f['query_layout']['evaluation'][0]
        idx=self.meta['additional_development_membership']
        return np.r_[p[:nsc+old_ntc],p[d0+np.asarray(idx['calibration'])],p[nsc+old_ntc:old_ncal],
                     p[d0+np.asarray(idx['remaining'])],p[e0:]]

    def normalize(self,p,threshold):
        tau=self.previous['configurations']['Base']['threshold']
        return p.copy() if abs(tau-threshold)<1e-12 else expit(logit(np.clip(p,1e-6,1-1e-6))-logit(tau)+logit(threshold))

    def additional_base_candidates(self):
        return [(self.remap(self.previous_predictions['Base']),dict(
            self.previous['configurations']['Base'],kind='Frozen preceding Base candidate'))]

    def additional_shared_candidates(self,threshold):
        return [(self.normalize(self.remap(self.previous_predictions['Shared']),threshold),
                 dict(kind='Frozen preceding Shared candidate',threshold=threshold))]

    def additional_specific_candidates(self,threshold):
        return [(self.normalize(self.remap(self.previous_predictions['Specific']),threshold),
                 dict(kind='Frozen preceding Specific candidate',threshold=threshold))]

    def frozen_shared_oof(self,threshold):
        with np.load(OLD/'readmission_focused_calibration_oof_v28.npz') as z:old_oof=z['Shared']
        # v28 retained v27 Shared; this file contains its held-out calibration scores.
        v28=json.loads((exp.ROOT/'results/readmission_focused_v28/frozen_selection.json').read_text())
        assert v28['configurations']['Shared']['kind']=='Frozen v27 Shared'
        old=self.previous_predictions['Shared'].copy();old[:len(old_oof)]=old_oof
        return self.normalize(self.remap(old),threshold)[:self.ncal]

    def execute(self):
        self.run()
        path=exp.ROOT/'results'/self.results_tag/NAME/'frozen_selection.json'
        f=json.loads(path.read_text())
        f.update(protocol=json.loads((exp.ROOT/'configs/readmission_development_adaptation_v29.json').read_text()),
            original_test_adaptation32_unchanged=True,extra_development_fit_labels=64,
            extra_development_calibration_labels=32,total_target_fitting_labels=192,
            no_new_evaluation_labels_used=True,all_other_tasks_unchanged=True,
            focus_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        exp.write_json(path,f)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):
        prepare();DevelopmentWorker(a.gpu).execute()
