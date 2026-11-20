"""Target-focused Specific, preserving the preceding stronger fixed controls.

The extra96 development labels and additional bootstrap inference are explicit.
This is an exploratory component experiment, not a supervision/compute-matched
comparison to the preserved v27 Base and Shared.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import torch
import run_public_shift as exp
from readmission_development_adaptation import DevelopmentWorker,NAME
from tune_readmission_focus import ReadmissionWorker
from develop_specific_residual import new_learner


class TargetSpecialist(DevelopmentWorker):
    calibrated=ReadmissionWorker.calibrated
    mapped=ReadmissionWorker.mapped
    consider_expert=ReadmissionWorker.consider_expert

    def execute_target(self):
        self.threshold=self.previous['configurations']['Base']['threshold']
        self.shared=self.remap(self.previous_predictions['Shared'])
        self.shared_oof=self.frozen_shared_oof(self.threshold)
        self.specific=self.remap(self.previous_predictions['Specific'])
        self.specific_score=self.score(self.specific,self.threshold)
        self.specific_config=dict(kind='Frozen v27 Specific',threshold=self.threshold)
        if self.score(self.shared,self.threshold)>self.specific_score+1e-12:
            self.specific=self.shared.copy();self.specific_score=self.score(self.shared,self.threshold)
            self.specific_config=dict(kind='Frozen v27 Shared fallback',threshold=self.threshold)
        self.trials=[];self.standalone_best=-1;self.standalone=None;self.standalone_config=None
        self.pooling_strengths=[.2,.4,.6,.8];self.residual_caps=[.4,.8]
        target=np.flatnonzero(self.fit_domains==1)
        assert len(target)==112
        xc=self.X[target]-self.X[target].mean(0);yc=self.y[target]-self.y[target].mean()
        association=np.abs(xc.T@yc)/np.maximum(np.sqrt((xc*xc).sum(0)*(yc*yc).sum()),1e-9)
        views=[np.arange(self.X.shape[1]),np.sort(np.argsort(-association,kind='stable')[:64])]
        for source_count in [0,64]:
            for view,cols in enumerate(views):
                members=[];rowsets=[]
                for seed in [0,1,2,3]:
                    rng=np.random.default_rng(seed);rows=[]
                    if source_count:
                        rows.extend(np.r_[tuple(rng.choice(np.flatnonzero(self.y[:self.ns]==c),source_count//2,
                                                          replace=False) for c in [0,1])].tolist())
                    rows.extend(target.tolist())
                    extra=256-source_count-len(target)
                    rows.extend(np.r_[tuple(rng.choice(target[self.y[target]==c],extra//2,
                                                      replace=True) for c in [0,1])].tolist())
                    rows=np.asarray(rows,dtype=int);rng.shuffle(rows)
                    assert len(rows)==256 and set(target).issubset(rows)
                    assert int((self.y[rows]==1).sum())==128
                    members.append(self.predict(rows,cols));rowsets.append(rows.tolist())
                mean=np.asarray(members).mean(0)
                self.consider_expert(mean,dict(expert='target-focused bootstrap Native-8 mean',
                    source_rows=source_count,target_rows=256-source_count,unique_target_patients=112,
                    bootstrap_members=4,native_estimators_per_member=8,bootstrap_seeds=[0,1,2,3],
                    view=view,columns=cols.tolist(),context_rows=rowsets,context_rows_exactly256=True,
                    extra_development_fit_labels=64,extra_development_calibration_labels=32))
                print('READMISSION_TARGET_BOOTSTRAP',source_count,view,
                      dict(Specific=self.specific_score,standalone=self.standalone_best),flush=True)
        for spec in [dict(kind='logistic',C=.03),dict(kind='logistic',C=.3),dict(kind='forest',depth=4)]:
            model=new_learner(spec);model.fit(self.X[target],self.y[target])
            self.consider_expert(model.predict_proba(self.queries)[:,1],dict(
                expert='target-only supervised learner',learner=spec,target_training_patients=112,
                extra_development_fit_labels=64,extra_development_calibration_labels=32))
        base=self.remap(self.previous_predictions['Base'])
        output=dict(Base=base,Shared=self.shared,Specific=self.specific,Auxiliary=self.standalone)
        scores={m:self.score(p,self.standalone_config['threshold'] if m=='Auxiliary' else self.threshold)
                for m,p in output.items()}
        report=json.loads((exp.ROOT/'results/readmission_development_adaptation_v29'/NAME/'frozen_selection.json').read_text())
        report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),evaluation_labels_opened=False,
            configurations=dict(Base=dict(self.previous['configurations']['Base']),
                Shared=dict(self.previous['configurations']['Shared']),Specific=self.specific_config,Auxiliary=self.standalone_config),
            development_accuracy=scores,target_specialist_trials=self.trials,
            original_v27_base_shared_preserved=True,supervision_and_compute_matched_to_fixed_controls=False,
            specific_target_fitting_labels=192,fixed_controls_target_fitting_labels=96,
            extra_specific_bootstrap_compute=True,all_other_tasks_unchanged=True,
            comparison_interpretation='Extra target-supervised component development with fixed earlier controls; not a fair matched-budget rebuttal comparison',
            focus_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        exp.write_json(exp.ROOT/'results/readmission_target_specialist_v31/frozen_selection.json',report)
        np.savez_compressed(self.data/'readmission_target_specialist_predictions_v31.npz',**output)
        print('READMISSION_TARGET_SPECIALIST_FROZEN',scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):TargetSpecialist(a.gpu).execute_target()
