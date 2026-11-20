"""Explicit RF-dominated extra-supervision mixture development, not original Specific."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
import torch
from threadpoolctl import threadpool_limits
from develop_specific_residual import fit_residual,apply_residual
from readmission_fresh_specialist import FreshWorker
import run_public_shift as exp


def main():
    w=FreshWorker(0);w.anchors()
    path=exp.ROOT/'results/readmission_fresh_specialist_v32/frozen_selection.json'
    parent=json.loads(path.read_text())
    with np.load(w.data/'fresh_specialist_predictions_v32.npz') as z:p={k:z[k] for k in z.files}
    assert np.array_equal(w.base,p['Base']) and np.array_equal(w.shared,p['Shared'])
    conf=parent['configurations']['Auxiliary'];tau=conf['threshold']
    expert=p['Auxiliary'];shared=p['Shared'];threshold=w.threshold
    best=p['Specific'].copy();best_score=w.score(best,threshold)
    best_conf=dict(kind='Frozen v32 Specific',threshold=threshold,full_configuration=parent['configurations']['Specific'])
    s=logit(np.clip(shared,1e-6,1-1e-6))-logit(threshold)
    t=logit(np.clip(expert,1e-6,1-1e-6))-logit(tau)
    trials=[]
    for alpha in [.7,.8,.9,.95,.98]:
        pred=expit((1-alpha)*s+alpha*t+logit(threshold));score=w.score(pred,threshold)
        config=dict(kind='RF-dominated constant logit mixture',alpha=alpha,threshold=threshold,
                    expert_threshold=tau,expert_configuration=conf)
        trials.append(dict(configuration=config,development_accuracy=score))
        if score>best_score+1e-12:best=pred;best_score=score;best_conf=config
    raw=(logit(np.clip(expert,1e-6,1-1e-6))-conf['platt_b'])/conf['platt_a']
    for share in [.5,.8,1.]:
        teacher,oof,teacher_tau,calconf=w.calibrated(raw,raw[:w.ncal],share)
        cs=logit(np.clip(w.shared_oof,1e-6,1-1e-6))-logit(threshold)
        ct=logit(np.clip(oof,1e-6,1-1e-6))-logit(teacher_tau)
        for cap in [.9,.98]:
            for reg in [.01,.1]:
                gate=fit_residual(cs,ct,w.cal_y,w.weights(share),cap,reg)
                pred=apply_residual(shared,teacher,threshold,teacher_tau,gate)
                score=w.score(pred,threshold)
                config=dict(kind='RF residual gate',threshold=threshold,expert_threshold=teacher_tau,
                            gate=gate,expert_configuration=conf,recalibration=calconf)
                trials.append(dict(configuration=config,development_accuracy=score))
                if score>best_score+1e-12:best=pred;best_score=score;best_conf=config
    output=dict(p,Specific=best)
    scores={m:w.score(v,conf['threshold'] if m=='Auxiliary' else threshold) for m,v in output.items()}
    report=dict(parent)
    report.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),evaluation_labels_opened=False,
        configurations=dict(parent['configurations'],Specific=best_conf),development_accuracy=scores,
        additional_rf_fusion_trials=trials,base_shared_predictions_bitwise_unchanged=True,
        extra_supervision_unchanged=True,specific_target_fitting_labels=416,fixed_controls_target_fitting_labels=96,
        method_status='RF-dominated extra-supervision hybrid development, not original TabACE Specific',
        parent_snapshot=str(path.relative_to(exp.ROOT)),parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    exp.write_json(exp.ROOT/'results/readmission_forest_fusion_v33/frozen_selection.json',report)
    np.savez_compressed(w.data/'readmission_forest_fusion_predictions_v33.npz',**output)
    print('RF_FUSION_FROZEN',scores,best_conf,flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):main()
