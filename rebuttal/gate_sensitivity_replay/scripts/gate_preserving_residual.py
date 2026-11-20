"""Gate-preserving residual calibration; retain and audit the V7 failure."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from replay_diagnostics import OUT,ROOT,write_json
from calibrated_regression_repairs import components
import common_mode_residual as core

DEST=OUT/'gate_preserving_v8'
_ungated_state=core.state

def state(b,k,t,beta):
    evidence,x=_ungated_state(b,k,t,beta)
    ag,alpha,f=components(b,k,t)
    original=alpha*(ag-b)
    activity=alpha**2
    return original+activity*(evidence-original),x*activity[:,None]

core.state=state
core.DEST=DEST

def prepare():
    p=DEST/'frozen_plan.json'
    if p.exists():raise RuntimeError('Do not overwrite evaluated plan')
    write_json(p,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_implementation_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        purpose='Keep V7 corruption gains while preventing unconstrained changes to low-activity original gates',
        hypothesis='An ungated residual intercept overwrites already accurate clean predictions',
        formula='delta=delta_original+alpha_original^2*(delta_evidence-delta_original+blend*X@beta); final=Base+6*source_SD*tanh(delta/(6*source_SD))',
        zero_original_gate_implies_zero_correction=True,ridges=core.RIDGES,shared_penalties=core.SHARED,blends=core.BLENDS,
        selection='Source inner datasets: nonclean loss <=Original+0.001, clean loss <=Original+0.005, stress loss <=Original; minimize nonclean loss +0.1 max(clean loss-original clean loss,0)',
        source_calibration_only=True,evaluation_labels_not_used_for_fitting_or_selection=True,
        original_benchmark_predictions_unchanged=True,backbone_calls=0,markdown_written=False,
        evaluation_status='Reconstructed iterative development evaluation; not a new independent test'))

def train(backbone,track):
    dataset=core.load(backbone,track)
    for record in core.policy_records(backbone,track):
        parent=json.loads((ROOT/record['path']).read_text());t=np.asarray(parent['parameters'])
        train_ids=set(parent['train_data_ids']);test_ids=set(parent['test_data_ids']);assert train_ids.isdisjoint(test_ids)
        prior=json.loads((OUT/'algorithm_repairs_v3'/backbone/track/f'fold_{parent["fold"]}.json').read_text())
        beta=np.asarray(prior['parameters']);outer=[c for c in dataset if c['id'] in train_ids]
        inner=min(c['fold'] for c in outer);innertrain=[c for c in outer if c['fold']!=inner];val=[c for c in outer if c['fold']==inner]
        fitdata=core.pack(innertrain,track);natural=core.pack([c for c in val if c['meta']['rho']!=0],track)
        clean=core.pack([c for c in val if c['meta']['rho']==0],track)
        baseline=core.oldloss(natural,t);cleanbase=core.oldloss(clean,t);stressbase=core.stressloss(natural,t,beta,None,0,old=True)
        candidates=[]
        for ridge in core.RIDGES:
            for shared in core.SHARED:
                coef=core.fit(fitdata,t,beta,ridge,shared)
                for blend in core.BLENDS:
                    normal=core.loss(natural,t,beta,coef,blend);cl=core.loss(clean,t,beta,coef,blend)
                    st=core.stressloss(natural,t,beta,coef,blend)
                    candidates.append(dict(ridge=ridge,shared_penalty=shared,blend=blend,natural=normal,clean=cl,stress=st,
                        baseline_natural=baseline,baseline_clean=cleanbase,baseline_stress=stressbase,
                        feasible=normal<=baseline+.001 and cl<=cleanbase+.005 and st<=stressbase,
                        score=normal+.1*max(cl-cleanbase,0.)))
        good=[c for c in candidates if c['feasible']];choice=min(good,key=lambda c:c['score']) if good else None
        coef=core.fit(core.pack(outer,track),t,beta,choice['ridge'],choice['shared_penalty']) if choice else None
        write_json(DEST/backbone/track/f'fold_{parent["fold"]}.json',dict(backbone=backbone,track=track,outer_fold=parent['fold'],
            source_data_ids=sorted(train_ids),evaluation_data_ids=sorted(test_ids),
            evidence_parameters=beta.tolist(),residual_coefficients=coef.tolist() if coef is not None else None,
            blend=choice['blend'] if choice else 0.,selected=choice,source_candidates=candidates,
            held_out_labels_used_for_fit=False,original_policy_sha256=record['sha256']))
        print('GATE_PRESERVING_FIT',backbone,track,parent['fold'],choice,flush=True)
    print('GATE_PRESERVING_TRAIN_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['train','evaluate'])
    p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    elif a.stage=='train':train(a.backbone,a.track)
    else:core.evaluate(a.backbone,a.track)
