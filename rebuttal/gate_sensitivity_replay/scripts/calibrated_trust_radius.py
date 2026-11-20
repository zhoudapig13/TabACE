"""Single-factor development experiment: calibrate the trust radius of V3.

The learned gate, original fusion parameters, contexts, and input data are frozen.
Radius selection uses source-calibration records exclusively. R=6 is included
as a predeclared fixed control; all evaluated variants are retained.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,ROOT,cells,regress,write_json
from calibrated_regression_repairs import components
from correlated_risk_calibration import load,pack,policy_records

DEST=OUT/'calibrated_trust_v5'
RADII=[3.,4.5,6.,8.,12.]
MAGNITUDES=[.25,.5,1.,2.,3.,5.,8.,10.]

def prepare():
    path=DEST/'frozen_plan.json'
    if path.exists():raise RuntimeError('Do not overwrite evaluated plan')
    hashes=[]
    for backbone in ['tabicl','tabpfn']:
        for track in ['five','seven_reimplemented']:
            for f in range(5):
                p=OUT/'algorithm_repairs_v3'/backbone/track/f'fold_{f}.json'
                hashes.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_json(path,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_gate_hashes=hashes,
        purpose='Isolate the update-radius tradeoff while retaining the previously source-trained query-dependent gain',
        only_changed_factor='Trust radius',radius_candidates=RADII,predeclared_control_radius=6.,
        learned_gate_parameters_frozen=True,original_fusion_parameters_frozen=True,
        selection='Minimize source-calibration common-bias normalized MSE subject to natural normalized MSE <= Original +0.001; identity fallback if infeasible',
        calibration_bias_magnitudes=[.5,1.,2.,5.],calibration_both_signs=True,maximum_calibration_queries_per_condition=96,
        evaluation_bias_magnitudes=MAGNITUDES,patterns=['independent_random_signs','shared_bias_3of5','shared_bias_5of5'],
        same_label_free_biases_for_all_methods=True,held_out_labels_used_for_radius_selection=False,
        original_benchmark_unchanged=True,backbone_calls=0,markdown_written=False,
        evaluation_status='Reconstructed development evaluation; previous aggregate evaluations informed algorithm iteration'))

def predict(b,k,theta,beta,radius):
    ag,alpha,features=components(b,k,theta)
    if radius is None:return b+alpha*(ag-b)
    learned=np.clip(alpha*(1+.5*np.tanh(features@beta)),0,1)
    learned*=np.minimum(1.,(radius/np.maximum(np.abs(ag-b),1e-12))**2)
    p=b+learned*(ag-b)
    assert np.max(np.abs(p-b))<=radius+1e-7 and np.isfinite(p).all()
    return p

def select(backbone,track):
    data=load(backbone,track)
    for record in policy_records(backbone,track):
        policy=json.loads((ROOT/record['path']).read_text());theta=np.asarray(policy['parameters'])
        train_ids=set(policy['train_data_ids']);test_ids=set(policy['test_data_ids']);assert train_ids.isdisjoint(test_ids)
        parent=json.loads((OUT/'algorithm_repairs_v3'/backbone/track/f'fold_{policy["fold"]}.json').read_text())
        assert set(parent['source_data_ids'])==train_ids and set(parent['evaluation_data_ids'])==test_ids
        beta=np.asarray(parent['parameters'])
        b,k,y,w=pack([c for c in data if c['id'] in train_ids],track)
        old=predict(b,k,theta,beta,None);baseline=float(np.sum(w*(old-y)**2))
        records=[]
        for radius in RADII:
            p=predict(b,k,theta,beta,radius);natural=float(np.sum(w*(p-y)**2));stress=0.
            for magnitude in [.5,1.,2.,5.]:
                for sign in [-1.,1.]:
                    p=predict(b,k+sign*magnitude,theta,beta,radius)
                    stress+=float(np.sum(w*(p-y)**2))/8
            records.append(dict(radius=radius,natural=natural,stress=stress,baseline_natural=baseline,
                feasible=natural<=baseline+.001))
        feasible=[r for r in records if r['feasible']]
        choice=min(feasible,key=lambda r:(r['stress'],r['natural'],r['radius'])) if feasible else None
        write_json(DEST/backbone/track/f'fold_{policy["fold"]}.json',dict(backbone=backbone,track=track,
            outer_fold=policy['fold'],source_data_ids=sorted(train_ids),evaluation_data_ids=sorted(test_ids),
            gate_parameters=beta.tolist(),selected_radius=choice['radius'] if choice else None,
            source_selection_candidates=records,held_out_labels_used_for_selection=False,
            original_policy_sha256=record['sha256']))
        print('RADIUS_SELECTED',backbone,track,policy['fold'],choice,flush=True)
    print('RADIUS_SELECTION_COMPLETE',backbone,track,flush=True)

def evaluate(backbone,track):
    dest=DEST/backbone/track
    heads={fold:json.loads((dest/f'fold_{fold}.json').read_text()) for fold in range(5)}
    scores=[];condition_stress=[];pool=defaultdict(list)
    for i,m,rawb,rawk,y,sm,ss,theta,fold,ref,path in cells(backbone,track,'regression'):
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T;head=heads[fold];beta=np.asarray(head['gate_parameters'])
        radii=dict(Original=None,Learned_radius12=12.,Learned_radius6=6.,Learned_calibrated_radius=head['selected_radius'])
        initial={name:predict(b,k,theta,beta,radius) for name,radius in radii.items()}
        scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        old=initial['Original']*ss+sm
        assert abs(float(np.sqrt(np.mean((old-y)**2))/scale)-ref['nrmse'])<1e-8*max(1,ref['nrmse'])
        for method,p in initial.items():
            pred=p*ss+sm
            scores.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                outer_fold=fold,method=method,n=len(y),r2=1-float(np.sum((pred-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                nrmse=float(np.sqrt(np.mean((pred-y)**2))/scale)))
        rng=np.random.default_rng(0);sign=rng.choice([-1.,1.],len(b));independent=rng.choice([-1.,1.],k.shape)
        subset=np.argsort(rng.random(k.shape),axis=1)[:,:3];mask=np.zeros_like(k)
        np.put_along_axis(mask,subset,sign[:,None],axis=1)
        patterns=dict(independent_random_signs=independent,shared_bias_3of5=mask,shared_bias_5of5=sign[:,None]*np.ones_like(k))
        for magnitude in MAGNITUDES:
            for pattern,shift in patterns.items():
                matrix=k+magnitude*shift
                for method,radius in radii.items():
                    p=predict(b,matrix,theta,beta,radius);d=np.abs(p-initial[method]);pool[(pattern,magnitude,method)].append(d)
                    output=p*ss+sm
                    condition_stress.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                        method=method,pattern=pattern,magnitude_source_sd=magnitude,n=len(y),
                        r2=1-float(np.sum((output-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                        mean_displacement_source_sd=float(d.mean()),p95_displacement_source_sd=float(np.quantile(d,.95))))
        if i%100==0:print('RADIUS_EVALUATION',backbone,track,i,flush=True)
    pd.DataFrame(scores).to_csv(dest/'performance.csv',index=False)
    pd.DataFrame(condition_stress).to_csv(dest/'stress_by_condition.csv',index=False)
    pooled=[]
    for (pattern,magnitude,method),parts in pool.items():
        d=np.concatenate(parts)
        pooled.append(dict(backbone=backbone,track=track,pattern=pattern,magnitude_source_sd=magnitude,method=method,
            n_query_conditions=len(d),mean_displacement_source_sd=float(d.mean()),
            p95_displacement_source_sd=float(np.quantile(d,.95)),max_displacement_source_sd=float(d.max())))
    pd.DataFrame(pooled).to_csv(dest/'stress_pooled.csv',index=False)
    write_json(dest/'evaluation_audit.json',dict(original_metrics_reproduced=True,
        same_stress_signs_and_subsets_as_v4=True,labels_used_for_scoring_only=True,seed=0))
    print('RADIUS_EVALUATION_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['select','evaluate'])
    p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    elif a.stage=='select':select(a.backbone,a.track)
    else:evaluate(a.backbone,a.track)
