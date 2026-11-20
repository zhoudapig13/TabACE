"""Bounded final-update influence; retain the frozen V3 evidence head.

Predeclared last iteration: replace an abrupt, distance-based trust adjustment
with smooth saturation of the proposed output update. Both cap controls are
reported; no held-out-label selection between them is performed.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from replay_diagnostics import OUT,ROOT,cells,write_json
from calibrated_regression_repairs import components

DEST=OUT/'smooth_update_v6'
MAGNITUDES=[.25,.5,1.,2.,3.,5.,8.,10.]

def prepare():
    path=DEST/'frozen_plan.json'
    if path.exists():raise RuntimeError('Do not overwrite evaluated plan')
    hashes=[]
    for b in ['tabicl','tabpfn']:
        for t in ['five','seven_reimplemented']:
            for fold in range(5):
                p=OUT/'algorithm_repairs_v3'/b/t/f'fold_{fold}.json'
                hashes.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_json(path,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        purpose='Suppress moderate as well as extreme common errors with smooth saturation of the proposed update',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_gate_hashes=hashes,
        formula='output=Base+c*source_SD*tanh(alpha_evidence*(aggregate-Base)/(c*source_SD))',
        fixed_caps_source_sd=[3.,6.],priority_control=6.,new_parameter_fitting=False,
        evidence_head='Frozen, source-trained V3 bounded gain',bias_magnitudes=MAGNITUDES,
        patterns=['independent_random_signs','shared_bias_3of5','shared_bias_5of5'],
        same_label_free_stress_as_v4_v5=True,all_controls_reported=True,
        held_out_labels_used_for_cap_selection=False,benchmark_unchanged=True,backbone_calls=0,markdown_written=False,
        evaluation_status='Final bounded development iteration in this task; previously inspected reconstructed evaluation, not a new independent test'))

def predict(b,k,t,beta):
    ag,a,f=components(b,k,t)
    proposed=np.clip(a*(1+.5*np.tanh(f@beta)),0,1)*(ag-b)
    out=dict(Original=b+a*(ag-b))
    for cap in [3.,6.]:
        out[f'Smooth_cap{int(cap)}']=b+cap*np.tanh(proposed/cap)
        assert np.isfinite(out[f'Smooth_cap{int(cap)}']).all()
        assert np.max(np.abs(out[f'Smooth_cap{int(cap)}']-b))<=cap+1e-8
    return out

def run(backbone,track):
    heads={fold:json.loads((OUT/'algorithm_repairs_v3'/backbone/track/f'fold_{fold}.json').read_text()) for fold in range(5)}
    scores=[];stress=[];pool=defaultdict(list)
    for i,m,rawb,rawk,y,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T;beta=np.asarray(heads[fold]['parameters'])
        initial=predict(b,k,t,beta)
        scale=max(float(y.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        original=initial['Original']*ss+sm
        assert abs(float(np.sqrt(np.mean((original-y)**2))/scale)-ref['nrmse'])<1e-8*max(1,ref['nrmse'])
        for method,p in initial.items():
            output=p*ss+sm
            scores.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                outer_fold=fold,method=method,n=len(y),r2=1-float(np.sum((output-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                nrmse=float(np.sqrt(np.mean((output-y)**2))/scale)))
        rng=np.random.default_rng(0);sign=rng.choice([-1.,1.],len(b));independent=rng.choice([-1.,1.],k.shape)
        subset=np.argsort(rng.random(k.shape),axis=1)[:,:3];mask=np.zeros_like(k)
        np.put_along_axis(mask,subset,sign[:,None],axis=1)
        patterns=dict(independent_random_signs=independent,shared_bias_3of5=mask,shared_bias_5of5=sign[:,None]*np.ones_like(k))
        for magnitude in MAGNITUDES:
            for pattern,shift in patterns.items():
                pred=predict(b,k+magnitude*shift,t,beta)
                for method,p in pred.items():
                    d=np.abs(p-initial[method]);pool[(pattern,magnitude,method)].append(d)
                    output=p*ss+sm
                    stress.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],
                        method=method,pattern=pattern,magnitude_source_sd=magnitude,n=len(y),
                        r2=1-float(np.sum((output-y)**2))/max(float(np.sum((y-y.mean())**2)),1e-12),
                        mean_displacement_source_sd=float(d.mean()),p95_displacement_source_sd=float(np.quantile(d,.95))))
        if i%100==0:print('SMOOTH_EVALUATION',backbone,track,i,flush=True)
    dest=DEST/backbone/track;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(scores).to_csv(dest/'performance.csv',index=False)
    pd.DataFrame(stress).to_csv(dest/'stress_by_condition.csv',index=False)
    pooled=[]
    for (pattern,magnitude,method),parts in pool.items():
        d=np.concatenate(parts)
        pooled.append(dict(backbone=backbone,track=track,pattern=pattern,magnitude_source_sd=magnitude,method=method,
            n_query_conditions=len(d),mean_displacement_source_sd=float(d.mean()),
            p95_displacement_source_sd=float(np.quantile(d,.95)),max_displacement_source_sd=float(d.max())))
    pd.DataFrame(pooled).to_csv(dest/'stress_pooled.csv',index=False)
    write_json(dest/'evaluation_audit.json',dict(original_frozen_metrics_reproduced=True,
        no_new_fitting=True,parent_source_evaluation_ids_disjoint=all(set(h['source_data_ids']).isdisjoint(h['evaluation_data_ids']) for h in heads.values()),
        same_stress_as_v4_v5=True,labels_used_for_scoring_only=True))
    print('SMOOTH_EVALUATION_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.backbone,a.track)
