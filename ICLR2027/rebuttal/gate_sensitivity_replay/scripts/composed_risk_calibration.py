"""Fixed composition of source-trained residual calibration and harm-risk gate."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
from replay_diagnostics import OUT,cells,write_json
from excess_risk_gate import features
import gate_preserving_residual as residual

DEST=OUT/'composed_risk_v11'

def prepare():
    p=DEST/'frozen_plan.json'
    if p.exists():raise RuntimeError('Do not overwrite evaluated composition')
    write_json(p,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),seed=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='Keep actual excess-harm recovery of the risk gate and natural-error gains of source-calibrated residual fusion',
        formula='output=risk_strength*Base + (1-risk_strength)*bounded_residual_prediction; risk_strength=selected_strength*I(source_calibrated_risk_score>=selected_threshold)',
        components=['gate_preserving_v8','excess_risk_v10'],new_parameter_fitting=False,
        composition_fixed_before_evaluation=True,no_per_test_case_variant_selection=True,
        residual_cap_source_sd=6.,source_calibration_only=True,benchmark_unchanged=True,
        backbone_calls=0,markdown_written=False,evaluation_status='Development composition of previously evaluated source-trained components; not independent confirmation'))

def run(backbone,track):
    rh={f:json.loads((OUT/'gate_preserving_v8'/backbone/track/f'fold_{f}.json').read_text()) for f in range(5)}
    gh={f:joblib.load(OUT/'excess_risk_v10'/backbone/track/f'fold_{f}.joblib') for f in range(5)}
    rows=[];budgets=[];pressure=[]
    for i,m,rawb,rawk,rawy,sm,ss,t,fold,ref,path in cells(backbone,track,'regression'):
        b=(rawb-sm)/ss;k=((rawk-sm)/ss).T;y=(rawy-sm)/ss;h=rh[fold];riskhead=gh[fold]
        beta=np.asarray(h['evidence_parameters']);coef=np.asarray(h['residual_coefficients']) if h['residual_coefficients'] is not None else None
        def predict(matrix):
            x,old=features(b,matrix,t,riskhead['moment']);choice=riskhead['choice']
            score=riskhead['model'].predict_proba(x)[:,1]
            rate=choice['strength']*(score>=choice['threshold']) if choice else np.zeros(len(y))
            calibrated=residual.core.predict(b,matrix,t,beta,coef,h['blend'])
            composed=rate*b+(1-rate)*calibrated
            assert np.max(np.abs(composed-b))<=6.+1e-7
            return dict(Original=old,Composite=composed)
        values=predict(k);scale=max(float(rawy.std()),1e-12) if track.startswith('seven') else float(m['scale'])
        for method,p in values.items():
            output=p*ss+sm
            rows.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],method=method,n=len(y),
                r2=1-float(np.sum((output-rawy)**2))/max(float(np.sum((rawy-rawy.mean())**2)),1e-12),nrmse=float(np.sqrt(np.mean((output-rawy)**2))/scale)))
        assert abs(float(np.sqrt(np.mean((values['Original']*ss+sm-rawy)**2))/scale)-ref['nrmse'])<1e-8*max(1,ref['nrmse'])
        if m['rho']==0:continue
        sd=max(float(y.std()),1e-12);base=((b-y)/sd)**2;old=((values['Original']-y)/sd)**2;e=(k-y[:,None])/sd
        masks={'all':np.ones(len(y),bool)}
        for threshold in [1,3]:
            joint=(e>threshold).all(1)|(e<-threshold).all(1);masks[f'joint{threshold}']=joint;masks[f'harmful_joint{threshold}']=joint&(old>base+.01)
        for method,p in values.items():
            se=((p-y)/sd)**2
            for name,mask in masks.items():
                budgets.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],method=method,cohort=name,
                    n_total=len(y),n_cohort=int(mask.sum()),sum_normalized_loss=float(se[mask].sum()),sum_excess_vs_Base=float(np.maximum(se-base,0.)[mask].sum()),
                    no_longer_materially_worse_than_Base=int((mask&(se<=base+.01)).sum()),
                    improved_vs_original=int((mask&(se<old-1e-10)).sum()),worsened_vs_original=int((mask&(se>old+1e-10)).sum())))
        rng=np.random.default_rng(0);sign=rng.choice([-1.,1.],len(y))
        for magnitude in [1.,3.,5.,10.]:
            shifted=k+magnitude*sign[:,None];pred=predict(shifted)
            for method,p in pred.items():
                se=((p-y)/sd)**2
                pressure.append(dict(backbone=backbone,track=track,data_id=int(m['data_id']),family=m['family'],rho=m['rho'],method=method,magnitude_source_sd=magnitude,n=len(y),
                    normalized_mse=float(se.mean()),mean_output_displacement_source_sd=float(np.abs(p-values[method]).mean())))
    dest=DEST/backbone/track;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'performance.csv',index=False);pd.DataFrame(budgets).to_csv(dest/'condition_budgets.csv',index=False)
    pd.DataFrame(pressure).to_csv(dest/'common_bias_stress.csv',index=False)
    write_json(dest/'audit.json',dict(original_metrics_reproduced=True,no_new_parameter_fitting=True,
        all_fixed_composition_results_reported=True,query_labels_used_for_scoring_only=True))
    print('COMPOSITE_COMPLETE',backbone,track,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--backbone');p.add_argument('--track');a=p.parse_args()
    if a.prepare:prepare()
    else:run(a.backbone,a.track)
