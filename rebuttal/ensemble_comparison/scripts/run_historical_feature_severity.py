"""Historical feature-only interventions; source labels remain unchanged.

This is an additional declared protocol variant. All candidate outcomes are
retained, and severity selection sees only Base on disjoint calibration queries.
"""
import argparse,json,logging,warnings
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from prepare_data import RUN,atomic_json
from recovered_public_shift import apply_public_shift_to_split
from scaled_historical_feature_protocol import apply_feature_scaled_shift
from run_tabicl import Predictor,metrics

BETAS=[2.,4.,6.,8.,12.,20.]
CONFIGURATIONS=[{'beta':b,'feature_multiplier':1} for b in BETAS]+[
    {'beta':b,'feature_multiplier':m} for m in [2,4,8] for b in [2.,4.]]+[
    {'beta':b,'feature_multiplier':8} for b in [5.,6.]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    manifest=json.loads((RUN/'configs/severity_calibration_manifest.json').read_text())
    jobs=[(m,c) for c in CONFIGURATIONS for m in manifest['datasets'] if m['calibration_status']=='READY'][a.shard::a.shards]
    predictor=Predictor('classification','cuda:0');failures=0
    for meta,configuration in jobs:
        beta=configuration['beta'];multiplier=configuration['feature_multiplier']
        name=f'beta_{beta:g}' if multiplier==1 else f'beta_{beta:g}_features{multiplier}'
        out=RUN/'results/severity_calibration/historical_feature'/name/f"{meta['task_id']}.json"
        if out.exists():continue
        try:
            with np.load(RUN/'cache/severity_calibration/classification'/f"{meta['task_id']}.npz") as f:
                X,y,Q,t=f['X_source'],f['y_source'],f['X_target'],f['y_target']
            teacher=ExtraTreesClassifier(n_estimators=128,random_state=0,n_jobs=1).fit(X,y);pseudo=teacher.predict(Q)
            rows=[];bank=[]
            for family in ['shortcut','boundary','band','local','mixed']:
                for rho in [20,50,80,100]:
                    kind='aucband' if family=='band' else family
                    name=f'publicshift_d{meta["data_id"]}_{kind}_rho{rho}_beta{int(100*beta)}'
                    shifted,details=apply_feature_scaled_shift(name,X,y,Q,pseudo,0,feature_multiplier=multiplier)
                    prediction,_=predictor(X,y,shifted,n_classes=meta['n_classes'])
                    rows.append({'family':family,'rho':rho,**metrics(t,prediction,'classification',1.),'operator_details':details})
                    bank.append(prediction)
            out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),predictions=np.stack(bank),y_target=t)
            atomic_json(out,{'status':'COMPLETE','task_id':meta['task_id'],'data_id':meta['data_id'],'beta':beta,'feature_multiplier':multiplier,'rows':rows,
                'source_labels_changed':False,'original_test_labels_used':False,'generator_teacher':'ExtraTrees128 fitted to source only',
                'operator':'recovered historical shift with declared top-feature-coverage multiplier','variant':'historical feature-only stress calibration'})
            print('HISTORICAL_FEATURE',beta,meta['task_id'],np.mean([r['accuracy'] for r in rows]),np.mean([r['auc'] for r in rows]),flush=True)
        except Exception as exc:failures+=1;print('HISTORICAL_FEATURE_FAILED',beta,meta['task_id'],repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
