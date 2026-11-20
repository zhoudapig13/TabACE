"""Calibrate seven-shift regression amplitude on independent source-held-out rows."""
import argparse,json,logging,warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json
from run_tabicl import Predictor,metrics
from calibrated_seven_protocol import candidates,materialize
from seven_shifts import FAMILIES


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    m=json.loads((RUN/'configs/seven_severity_calibration_manifest.json').read_text())
    datasets=[r for r in m['datasets'] if r['calibration_status']=='READY']
    configs=[{'name':'clean','amplitude_multiplier':0.}]+candidates()
    jobs=[(d,c) for c in configs for d in datasets][args.shard::args.shards]
    predictor=Predictor('regression','cuda:0');failures=0
    for meta,config in jobs:
        out=RUN/'results/severity_calibration/seven_regression'/config['name']/f"{meta['task_id']}.json"
        if out.exists():continue
        with np.load(RUN/'cache/severity_calibration/seven_regression'/f"{meta['task_id']}.npz") as f:
            clean=tuple(f[k] for k in ['X_history','y_history','X_reference','y_reference','X_source','y_source','X_target','y_target'])
        conditions=[('covariate',0)] if config['name']=='clean' else [(f,r) for f in FAMILIES for r in [20,50,80,100]]
        rows=[];bank=[]
        try:
            for family,rho in conditions:
                data,info=materialize(meta,family,rho,clean,config['amplitude_multiplier'])
                X,y,Q,t=[data[k] for k in ['X_source','y_source','X_target','y_target']]
                prediction,_=predictor(X,y,Q)
                rows.append({'family':family,'rho':rho,**metrics(t,prediction,'regression',max(float(t.std()),1e-12))})
                bank.append(prediction)
            out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),predictions=np.stack(bank),y_target=t)
            atomic_json(out,{'status':'COMPLETE','task_id':meta['task_id'],'data_id':meta['data_id'],'configuration':config,
                'rows':rows,'original_test_labels_used':False,'TabACE_results_used_for_selection':False})
            print('SEVEN_SEVERITY_DONE',config['name'],meta['task_id'],np.mean([r['r2'] for r in rows]),np.mean([r['nrmse'] for r in rows]),flush=True)
        except Exception as exc:failures+=1;print('SEVEN_SEVERITY_FAILED',meta['task_id'],repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
