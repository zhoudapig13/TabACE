"""User-authorized benchmark-development calibration; never labelled test-only."""
import argparse,json,logging,warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json
from run_tabicl import Predictor,metrics
from seven_shifts import registry,regression_clean,FAMILIES
from calibrated_seven_protocol import materialize

MULTIPLIERS=[.35,.4,.45,.5,.55,.59,0.,.1,.75,1.25,1.5]


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    jobs=[(m,s) for s in MULTIPLIERS for m in registry() if m['task']=='regression'][a.shard::a.shards]
    model=Predictor('regression','cuda:0');failures=0
    for meta,s in jobs:
        out=RUN/'results/severity_calibration/seven_benchmark_development'/f'amplitude_{s:g}'/f"{meta['task_id']}.json"
        if out.exists():continue
        try:
            clean=regression_clean(meta);rows=[];bank=[]
            for family in FAMILIES:
                for rho in [20,50,80,100]:
                    data,info=materialize(meta,family,rho,clean,s)
                    cached=RUN/'results/calibrated_base/seven_calibrated/regression'/str(meta['task_id'])/f'{family}_{rho}.npz'
                    if s==.59 and cached.exists():
                        with np.load(cached) as f:prediction=f['prediction'].copy();assert np.array_equal(f['y_target'],data['y_target'])
                    else:prediction,_=model(data['X_source'],data['y_source'],data['X_target'])
                    rows.append({'family':family,'rho':rho,**metrics(data['y_target'],prediction,'regression',max(float(data['y_target'].std()),1e-12))})
                    bank.append(prediction)
            out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),predictions=np.stack(bank),y_target=data['y_target'])
            atomic_json(out,{'status':'COMPLETE','task_id':meta['task_id'],'data_id':meta['data_id'],'amplitude_multiplier':s,
                'rows':rows,'benchmark_labels_used_for_severity_selection':True,'results_role':'benchmark development, not independent test',
                'TabACE_results_used_for_selection':False})
            print('BENCHMARK_SEVERITY',s,meta['task_id'],np.mean([r['r2'] for r in rows]),np.mean([r['nrmse'] for r in rows]),flush=True)
        except Exception as exc:failures+=1;print('BENCHMARK_SEVERITY_FAILED',s,meta['task_id'],repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
