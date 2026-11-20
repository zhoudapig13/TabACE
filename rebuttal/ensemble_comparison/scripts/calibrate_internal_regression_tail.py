"""Uniform source-tail coefficient; development calibration, never paper replication."""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
from prepare_data import RUN,atomic_json,protocol
from run_tabicl import Predictor,metrics
from internal_regression_benchmark import OUT,manifest,five_condition

CONFIG=RUN/'configs/internal_regression_five_tail_calibration.json'


def run(shard,shards,limit,config_path=CONFIG):
    import torch
    torch.set_num_threads(1);warnings.filterwarnings('ignore');predictor=Predictor('regression','cuda:0')
    config_path=config_path.resolve()
    settings=json.loads(config_path.read_text())['candidates'];cache={}
    jobs=[(m,f,r) for m in manifest() for f,r in [('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]][shard::shards]
    if limit:jobs=jobs[:limit]
    for m,family,rho in jobs:
        known={}
        baseline=OUT/'five_candidates/shift8_label1'/str(m['task_id'])/f'{family}_{rho}.npz'
        if baseline.with_suffix('.json').exists():
            old=json.loads(baseline.with_suffix('.json').read_text());known[json.dumps(old['input_hashes'],sort_keys=True)]=baseline
        for setting in settings:
            out=OUT/'five_candidates'/setting['name']/str(m['task_id'])/f'{family}_{rho}.npz'
            if out.exists() and out.with_suffix('.json').exists():continue
            data,info=five_condition(m,family,rho,setting['parameters'],cache)
            hashes={k:protocol.array_sha256(v) for k,v in data.items()};key=json.dumps(hashes,sort_keys=True)
            if key in known:
                with np.load(known[key]) as z:p=z['prediction'];assert np.array_equal(data['y_target'],z['y_target'])
                reused=str(known[key].relative_to(RUN))
            else:p,_=predictor(data['X_source'],data['y_source'],data['X_target']);reused=None
            observed=metrics(data['y_target'],p,'regression',info['scale'])
            out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=p,y_target=data['y_target']);tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','metrics':observed,'input_hashes':hashes,
                'prediction_reused_from':reused,'uniform_noise_scale_formula':True,'internal_only':True,
                'calibration_plan':str(config_path.relative_to(RUN))})
            known[key]=out
        print('TAIL_GRID_DONE',m['dataset'],family,rho,flush=True)
    print('WORKER_FINISHED',shard,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--config',type=str)
    a=p.parse_args();run(a.shard,a.shards,a.limit,Path(a.config) if a.config else CONFIG)
