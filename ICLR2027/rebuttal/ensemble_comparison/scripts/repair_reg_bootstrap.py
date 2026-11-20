"""Recompute only the invalid regression resampling component; retain other runs."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from prepare_data import RUN,protocol,atomic_json
from run_tabicl import Predictor,metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(1);torch.cuda.set_device(a.gpu)
    predictor=Predictor('regression',f'cuda:{a.gpu}')
    files=sorted((RUN/'results/tabicl/five/regression').rglob('*.npz'))[a.shard::a.shards]
    for path in files:
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta.get('bootstrap_revision',1)>=2:continue
        source=RUN/'cache/five/regression'/f"{meta['task_id']}.npz"
        with np.load(source) as d:
            change=protocol.manipulate_source_context('regression',d['X_source'],d['y_source'],
                family=meta['family'],rho=meta['rho']/100,seed=1729+meta['rho'],shift_scale=10.,label_scale=1.)
            X,y,Q=change.X_context,change.y_context,d['X_target']
            bank=[];iqr=[];start=time.perf_counter()
            for i in range(6):
                ids=np.random.default_rng(1009*(i+1)).choice(len(y),len(y),replace=True)
                pred,uncertainty=predictor(X[ids],y[ids],Q,quantiles=True)
                bank.append(pred);iqr.append(uncertainty)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        bank=np.stack(bank);iqr=np.stack(iqr)
        with np.load(path) as old:arrays={k:old[k] for k in old.files}
        mean=bank.mean(0);selected=bank[int(np.argmin(iqr.mean(1)))]
        arrays['bootstrap_bank']=bank;arrays['bootstrap_iqr']=iqr
        arrays['prediction__bootstrap_mean_6']=mean
        arrays['prediction__oligoicp_iqr_6_keep1_adapted']=selected
        for name,values in [('bootstrap_mean_6',mean),('oligoicp_iqr_6_keep1_adapted',selected)]:
            meta['methods'][name]={**metrics(arrays['y_target'],values,'regression',meta['scale']),'seconds':elapsed}
        backup=RUN/'results/invalidated_bootstrap_v1/raw_regression'/path.parent.name
        backup.mkdir(parents=True,exist_ok=True)
        if not (backup/path.name).exists():
            shutil.copy2(path,backup/path.name);shutil.copy2(path.with_suffix('.json'),backup/path.with_suffix('.json').name)
        meta['bootstrap_revision']=2
        meta['bootstrap_repair']='Regression now samples rows uniformly with replacement; classification-only class preservation removed. Native and other predictions reused.'
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
        atomic_json(path.with_suffix('.json'),meta)
        print('REPAIRED',meta['task_id'],meta['family'],meta['rho'],flush=True)

if __name__=='__main__':main()
