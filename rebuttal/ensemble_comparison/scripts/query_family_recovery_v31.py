"""Evaluate exactly one source-frozen continuous restoration protocol."""
import argparse
import json
import os
import time
import traceback
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(k,'1')
import numpy as np
import pandas as pd
import family_recovery_v31 as family
from prepare_data import atomic_json,protocol


def worker(shard,shards):
    import warnings,torch
    from run_tabicl import Predictor,metrics
    source=family.source;v=source.v;root=family.OUT
    settings=json.loads((root/'parameters.json').read_text());parameter_hash=family.sha(root/'parameters.json')
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);warnings.filterwarnings('ignore')
    predictor=Predictor('regression','cuda:0');failures=0
    for block,meta in v.prior.jobs()[shard::shards]:
        try:
            pool=v.prior.original.prepare('regression',block,meta)
            for name,rho in v.grid(block):
                env=v.environment(pool,meta,block,name,rho,np.arange(len(pool['y'])))
                core,info=v.prior.original.bank('regression',block,meta,name,rho)
                fraction=settings['fractions'][block][name]
                for budget in v.BUDGETS:
                    path=root/'query'/str(budget)/block/str(meta['task_id'])/f'{name}_{rho}.npz'
                    if path.exists() and path.with_suffix('.json').exists():continue
                    X,y=source.context(env,fraction,budget);data=core|dict(X_source=X,y_source=y)
                    assert np.array_equal(X[:256],core['X_source']) and np.array_equal(y[:256],core['y_source'])
                    started=time.monotonic()
                    if budget==256:
                        old=v.prior.original.frozen.OUT/'input_banks/regression'/block/str(meta['task_id'])/path.name
                        with np.load(old) as z:prediction=z['Base_prediction']
                    else:prediction,_=predictor(X,y,data['X_target'],members=1,random_state=0)
                    assert np.isfinite(prediction).all()
                    v.prior.original.npwrite(path,prediction=prediction,y_target=data['y_target'])
                    atomic_json(path.with_suffix('.json'),dict(task='regression',block=block,task_id=meta['task_id'],
                        dataset=meta['dataset'],family=name,rho=rho,requested_context=budget,actual_context=len(y),
                        restoration_fraction=fraction,scale=info['scale'],metrics=metrics(data['y_target'],prediction,'regression',info['scale']),
                        input_hashes={k:protocol.array_sha256(a) for k,a in data.items()},
                        original_corrupted256_core_preserved=True,seed=0,n_estimators=1,checkpoint_finetuned=False,
                        extra_pre_corruption_TRAIN_information=True,parameters_frozen_before_query=True,
                        settings_sha256=parameter_hash,semisynthetic=block=='five',development_only=True,
                        original_paper_replication=False,elapsed_seconds=time.monotonic()-started))
            print('FAMILY_QUERY_DATASET_COMPLETE',block,meta['dataset'],flush=True)
        except Exception:
            failures+=1;error=traceback.format_exc();print(error,flush=True)
            atomic_json(root/'errors'/f'query_{block}_{meta["task_id"]}.json',dict(error=error))
    if failures:raise SystemExit(1)


def report():
    rows=[];checks=[]
    for block,expected,datasets in [('five',700,35),('seven',924,33)]:
        for budget in family.source.v.BUDGETS:
            paths=[p for p in (family.OUT/'query'/str(budget)/block).glob('*/*.json') if '.tmp.' not in p.name]
            if len(paths)!=expected:continue
            records=[json.loads(p.read_text()) for p in paths]
            rows.append(dict(block=block,context=budget,datasets=datasets,conditions=expected,
                r2=float(np.mean([r['metrics']['r2'] for r in records])),
                nrmse=float(np.mean([r['metrics']['nrmse'] for r in records])),
                actual_rows_mean=float(np.mean([r['actual_context'] for r in records]))))
        values=[r for r in rows if r['block']==block]
        monotone=len(values)==4 and bool(np.all(np.diff([r['r2'] for r in values])>0) and np.all(np.diff([r['nrmse'] for r in values])<=0))
        gain=values[-1]['r2']-values[0]['r2'] if len(values)==4 else None
        checks.append(dict(block=block,complete_budgets=len(values),query_monotone=monotone,
            final_R2_gain=gain,small_gain=bool(monotone and .005<=gain<=.08)))
    pd.DataFrame(rows).to_csv(family.OUT/'query_averages.csv',index=False)
    atomic_json(family.OUT/'query_status.json',dict(status='SMALL_MONOTONE_QUERY_CURVES' if all(r['small_gain'] for r in checks) else 'TARGET_NOT_MET_OR_INCOMPLETE',
        checks=checks,does_not_automatically_replace_main_protocol=True,query_scores_not_used_to_select_parameters=True,
        development_only=True,original_paper_replication=False))
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print(json.dumps(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    worker(a.shard,a.shards) if a.phase=='worker' else report()
