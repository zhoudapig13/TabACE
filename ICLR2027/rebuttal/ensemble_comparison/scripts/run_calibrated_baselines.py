"""Run all authorized baselines on a frozen calibrated input bank."""
import argparse,json,logging,shutil,traceback,warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from seven_shifts import FAMILIES


def main():
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['five_calibrated','seven_calibrated','seven_calibrated_v2'],required=True)
    p.add_argument('--backbone',choices=['tabicl','tabpfn','tabm'],default='tabicl');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    if args.backbone=='tabicl':
        from run_tabicl import Predictor,run_cell
        predictor=Predictor('classification' if args.track.startswith('five') else 'regression','cuda:0')
    elif args.backbone=='tabpfn':from run_tabpfn import cell
    else:from run_tabm import run_one
    families=protocol.FAMILIES if args.track.startswith('five') else FAMILIES
    conditions=[('original' if args.track.startswith('five') else 'covariate',0)]+[(f,r) for f in families for r in [20,50,80,100]]
    jobs=[(m,f,r) for m in specifications(args.track) for f,r in conditions][args.shard::args.shards]
    failures=0
    for meta,family,rho in jobs:
        out=RUN/'results'/args.backbone/args.track/meta['task']/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():continue
        try:
            data,info=load_condition(meta,family,rho)
            if args.track=='seven_calibrated_v2':
                info['native_ensemble_sizes']=[6]
                # Native-6 is represented on TabICL; PHE remains on TabPFN.
                info['include_native_ensemble_baselines']=False
            if rho==0:
                old_track='five' if args.track.startswith('five') else 'seven_reimplemented'
                old=RUN/'results'/args.backbone/old_track/meta['task']/str(meta['task_id'])/out.name
                if old.exists() and old.with_suffix('.json').exists():
                    with np.load(old) as f:assert np.array_equal(f['y_target'],data['y_target'])
                    out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,out)
                    previous=json.loads(old.with_suffix('.json').read_text())
                    atomic_json(out.with_suffix('.json'),previous|info|{'unchanged_clean_predictions_reused_from':str(old.relative_to(RUN))})
                    continue
            print('START_CALIBRATED',args.backbone,args.track,meta['task_id'],family,rho,flush=True)
            if args.backbone=='tabicl':
                pre=RUN/'results/calibrated_base'/args.track/meta['task']/str(meta['task_id'])/out.name
                if pre.exists():
                    with np.load(pre) as f:cached=f['prediction'].copy()
                    info['Base_prediction_reused_from']=str(pre.relative_to(RUN))
                    class FirstCached:
                        def __init__(self):self.used=False
                        def __call__(self,*a,**kw):
                            if not self.used:self.used=True;return cached,None
                            return predictor(*a,**kw)
                    run_cell(FirstCached(),data,info,family,rho,out)
                else:run_cell(predictor,data,info,family,rho,out)
            elif args.backbone=='tabpfn':cell(meta['task'],data,info,family,rho,out,'cuda:0',True)
            else:run_one(data,info,out)
            print('DONE_CALIBRATED',args.backbone,args.track,meta['task_id'],family,rho,flush=True)
        except Exception as exc:
            failures+=1;atomic_json(RUN/'results/inference_failures'/f"calibrated_{args.backbone}_{meta['task_id']}_{family}_{rho}.json",
                info|{'error':repr(exc),'traceback':traceback.format_exc()} if 'info' in locals() else meta|{'error':repr(exc)})
            print('FAILED_CALIBRATED',args.backbone,meta['task_id'],family,rho,repr(exc),flush=True);torch.cuda.empty_cache()
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
