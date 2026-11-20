"""Compute fixed-input Base, ensemble and R5 banks for the additional feature variant."""
import argparse, json, logging, shutil, traceback, warnings
from functools import partial
import numpy as np
import torch
from prepare_data import RUN, atomic_json, protocol
from shared_reference import Predictor, PFNPredictor, run_one, specifications, load_condition
from specific_reference import specific_contexts


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],default='tabicl')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    if a.backbone=='tabicl':
        from run_tabicl import run_cell
        predictor=Predictor('classification','cuda:0')
    else:
        from run_tabpfn import cell
        predictor=PFNPredictor('classification')
    failures=0
    for meta in specifications('five_feature_calibrated')[a.shard::a.shards]:
        conditions=[('original',0)]+[(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        for family,rho in conditions:
            try:
                data,info=load_condition(meta,family,rho)
                name=f'{family}_{rho}.npz'
                out=RUN/'results'/a.backbone/info['track']/'classification'/str(meta['task_id'])/name
                if not out.with_suffix('.json').exists():
                    if rho==0:
                        old=RUN/'results'/a.backbone/'five/classification'/str(meta['task_id'])/name
                        out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,out)
                        atomic_json(out.with_suffix('.json'),json.loads(old.with_suffix('.json').read_text())|info|
                            {'unchanged_clean_predictions_reused_from':str(old.relative_to(RUN))})
                    elif a.backbone=='tabicl':run_cell(predictor,data,info,family,rho,out)
                    else:cell('classification',data,info,family,rho,out,'cuda:0',True)
                shared=RUN/'results/reference'/a.backbone/info['track']/'classification'/str(meta['task_id'])/name
                if not shared.with_suffix('.json').exists():run_one(predictor,data,info,shared,a.backbone,include_native=False)
                specific=RUN/'results/reference_specific'/a.backbone/info['track']/'classification'/str(meta['task_id'])/name
                if rho:
                    if not specific.with_suffix('.json').exists():
                        factory=partial(specific_contexts,family=family,track=info['track'])
                        run_one(predictor,data,info,specific,a.backbone,context_factory=factory,variant='specific',include_native=False)
                else:
                    for clean_family in protocol.FAMILIES:
                        clean=specific.with_name(f'{clean_family}_0.npz')
                        if clean.with_suffix('.json').exists():continue
                        factory=partial(specific_contexts,family=clean_family,track=info['track'])
                        run_one(predictor,data,info|{'family':clean_family},clean,a.backbone,context_factory=factory,variant='specific',include_native=False)
                print('FEATURE_VARIANT_DONE',a.backbone,meta['task_id'],family,rho,flush=True)
            except Exception as exc:
                failures+=1;atomic_json(RUN/'results/inference_failures'/f"feature_variant_{a.backbone}_{meta['task_id']}_{family}_{rho}.json",
                    meta|{'error':repr(exc),'traceback':traceback.format_exc()})
                print('FEATURE_VARIANT_FAILED',a.backbone,meta['task_id'],family,rho,repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
