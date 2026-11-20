"""Finish the one legacy constant-feature corner with a source-prior fallback."""
import argparse
from functools import partial
import torch
from shared_reference import RUN,specifications,load_condition,run_one,Predictor,PFNPredictor
from specific_reference import specific_contexts

p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True);a=p.parse_args()
torch.set_num_threads(1)
m=next(m for m in specifications('five') if m['task']=='classification' and m['task_id']==10101)
d,info=load_condition(m,'shortcut',100)
out=RUN/'results/reference_specific'/a.backbone/'five/classification/10101/shortcut_100.npz'
if not out.with_suffix('.json').exists():
    pred=Predictor('classification','cuda:0') if a.backbone=='tabicl' else PFNPredictor('classification')
    factory=partial(specific_contexts,family='shortcut',track='five',history=None,reference=None)
    run_one(pred,d,info,out,a.backbone,context_factory=factory,variant='specific',include_native=False)
print('REPAIRED',a.backbone,out,flush=True)
