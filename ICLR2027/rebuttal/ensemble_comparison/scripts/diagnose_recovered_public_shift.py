"""Run historical classification operators at predeclared neighboring betas."""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from prepare_data import RUN,atomic_json,protocol
from fit_combiners import fold_mapping
from recovered_public_shift import apply_public_shift_to_split
from run_tabicl import Predictor,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--datasets',type=int,default=12);args=p.parse_args();torch.set_num_threads(1)
    mapping,_=fold_mapping('classification')
    candidates=[json.loads(p.read_text()) for p in (RUN/'cache/five/classification').glob('*.json')]
    # Fixed outer-training subset for protocol diagnosis, not held-out fold 0.
    selected=sorted([m for m in candidates if mapping[int(m['data_id'])]!=0],key=lambda m:int(m['data_id']))[:args.datasets]
    predictor=Predictor('classification','cuda:0')
    for meta in selected[args.shard::args.shards]:
        with np.load(RUN/'cache/five/classification'/f"{meta['task_id']}.npz") as f:
            X,y,Q,t=f['X_source'],f['y_source'],f['X_target'],f['y_target']
        teacher=ExtraTreesClassifier(n_estimators=128,random_state=0,n_jobs=1).fit(X,y)
        pseudo=teacher.predict(Q);classes=len(np.unique(y))
        for labels in ['benchmark_labels','source_only_pseudo_labels','source_context']:
            for beta in [100,125,150]:
                output=RUN/'results/protocol_diagnostics'/f"public_{meta['data_id']}_{labels}_beta{beta}.json"
                if output.exists():continue
                rows=[]
                for family in ['shortcut','boundary','band','local','mixed']:
                    kind='aucband' if family=='band' else family
                    for rho in [20,50,80,100]:
                        condition=f'publicshift_d{meta["data_id"]}_{kind}_rho{rho}_beta{beta}'
                        if labels=='source_context':
                            context,details=apply_public_shift_to_split(condition,X,y,X,y,0)
                            prediction,_=predictor(context,y,Q,n_classes=classes)
                        else:
                            query,details=apply_public_shift_to_split(condition,X,y,Q,t if labels=='benchmark_labels' else pseudo,0)
                            prediction,_=predictor(X,y,query,n_classes=classes)
                        rows.append({'family':family,'rho':rho,'beta':beta/100,'labels':labels,
                                     **metrics(t,prediction,'classification',1.),'generator_details':details})
                atomic_json(output,meta|{'seed':0,'diagnostic_only':True,'original_git_operator':True,
                    'operator_location':'source_features' if labels=='source_context' else 'target_features','generator_labels':labels,
                    'inference_receives_target_labels':False,'outer_fold':mapping[int(meta['data_id'])],
                    'beta':beta/100,'rows':rows})
                print('DIAG_DONE',meta['data_id'],labels,beta,
                    float(np.mean([r['accuracy'] for r in rows])),float(np.mean([r['auc'] for r in rows])),flush=True)


if __name__=='__main__':main()
