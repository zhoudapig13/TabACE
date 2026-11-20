"""Family-informed R=5 Specific; strengths never enter learned policy inputs."""
import argparse
import json
import time
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from fit_shared_reference import load_cells,fit,loss_on,predict
from run_tabicl import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],default='tabicl')
    p.add_argument('--track',choices=['five','five_corrected','five_calibrated','seven_calibrated','seven_reimplemented'],default='five');p.add_argument('--wait',action='store_true');args=p.parse_args()
    torch.set_num_threads(1)
    if args.track.startswith('five'):families=protocol.FAMILIES;dimensions=[('classification',72*25),('regression',35*25)]
    else:
        from seven_shifts import FAMILIES
        families=FAMILIES;dimensions=[('classification',35*35),('regression',33*35)]
    if args.track=='five_corrected':dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track=='five_calibrated':dimensions=[('classification',1800 if 'specific_reference' in __file__ else 1512)]
    if args.track=='seven_calibrated':dimensions=[('regression',1155 if 'specific_reference' in __file__ else 957)]
    for task,expected in dimensions:
        root=RUN/'results/reference_specific'/args.backbone/args.track/task
        while len(list(root.rglob('*.json')))<expected:
            if not args.wait:raise RuntimeError('incomplete Specific bank')
            time.sleep(10)
        all_cells,source=load_cells(args.backbone,args.track,task,variant='specific');assert len(all_cells)==expected
        for family in families:
            cells=[c for c in all_cells if c['meta']['family']==family]
            for fold in range(5):
                out=RUN/'results/combiners'/args.track/task/f'fold_{fold}_tabace_specific_{args.backbone}_{family}_reimplemented.json'
                if out.exists():continue
                train=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
                train_ids={c['meta']['data_id'] for c in train};test_ids={c['meta']['data_id'] for c in test};assert train_ids.isdisjoint(test_ids)
                inner=min(c['fold'] for c in train);inner_train=[c for c in train if c['fold']!=inner];inner_val=[c for c in train if c['fold']==inner]
                scores=[];inner_optimization=[]
                for reg in [.0001,.001,.01]:
                    parameters,optimization=fit(inner_train,task,reg)
                    scores.append((loss_on(inner_val,parameters,task),reg));inner_optimization.append(optimization)
                best=min(scores)[1];parameters,optimization=fit(train,task,best);rows=[]
                for c in test:
                    prediction=predict(c,parameters,task)
                    scale=max(float(c['y'].std()),1e-12) if task=='regression' and args.track.startswith('seven') else c['meta']['scale']
                    row=c['meta']|{'method':'TabACE_Specific_R5_reimplemented','backbone':args.backbone,'outer_fold':fold,
                        'family_identifier_visible':True,'strength_used_for_inference':False,
                        'seconds':c['meta']['base_seconds']+sum(c['meta']['context_seconds'])+c['meta']['construction_seconds']}
                    rows.append(row|metrics(c['y'],prediction,task,scale))
                    target=RUN/'results/reference_specific_predictions'/args.backbone/args.track/task/str(c['meta']['task_id'])/c['path'].name
                    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp,prediction=prediction,y_target=c['y']);tmp.replace(target)
                atomic_json(out,{'status':'COMPLETE','backbone':args.backbone,'track':args.track,'task':task,'family':family,'fold':fold,'seed':0,
                    'method':'TabACE_Specific_R5_reimplemented','parameters':parameters.tolist(),'regularization':best,
                    'inner_scores':scores,'inner_optimization':inner_optimization,'optimization':optimization,
                    'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),'fold_map_source':source,
                    'family_identifier_visible':True,'strength_used_for_inference':False,'held_out_labels_used_for_fit':False,
                    'deployed_alternative_branches':5,'archived_uncapped_Specific_reproduced':False,'rows':rows})
                print('DONE',args.backbone,args.track,task,family,fold,optimization,flush=True)


if __name__=='__main__':main()
