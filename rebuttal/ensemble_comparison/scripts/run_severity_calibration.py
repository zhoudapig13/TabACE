"""Evaluate predeclared source-corruption severities on reserved queries only."""
import argparse,json,logging,time,warnings
import numpy as np
import torch
from prepare_data import RUN,protocol,atomic_json
from calibrated_five_protocol import corrupt,candidates
from run_tabicl import Predictor,metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    manifest=json.loads((RUN/'configs/severity_calibration_manifest.json').read_text())
    datasets=[r for r in manifest['datasets'] if r['calibration_status']=='READY']
    configs=[{'name':'clean'}]+candidates()
    jobs=[(m,c) for c in configs for m in datasets][args.shard::args.shards]
    predictor=Predictor('classification','cuda:0');failures=0
    for meta,config in jobs:
        out=RUN/'results/severity_calibration/five_classification'/config['name']/f"{meta['task_id']}.json"
        if out.exists():continue
        with np.load(RUN/'cache/severity_calibration/classification'/f"{meta['task_id']}.npz") as f:
            X,y,Q,t=f['X_source'],f['y_source'],f['X_target'],f['y_target']
        rows=[];predictions=[]
        conditions=[('original',0)] if config['name']=='clean' else [(f,r) for f in protocol.FAMILIES for r in [20,50,80,100]]
        try:
            for family,rho in conditions:
                if rho==0:A,b,details=X,y,{'affected_rows':0,'labels_changed':0}
                else:A,b,details=corrupt(X,y,family=family,rho=rho/100.,**{k:v for k,v in config.items() if k!='name'})
                tick=time.perf_counter();prediction,_=predictor(A,b,Q,n_classes=meta['n_classes'])
                torch.cuda.synchronize()
                row={'family':family,'rho':rho,**metrics(t,prediction,'classification',1.),'seconds':time.perf_counter()-tick,
                     'source_X_sha256':protocol.array_sha256(A),'source_y_sha256':protocol.array_sha256(b),**details}
                rows.append(row);predictions.append(prediction)
            out.parent.mkdir(parents=True,exist_ok=True)
            temp=out.with_suffix('.tmp.npz');np.savez_compressed(temp,predictions=np.stack(predictions),y_target=t);temp.replace(out.with_suffix('.npz'))
            atomic_json(out,{'status':'COMPLETE','task_id':meta['task_id'],'data_id':meta['data_id'],'dataset':meta['dataset'],
                'seed':0,'configuration':config,'rows':rows,'calibration_queries_disjoint_from_original_source_and_test':True,
                'original_test_labels_used':False,'TabACE_results_used_for_selection':False})
            print('SEVERITY_DONE',config['name'],meta['task_id'],float(np.mean([r['accuracy'] for r in rows])),float(np.mean([r['auc'] for r in rows])),flush=True)
        except Exception as exc:
            failures+=1;print('SEVERITY_FAILED',config['name'],meta['task_id'],repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
