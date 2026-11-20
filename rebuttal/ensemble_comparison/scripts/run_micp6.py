"""Six MICP experts on frozen TabPFN, explicitly without CaPFN adaptation."""
import argparse,json,logging,time,traceback,warnings
import numpy as np
import torch
from prepare_data import RUN,atomic_json,protocol
from shared_reference import specifications,load_condition
from run_tabpfn import model,micp_no_capfn
from run_tabicl import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    jobs=[]
    for track in ['five_feature_calibrated','seven_reimplemented']:
        families=protocol.FAMILIES
        if track.startswith('seven'):
            from seven_shifts import FAMILIES
            families=FAMILIES
        conditions=[('original' if track.startswith('five') else 'covariate',0)]+[(f,r) for f in families for r in [20,50,80,100]]
        jobs.extend((m,f,r) for m in specifications(track) if m['task']=='classification' for f,r in conditions)
    est=model('classification','cuda:0');failures=0
    for meta,family,rho in jobs[a.shard::a.shards]:
        out=RUN/'results/micp6/tabpfn'/meta['track']/'classification'/str(meta['task_id'])/f'{family}_{rho}.npz'
        if out.with_suffix('.json').exists():continue
        try:
            data,info=load_condition(meta,family,rho)
            X,y=data.get('X_pool',data['X_source']),data.get('y_pool',data['y_source'])
            tick=time.perf_counter();prediction,routing=micp_no_capfn(est,X,y,data['X_target'],info['n_classes'],budget=len(data['y_source']),count_override=6)
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            name='MixturePFN_MICP6_no_CaPFN';observed=metrics(data['y_target'],prediction,'classification',1.)|{'seconds':elapsed}
            out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,y_target=data['y_target'],**{'prediction__'+name:prediction});tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','backbone':'tabpfn','seed':0,'methods':{name:observed},
                'registered_prompters':6,'nonempty_prompters':len(set(routing['routing'])),'context_budget':len(data['y_source']),
                'routing':routing,'adaptation_training':False,'original_full_MixturePFN_evaluated':False,
                'target_labels_used_for_fitting':False})
            print('MICP6_DONE',meta['track'],meta['task_id'],family,rho,flush=True)
        except Exception as exc:
            failures+=1;atomic_json(RUN/'results/inference_failures'/f"micp6_{meta['track']}_{meta['task_id']}_{family}_{rho}.json",
                {'error':repr(exc),'traceback':traceback.format_exc()});print('MICP6_FAILED',repr(exc),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
