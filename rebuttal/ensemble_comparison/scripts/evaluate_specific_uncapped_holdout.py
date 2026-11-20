"""Score frozen uncapped available-pool policies on already-reserved query banks."""
import hashlib,json,time
import numpy as np
from prepare_data import RUN,atomic_json
from fit_benchmark_specific_uncapped import predict
from evaluate_benchmark_tuning_holdout import cell,build_input
from run_tabicl import metrics


def main():
    manifest=json.loads((RUN/'configs/benchmark_tuning_holdout_manifest.json').read_text())
    pending=[]
    for bone in ['tabicl','tabpfn']:
        for track in ['five_calibrated','five_feature_calibrated','seven_calibrated']:
            key='seven_regression' if track.startswith('seven') else 'classification'
            families=['covariate','label','concept','selection','domain','temporal','support'] if track.startswith('seven') else ['shortcut','boundary','band','local','mixed']
            for meta in manifest[key]:
                if meta['holdout_status']!='READY':continue
                for family in families:
                    for rho in [20,50,80,100]:pending.append((bone,track,key,meta,family,rho))
    while pending:
        remaining=[];progress=0
        for bone,track,key,meta,family,rho in pending:
            task=meta['task'];name=f'{family}_{rho}.npz'
            out=RUN/'results/benchmark_specific_uncapped_holdout'/bone/track/task/str(meta['task_id'])/name
            if out.with_suffix('.json').exists():continue
            policy_path=RUN/'results/benchmark_specific_uncapped'/bone/track/task/f'specific_{family}.json'
            source=RUN/'results/benchmark_tuning_holdout'/bone/track/task/str(meta['task_id'])/name
            if not policy_path.exists() or not source.with_suffix('.json').exists():
                remaining.append((bone,track,key,meta,family,rho));continue
            policy=json.loads(policy_path.read_text());info=json.loads(source.with_suffix('.json').read_text())
            with np.load(source) as f:
                bank=np.concatenate([f['shared_bank'],f['specific_bank']],axis=0);base=f['prediction__Base'];truth=f['y_target'];ids=f['query_indices']
            if task=='regression':
                with np.load(RUN/'cache/tuning_holdout'/key/f"{meta['task_id']}.npz") as f:Q,t=f['X_target'],f['y_target']
                data,_=build_input(meta,track,family,rho,Q,t);labels=data.get('y_pool',data['y_source'])
                mean=float(labels.mean());scale=max(float(labels.std()),1e-9)
            else:mean=0.;scale=1.
            c=cell(info,base,bank,truth,mean,scale)
            prediction=predict(c,np.asarray(policy['parameters']),task,policy['gain'])
            observed=metrics(truth,prediction,task,max(float(truth.std()),1e-12) if task=='regression' else 1.)
            out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,prediction=prediction,y_target=truth,query_indices=ids);tmp.replace(out)
            atomic_json(out.with_suffix('.json'),info|{'status':'COMPLETE','method':'TabACE_Specific_uncapped_available_pool',
                'metrics':observed,'context_cap':None,'deployed_alternative_branches':bank.shape[0],
                'query_labels_used_for_fit_or_selection':False,
                'policy_sha256':hashlib.sha256(policy_path.read_bytes()).hexdigest()})
            progress+=1
        pending=remaining
        print('UNCAPPED_HOLDOUT_PROGRESS',progress,'remaining',len(pending),flush=True)
        if pending:time.sleep(10)


if __name__=='__main__':main()
