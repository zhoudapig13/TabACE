"""Check the frozen probe's coverage, immutable truths, scores and inputs."""
import json
import numpy as np
import query_seven_monotonic_v26 as probe
from prepare_data import atomic_json,protocol
from run_tabicl import metrics


def main():
    v=probe.v;settings=json.loads((probe.OUT/'parameters.json').read_text())
    settings_hash=v.prior.original.sha(probe.OUT/'parameters.json')
    failures=[];count=0;reconstructed=0
    for block,meta in v.prior.jobs():
        if block!='seven':continue
        pool=None
        for family,rho in v.grid('seven'):
            old=v.prior.original.frozen.OUT/'input_banks/regression/seven'/str(meta['task_id'])/f'{family}_{rho}.npz'
            with np.load(old) as z:truth=z['y_target'];original=z['Base_prediction']
            for budget in v.BUDGETS:
                path=probe.OUT/'predictions'/str(budget)/str(meta['task_id'])/old.name
                if not path.exists() or not path.with_suffix('.json').exists():failures.append([str(path),'missing']);continue
                record=json.loads(path.with_suffix('.json').read_text())
                with np.load(path) as z:p=z['prediction'];y=z['y_target']
                if not np.array_equal(y,truth):failures.append([str(path),'truth_changed'])
                if budget==256 and not np.array_equal(p,original):failures.append([str(path),'Base256_changed'])
                measured=metrics(y,p,'regression',record['scale'])
                if any(not np.isclose(value,record['metrics'][k],atol=1e-12,rtol=0) for k,value in measured.items()):failures.append([str(path),'metric_mismatch'])
                if record['settings_sha256']!=settings_hash:failures.append([str(path),'frozen_settings_hash'])
                if record['actual_context']>budget or record['seed']!=0 or record['n_estimators']!=1:failures.append([str(path),'inference_configuration'])
                if family=='domain' and rho==20 and budget>256:
                    if pool is None:pool=v.prior.original.prepare('regression','seven',meta)
                    env=v.environment(pool,meta,'seven',family,rho,np.arange(len(pool['y'])))
                    A,b,_=v.context(env,settings['repair_probability'],budget)
                    if any(protocol.array_sha256(x)!=record['input_hashes'][k] for k,x in [('X_source',A),('y_source',b)]):failures.append([str(path),'source_reconstruction'])
                    reconstructed+=1
                count+=1
    actual=len([p for p in (probe.OUT/'predictions').glob('*/*/*.json') if '.tmp.' not in p.name])
    if count!=3696 or actual!=3696:failures.append(['coverage',count,actual,3696])
    result=dict(status='PASS' if not failures else 'FAIL',verified_condition_predictions=count,
        expected_condition_predictions=3696,reconstructed_larger_context_sources=reconstructed,
        source_selection_used_original_query_labels=False,development_only=True,
        Base256_and_all_truths_preserved=True,failures=failures)
    atomic_json(probe.OUT/'audit.json',result);print(json.dumps(result),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
