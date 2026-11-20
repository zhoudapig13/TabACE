"""Audit the entire source-validation registry before parameter selection."""
import json
import numpy as np
import continuous_recovery_v30 as experiment
from prepare_data import atomic_json,protocol
from run_tabicl import metrics


def main():
    failures=[];count=0;reconstructed=0;datasets=0
    for block,meta in experiment.v.prior.jobs():
        path=experiment.OUT/'validation'/block/f'{meta["task_id"]}.npz'
        if not path.exists() or not path.with_suffix('.json').exists():failures.append([str(path),'missing']);continue
        pool=experiment.v.prior.original.prepare('regression',block,meta)
        split=experiment.v.split(meta,block,pool)
        if np.intersect1d(split['source_original_ids'],split['validation_original_ids']).size:failures.append([str(path),'source_validation_overlap'])
        records=json.loads(path.with_suffix('.json').read_text())['records'];seen=set()
        with np.load(path) as bank:
            truth=bank['y_validation'];scale=max(float(truth.std()),1e-12)
            if not np.array_equal(truth,split['y_validation']):failures.append([str(path),'truth_mismatch'])
            for r in records:
                key=(r['family'],r['rho'],r['restoration_fraction'],r['requested_context'])
                if key in seen:failures.append([str(path),'duplicate',key])
                seen.add(key)
                prediction=bank[r['prediction_key']]
                if not np.isfinite(prediction).all():failures.append([str(path),'nonfinite'])
                measured=metrics(truth,prediction,'regression',scale)
                if any(not np.isclose(value,r['metrics'][name],atol=1e-12,rtol=0) for name,value in measured.items()):failures.append([str(path),'metric_mismatch',key])
                if r['input_hashes']['X_target']!=protocol.array_sha256(split['X_validation']) or r['input_hashes']['y_target']!=protocol.array_sha256(truth):failures.append([str(path),'validation_partition',key])
                if r['seed']!=0 or r['n_estimators']!=1 or not r['original_core256_preserved']:failures.append([str(path),'inference_configuration',key])
                if r['family']==experiment.v.grid(block)[0][0] and r['rho']==20 and r['restoration_fraction']==.005:
                    env=experiment.v.environment(pool,meta,block,r['family'],r['rho'],split['train_positions'])
                    A,b=experiment.context(env,.005,r['requested_context'])
                    if any(protocol.array_sha256(a)!=r['input_hashes'][name] for name,a in [('X_source',A),('y_source',b)]):failures.append([str(path),'source_input_reconstruction',key])
                    reconstructed+=1
                count+=1
        expected={(family,rho,rate,budget) for family,rho in experiment.v.grid(block) for rate in experiment.INITIAL for budget in [1024,4096,8192]}
        if seen!=expected:failures.append([str(path),'configuration_coverage',len(seen),len(expected)])
        datasets+=1
    if count!=43848 or datasets!=68:failures.append(['global_coverage',count,datasets,43848,68])
    atomic_json(experiment.OUT/'validation_audit.json',dict(status='PASS' if not failures else 'FAIL',
        verified_condition_predictions=count,expected_condition_predictions=43848,
        verified_dataset_blocks=datasets,reconstructed_source_contexts=reconstructed,
        source_validation_ids_disjoint=True,original_queries_not_evaluated=True,failures=failures))
    print('CONTINUOUS_VALIDATION_AUDIT',count,datasets,len(failures),failures[:3],flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
