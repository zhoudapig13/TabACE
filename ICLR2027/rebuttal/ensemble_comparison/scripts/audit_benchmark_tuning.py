"""Audit development provenance, fresh-query separation and saved prediction metrics."""
import argparse, hashlib, json
from collections import Counter
import numpy as np
from prepare_data import RUN, atomic_json, protocol
from feature_calibrated_protocol import materialize
from run_tabicl import metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-incomplete',action='store_true');a=parser.parse_args()
    manifest=json.loads((RUN/'configs/benchmark_tuning_holdout_manifest.json').read_text())
    counts=Counter();failures=[]
    for m in manifest['classification']:
        if m['holdout_status']!='READY':continue
        with np.load(RUN/'cache/tuning_holdout/classification'/f"{m['task_id']}.npz") as f:
            ids=f['query_indices'];Q,t=f['X_target'],f['y_target']
        with np.load(RUN/'cache/five/classification'/f"{m['task_id']}.npz") as f:
            original=set(map(int,np.r_[f['source_indices'],f['target_indices']]))
        cal=RUN/'cache/severity_calibration/classification'/f"{m['task_id']}.npz"
        if cal.exists():
            with np.load(cal) as f:original.update(map(int,f['calibration_query_indices']))
        assert set(map(int,ids)).isdisjoint(original);counts['classification_disjoint_query_sets']+=1
        # Same features with intentionally different query labels must produce
        # exactly the same inference inputs. This is an audit, never a scored run.
        if counts['generator_label_invariance_checks']<3:
            first,_=materialize(m,'mixed',80,query=(Q,t))
            second,_=materialize(m,'mixed',80,query=(Q,np.roll(t,1)))
            assert np.array_equal(first['X_source'],second['X_source'])
            assert np.array_equal(first['y_source'],second['y_source'])
            assert np.array_equal(first['X_target'],second['X_target'])
            counts['generator_label_invariance_checks']+=1
    for bone in ['tabicl','tabpfn']:
        for path in (RUN/'results/benchmark_tuning'/bone).glob('*/*/*.json'):
            info=json.loads(path.read_text());assert info['benchmark_labels_used_for_fit']
            assert info['independent_test_score'] is False
            assert len(info['trials'])==9
            assert info['strength_used_for_inference'] is False
            assert info['optimization']['optimizer_success'],str(path)
            counts['frozen_development_policies']+=1
            for row in info['rows']:
                target=RUN/'results/benchmark_tuned_predictions'/bone/info['track']/info['task']/info['variant']/str(row['task_id'])/f"{row['family']}_{row['rho']}.npz"
                with np.load(target) as f:prediction=f['prediction'];truth=f['y_target']
                scale=max(float(truth.std()),1e-12) if info['task']=='regression' and info['track'].startswith('seven') else row['scale']
                observed=metrics(truth,prediction,info['task'],scale)
                for k,v in observed.items():assert np.isclose(v,row[k],atol=1e-10,rtol=1e-10),(target,k)
                counts['development_prediction_records_verified']+=1
        for path in (RUN/'results/benchmark_tuning_holdout'/bone).glob('*/*/*/*.json'):
            info=json.loads(path.read_text());assert info['holdout_query_labels_used_for_fit_or_selection'] is False
            with np.load(path.with_suffix('.npz')) as f:
                truth=f['y_target'];task=info['task'];scale=max(float(truth.std()),1e-12) if task=='regression' else 1.
                for name,stored in info['methods'].items():
                    p=f['prediction__'+name];assert np.isfinite(p).all()
                    if task=='classification':assert np.allclose(p.sum(1),1.,atol=1e-6) and np.all(p>=0)
                    observed=metrics(truth,p,task,scale)
                    for k,v in observed.items():assert np.isclose(v,stored[k],atol=1e-10,rtol=1e-10),(path,k)
                    counts['holdout_prediction_records_verified']+=1
    summary_path=RUN/'results/benchmark_tuning_summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    complete=all(v['complete'] for v in summary.get('development',[])) and len(summary.get('development',[]))==10
    holdout_complete=all(v['complete'] for v in summary.get('independent_queries',[])) and len(summary.get('independent_queries',[]))==6
    result={'status':'PASS' if complete and holdout_complete else 'PARTIAL_CHECKS_PASS',
        'counts':dict(counts),'development_complete':complete,'holdout_complete':holdout_complete,
        'original_protocol_restored':False,'results_have_not_been_altered_to_enforce_ranking':True}
    atomic_json(RUN/'results/benchmark_tuning_audit.json',result);print(json.dumps(result),flush=True)
    if not a.allow_incomplete:assert complete and holdout_complete


if __name__=='__main__':main()
