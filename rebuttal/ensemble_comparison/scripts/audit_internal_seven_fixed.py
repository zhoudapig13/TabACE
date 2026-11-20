"""Verify saved internal predictions, frozen inputs, policies and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from prepare_data import ROOT,RUN,atomic_json
from internal_seven_fixed_strength import OUT,SELECTED,cells


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-12,None);p/=p.sum(1,keepdims=True)
    aucs=[roc_auc_score(y==i,p[:,i]) for i in range(p.shape[1]) if np.any(y==i) and np.any(y!=i)]
    return {'accuracy':float(np.mean(p.argmax(1)==y)),'auc':float(np.mean(aucs)) if aucs else None}


def main(base_only):
    selected=json.loads(SELECTED.read_text());assert selected['status']=='FROZEN'
    assert selected['strengths']==[.2,.5,.8,1.] and selected['strength_rescaled'] is False
    assert selected['development_labels_used_for_benchmark_selection'] is True
    from run_internal_seven_ensembles import MAIN,policies
    counts={'selected_Base_predictions':0,'ensemble_conditions':0,'ensemble_predictions':0,
            'all_trial_strength_records':0,'disjoint_policy_datasets':0}
    for path in (OUT/'base').glob('*/*/*.json'):
        d=json.loads(path.read_text());assert d['rho'] in [20,50,80,100]
        assert d['strength']==d['rho']/100. and d['strength_rescaled'] is False
        counts['all_trial_strength_records']+=1
    ids=set()
    for meta,family,rho in cells():
        key=f'{family}_{rho}.npz';path=OUT/'base'/selected['candidate']/str(meta['task_id'])/key
        d=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            truth=z['y_target'];base=z['prediction'];observed=recompute(truth,base)
        for k in observed:assert abs(observed[k]-d['metrics'][k])<1e-12,(path,k)
        counts['selected_Base_predictions']+=1
        if meta['task_id'] not in ids:
            policies(int(meta['outer_fold']),int(meta['data_id']));ids.add(meta['task_id'])
            counts['disjoint_policy_datasets']+=1
        if base_only:continue
        path=OUT/'ensembles'/str(meta['task_id'])/key
        record=json.loads(path.with_suffix('.json').read_text())
        assert record['input_hashes']==d['input_hashes']
        assert record['rho']==rho and record['strength']==rho/100. and not record['strength_rescaled']
        assert record['generator_parameters']==selected['generator_parameters']
        assert record['rows_per_context']==256 and record['logical_bank_members']==6
        assert record['source_pool_rows']==320
        assert not record['method_parameters_refit_on_new_benchmark'] and not record['Shared_run']
        with np.load(path) as z:
            assert np.array_equal(z['y_target'],truth)
            assert np.array_equal(z['prediction__Base'],base)
            for bank in ['raw_bank','adapted_bank','feature_bank']:
                assert z[bank].shape==(6,len(truth),2) and np.isfinite(z[bank]).all()
            for name,reported in record['methods'].items():
                p=z['prediction__'+name];assert np.isfinite(p).all() and p.shape==(len(truth),2)
                observed=recompute(truth,p)
                for k in observed:assert abs(observed[k]-reported[k])<1e-12,(path,name,k)
                counts['ensemble_predictions']+=1
            assert all('prediction__'+name in z for name in MAIN)
        counts['ensemble_conditions']+=1
    assert counts['selected_Base_predictions']==980 and counts['disjoint_policy_datasets']==35
    if not base_only:assert counts['ensemble_conditions']==980
    # One complete verification at the end, not repeated in progress polling.
    frozen=json.loads((RUN/'configs/tabicl_clean_five_frozen.json').read_text())['files_sha256']
    changed=[name for name,expected in frozen.items() if sha(RUN/name)!=expected]
    assert not changed,changed
    provenance=OUT/'run_provenance.json'
    if provenance.exists():
        for name,expected in json.loads(provenance.read_text())['sha256'].items():
            assert sha(RUN/name)==expected,name
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    result={'status':'PASS','scope':'selected Base only' if base_only else 'selected Base and all ensemble comparisons',
        'counts':counts,'frozen_Clean_five_and_existing_summary_files_verified_unchanged':len(frozen),
        'all_saved_selected_prediction_metrics_recomputed':True,'strengths':[.2,.5,.8,1.],
        'strengths_not_rescaled':True,'generator_parameters_identical_across_methods':True,
        'method_policy_folds_disjoint':True,'benchmark_uses_development_labels':True,
        'internal_only':True,'paper_reproduction_claim':False,'Markdown_bytes':0,'Shared_run':False}
    atomic_json(OUT/('base_audit.json' if base_only else 'full_audit.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-only',action='store_true');a=p.parse_args();main(a.base_only)
