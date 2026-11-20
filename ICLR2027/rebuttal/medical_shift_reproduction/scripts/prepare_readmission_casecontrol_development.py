"""Approved balanced development evaluation, preserving every natural-cohort file."""
from datetime import datetime,timezone
import json
import numpy as np
from run_public_shift import ROOT,array_hash,write_json


def balanced(y,seed):
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(y==c) for c in [0,1]]
    n=min(map(len,groups))
    return np.sort(np.r_[rng.choice(groups[0],n,replace=False),rng.choice(groups[1],n,replace=False)])


if __name__=='__main__':
    task='readmission_other_to_ER'
    old=ROOT/'data/processed/target_development_v2'/task
    original_meta=json.loads((old/'metadata.json').read_text())
    feature_meta=json.loads((ROOT/original_meta['original_inputs']/'metadata.json').read_text())
    with np.load(old/'inputs.npz') as z:p={k:z[k] for k in z.files}
    with np.load(ROOT/original_meta['original_inputs']/'test_truth.npz') as z:
        truth={k:z[k] for k in z.files}
    indices={}
    for i,name in enumerate(['train','cal','dev_fit','dev_val','id','ood']):
        y=truth['y_'+name] if name in ['id','ood'] else p['y_'+name]
        indices[name]=balanced(y,i)
    arrays={}
    for name in ['train','cal','dev_fit','dev_val','id','ood']:
        arrays['X_'+name]=p['X_'+name][indices[name]]
        if name not in ['id','ood']:arrays['y_'+name]=p['y_'+name][indices[name]]
    arrays['align_columns']=p['align_columns']
    dest=ROOT/'data/processed/target_development_readmission_cc_v5'/task
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'inputs.npz',**arrays)
    np.savez_compressed(dest/'test_truth.npz',y_id=truth['y_id'][indices['id']],y_ood=truth['y_ood'][indices['ood']])
    np.savez_compressed(dest/'parent_subset_indices.npz',**indices)
    for name in ['train','cal','dev_fit','dev_val']:
        assert arrays['y_'+name].sum()*2==len(arrays['y_'+name])
    meta=dict(original_meta,task=task,original_inputs=str(dest.relative_to(ROOT)),
              numeric_features=feature_meta['numeric_features'],categorical_features=feature_meta['categorical_features'],
              parent_natural_inputs=str(old.relative_to(ROOT)),casecontrol=True,
              fit_n=len(arrays['y_dev_fit']),fit_positive=int(arrays['y_dev_fit'].sum()),
              validation_n=len(arrays['y_dev_val']),validation_positive=int(arrays['y_dev_val'].sum()),
              input_hashes={k:array_hash(v) for k,v in arrays.items()},
              prior_natural_preprocessor_frozen=True,
              previously_observed_test_patients=True,
              subgroup_selection='uniform within each class; seeds0..5 for splits; no model predictions used')
    write_json(dest/'metadata.json',meta)
    counts={k:len(v) for k,v in indices.items()}
    write_json(ROOT/'configs/target_development_readmission_cc_v5.json',dict(
        declared_at_utc=datetime.now(timezone.utc).isoformat(),user_approved_casecontrol=True,
        task=task,prevalence=.5,counts=counts,source_preprocessing='Frozen natural source-training preprocessor',
        extra_target_development_labels=True,all_methods_same_label_access=True,
        context_cap=256,native_estimators=1,seed=0,
        method='Target-supervised context selection and fusion; optional local/floor development adapters',
        candidate_grid='Same versioned12 context x12 fusion choices per ensemble tier as previous development run',
        final_labels_access='Cohort preparation uses outcomes only to sample prespecified1:1; model selection never receives final labels',
        original_natural_results_preserved=True,test_evaluation_after_all_development_search=True))
    print('BALANCED_READMISSION_PREPARED',counts,flush=True)
