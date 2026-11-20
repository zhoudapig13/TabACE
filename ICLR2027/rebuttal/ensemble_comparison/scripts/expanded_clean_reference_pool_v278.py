"""Expand permitted clean reference data, reserving original64 for calibration.

These are extra labeled data, not additional labels from final evaluation.
The pool can exceed256; every later TFM context must still have at most256 rows.
"""
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import clean_reference_manifest_v268 as old
from prepare_data import LOCAL_CC18,RUN,ROOT,atomic_json,protocol

source=old.source
OUT=RUN/'results/expanded_clean_reference_pool_v278'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'status':'EXTRA_LABELED_CLEAN_REFERENCE_POOL_NOT_ORIGINAL_SHARED',
        'seed':0,'maximum_reference_rows_including_reserved_calibration':2048,
        'reserved_calibration_rows':64,'maximum_additional_reference_training_rows':1984,
        'TFM_context_row_cap_unchanged':256,
        'reference_pool_size_and_TFM_context_size_are_distinct':True,
        'classification_reference_sampling':'uniform from original row identities excluding source, query and reserved calibration; source-supported classes only',
        'regression_reference_sampling':'remaining original training-cache rows excluding source, query and reserved calibration',
        'reference_labels_not_evaluation_labels':True,'extra_supervision_count_varies_with_available_rows':True,
        'reference_training_and_calibration_and_evaluation_identities_disjoint':True,
        'Base_and_benchmark_unchanged':True,'script_sha256':source.sha(__file__),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)
    oldrefs={(r['task'],r['task_id']):r for r in json.loads((old.OUT/'reference_manifest.json').read_text())}
    manifest=[]
    for c in json.loads((source.OUT/'case_manifest.json').read_text()):
        if c['track']!='perturbation':continue
        task,tid=c['task'],c['task_id']
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            X,y,Q,sid,qid=(z[k] for k in ('X_source','y_source','X_target','source_indices','target_indices'))
        ref=oldrefs[(task,tid)]
        assert source.sha(ref['reference_file'])==ref['reference_sha256']
        with np.load(ref['reference_file']) as z:
            CX,Cy,cid=(z[k] for k in ('X_reference','y_reference','reference_indices'))
        excluded=np.r_[sid,qid,cid]
        if task=='classification':
            folder=LOCAL_CC18/str(tid)
            frame=pd.read_parquet(folder/'X.parquet')
            labels=pd.read_parquet(folder/'y.parquet').iloc[:,0]
            valid=labels.notna().to_numpy();ids=np.flatnonzero(valid)
            allX,_=protocol._numeric_matrix(frame.loc[valid]);pos={int(row):i for i,row in enumerate(ids)}
            encoder=LabelEncoder().fit(labels.iloc[sid].to_numpy())
            np.testing.assert_array_equal(encoder.transform(labels.iloc[sid].to_numpy()),y)
            candidates=np.setdiff1d(ids,excluded)
            candidates=candidates[labels.iloc[candidates].isin(encoder.classes_).to_numpy()]
            chosen=np.random.default_rng(0).choice(candidates,min(1984,len(candidates)),replace=False)
            TX=allX[[pos[int(i)] for i in chosen]]
            Ty=encoder.transform(labels.iloc[chosen].to_numpy()).astype(y.dtype)
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in sid]],X)
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in qid]],Q)
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in cid]],CX)
        else:
            assert source.sha(c['original_raw_cache'])==c['original_cache_sha256']
            with np.load(c['original_raw_cache']) as z:
                ids,allX,ally=z['source_indices'],z['X_source'],z['y_source']
            pos={int(row):i for i,row in enumerate(ids)}
            candidates=np.setdiff1d(ids,excluded)
            chosen=np.random.default_rng(0).choice(candidates,min(1984,len(candidates)),replace=False)
            positions=[pos[int(i)] for i in chosen];TX,Ty=allX[positions],ally[positions]
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in sid]],X)
            np.testing.assert_array_equal(ally[[pos[int(i)] for i in sid]],y)
            np.testing.assert_array_equal(allX[[pos[int(i)] for i in cid]],CX)
        assert len(chosen)>0 and set(map(int,chosen)).isdisjoint(map(int,excluded))
        assert set(map(int,cid)).isdisjoint(map(int,np.r_[sid,qid]))
        dest=OUT/'references'/task/f'{tid}.npz'
        source.npwrite(dest,X_train_reference=TX,y_train_reference=Ty,train_reference_indices=chosen,
            X_calibration=CX,y_calibration=Cy,calibration_indices=cid)
        meta=({k:c[k] for k in ('task','task_id','data_id','dataset','fold')}|
            {'reference_file':str(dest),'reference_sha256':source.sha(dest),
             'additional_training_reference_rows':len(Ty),'reserved_calibration_rows':64,
             'total_additional_labeled_reference_rows':len(Ty)+64,
             'benchmark_input_sha256':c['input_sha256'],'old_calibration_file_sha256':ref['reference_sha256'],
             'source_query_calibration_training_reference_row_ids_disjoint':True,
             'evaluation_labels_opened':False})
        atomic_json(dest.with_suffix('.json'),meta);manifest.append(meta)
        print('EXPANDED_REFERENCE_READY',task,tid,len(Ty),64,flush=True)
    assert len(manifest)==107
    atomic_json(OUT/'reference_manifest.json',manifest)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(OUT/'audit.json',{'status':'PASS','datasets':107,
        'training_reference_rows_range':[min(r['additional_training_reference_rows'] for r in manifest),max(r['additional_training_reference_rows'] for r in manifest)],
        'original64_reserved_calibration_preserved':True,'all_row_identities_disjoint':True,
        'benchmark_source_and_query_features_replayed':True,'evaluation_labels_opened':False,'Markdown_bytes':0})


if __name__=='__main__':main()
