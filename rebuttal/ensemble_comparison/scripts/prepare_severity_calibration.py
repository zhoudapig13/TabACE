"""Reserve new calibration queries disjoint from every original benchmark row."""
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from prepare_data import RUN,protocol,atomic_json


def prepare(path):
    meta=json.loads(path.read_text());out=RUN/'cache/severity_calibration/classification'/path.name
    if out.exists():return json.loads(out.read_text())
    spec=protocol.DatasetSpec(meta['task_id'],meta['data_id'],meta['dataset'],'target')
    frame,labels,_,(_,original)=protocol._task_data(spec)
    numeric,_=protocol._numeric_matrix(frame);encoded=LabelEncoder().fit_transform(labels)
    positions={int(v):i for i,v in enumerate(original)}
    with np.load(path.with_suffix('.npz')) as raw:
        S=raw['source_indices'];T=raw['target_indices'];X=raw['X_source'];y=raw['y_source']
        source_rows=np.array([positions[int(v)] for v in S]);test_rows=np.array([positions[int(v)] for v in T])
        assert np.array_equal(numeric[source_rows].astype(np.float32),X)
        assert np.array_equal(numeric[test_rows].astype(np.float32),raw['X_target'])
        assert np.array_equal(encoded[source_rows],y)
    excluded=set(map(int,np.r_[S,T]));available=np.array([i for i,v in enumerate(original) if int(v) not in excluded])
    record=meta|{'calibration_status':'INELIGIBLE','available_disjoint_rows':len(available),
                 'source_budget':len(y),'query_budget':128,'calibration_seed':314159+int(meta['data_id'])}
    if len(available)>=128:
        selected=protocol._balanced_cap_indices(encoded[available],128,record['calibration_seed'])
        query_rows=available[selected];Q=numeric[query_rows].astype(np.float32);truth=encoded[query_rows]
        if len(np.unique(truth))>=2 and set(truth).issubset(set(y)):
            ids=original[query_rows]
            assert set(map(int,ids)).isdisjoint(excluded)
            record.update({'calibration_status':'READY','n_classes':len(np.unique(y)),
                'query_indices_sha256':protocol.array_sha256(ids),'query_X_sha256':protocol.array_sha256(Q),
                'query_y_sha256':protocol.array_sha256(truth),'original_source_and_test_excluded':True})
            out.parent.mkdir(parents=True,exist_ok=True)
            tmp=out.with_suffix('.tmp.npz');np.savez_compressed(tmp,X_source=X,y_source=y,X_target=Q,y_target=truth,
                source_indices=S,original_test_indices=T,calibration_query_indices=ids);tmp.replace(out.with_suffix('.npz'))
    atomic_json(out,record)
    print('CALIBRATION_RESERVED',meta['task_id'],record['calibration_status'],len(available),flush=True)
    return record


def main():
    paths=sorted((RUN/'cache/five/classification').glob('*.json'))
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(prepare,paths))
    manifest={'status':'READY','development_queries':'new rows outside the original source and target indices',
        'source_contexts':'unchanged original 256 rows','query_rows':128,'seed':0,
        'ready_datasets':sum(r['calibration_status']=='READY' for r in records),'datasets':records,
        'original_test_labels_used_for_severity_selection':False}
    atomic_json(RUN/'configs/severity_calibration_manifest.json',manifest)
    print('CALIBRATION_DATASETS',manifest['ready_datasets'],flush=True)


if __name__=='__main__':main()
