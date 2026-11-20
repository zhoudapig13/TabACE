"""Reserve a third query set for checking benchmark-tuned fusion policies."""
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from prepare_data import RUN,LOCAL_CC18,protocol,atomic_json
from seven_shifts import registry

SEVEN={m['task_id']:m for m in registry() if m['task']=='classification'}


def classification(path):
    meta=json.loads(path.read_text());out=RUN/'cache/tuning_holdout/classification'/path.name
    if out.exists():return json.loads(out.read_text())
    spec=protocol.DatasetSpec(meta['task_id'],meta['data_id'],meta['dataset'],'target')
    frame,labels,_,(_,original)=protocol._task_data(spec);X,_=protocol._numeric_matrix(frame);y=LabelEncoder().fit_transform(labels)
    with np.load(path.with_suffix('.npz')) as f:S,T=f['source_indices'],f['target_indices']
    excluded=set(map(int,np.r_[S,T]));cal=RUN/'cache/severity_calibration/classification'/path.with_suffix('.npz').name
    if cal.exists():
        with np.load(cal) as f:excluded.update(map(int,f['calibration_query_indices']))
    if meta['task_id'] in SEVEN:
        ids=protocol._balanced_cap_indices(y,448,0)
        source,target=train_test_split(ids,train_size=320,test_size=128,stratify=y[ids],random_state=0)
        excluded.update(map(int,original[np.r_[source,target]]))
    available=np.array([i for i,idx in enumerate(original) if int(idx) not in excluded]);status='INELIGIBLE'
    if len(available)>=128:
        rows=available[protocol._balanced_cap_indices(y[available],128,271828+int(meta['data_id']))]
        if len(np.unique(y[rows]))>=2:
            out.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out.with_suffix('.npz'),X_target=X[rows],y_target=y[rows],query_indices=original[rows])
            assert set(map(int,original[rows])).isdisjoint(excluded)
            status='READY'
    record=meta|{'holdout_status':status,'query_budget':128,'available_unused_rows':len(available),
        'original_source_and_benchmark_queries_and_severity_calibration_queries_excluded':True,
        'used_to_tune_any_parameter':False}
    atomic_json(out,record);print('TUNING_HOLDOUT',meta['task_id'],status,flush=True);return record


def regression(meta):
    out=RUN/'cache/tuning_holdout/seven_regression'/f"{meta['task_id']}.json"
    if out.exists():return json.loads(out.read_text())
    with np.load(RUN/'cache/five/regression'/f"{meta['task_id']}.npz") as f:
        X,y,S,T=f['X_source'],f['y_source'],f['source_indices'],f['target_indices']
    order=np.random.default_rng(2718).permutation(len(y));excluded=set(map(int,order[:min(len(y),1024)]))
    cal=RUN/'cache/severity_calibration/seven_regression'/f"{meta['task_id']}.npz"
    if cal.exists():
        with np.load(cal) as f:cal_ids=set(map(int,f['calibration_indices']))
        excluded.update(i for i,idx in enumerate(S) if int(idx) in cal_ids)
    unused=np.asarray([i for i in range(len(y)) if i not in excluded]);count=min(256,len(T));status='INELIGIBLE'
    if len(unused)>=count:
        ids=np.random.default_rng(141421+int(meta['data_id'])).choice(unused,count,replace=False)
        out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),X_target=X[ids],y_target=y[ids],query_indices=S[ids])
        assert set(map(int,S[ids])).isdisjoint(set(map(int,T)))
        status='READY'
    record=meta|{'holdout_status':status,'query_budget':count,'available_unused_rows':len(unused),
        'source_pool_benchmark_queries_and_severity_calibration_queries_excluded':True,'used_to_tune_any_parameter':False}
    atomic_json(out,record);print('REGRESSION_TUNING_HOLDOUT',meta['task_id'],status,flush=True);return record


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        classification_rows=list(pool.map(classification,sorted((RUN/'cache/five/classification').glob('*.json'))))
    regression_rows=[regression(m) for m in registry() if m['task']=='regression']
    atomic_json(RUN/'configs/benchmark_tuning_holdout_manifest.json',{'classification':classification_rows,'seven_regression':regression_rows,
        'classification_ready':sum(x['holdout_status']=='READY' for x in classification_rows),
        'seven_regression_ready':sum(x['holdout_status']=='READY' for x in regression_rows),
        'independent_query_labels_used_for_training_or_configuration_selection':False})


if __name__=='__main__':main()
