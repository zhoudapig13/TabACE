"""Reserve regression calibration queries outside original history/reference/test."""
import json
import numpy as np
from prepare_data import RUN,atomic_json,protocol
from seven_shifts import registry,regression_clean


def main():
    records=[]
    for meta in registry():
        if meta['task']!='regression':continue
        folder=RUN/'cache/severity_calibration/seven_regression';out=folder/f"{meta['task_id']}.json"
        if out.exists():records.append(json.loads(out.read_text()));continue
        with np.load(RUN/'cache/five/regression'/f"{meta['task_id']}.npz") as f:
            X,y,S,T=f['X_source'],f['y_source'],f['source_indices'],f['target_indices']
        rng=np.random.default_rng(2718);order=rng.permutation(len(y));pool=order[:min(len(y),1024)];unused=order[min(len(y),1024):]
        H,hy,R,ry,A,b,Q,t=regression_clean(meta);budget=len(t)
        record=meta|{'calibration_status':'INELIGIBLE','available_disjoint_rows':len(unused),'query_budget':budget}
        if len(unused)>=budget:
            ids=np.random.default_rng(161803+int(meta['data_id'])).choice(unused,budget,replace=False)
            assert set(map(int,S[ids])).isdisjoint(set(map(int,S[pool])))
            assert set(map(int,S[ids])).isdisjoint(set(map(int,T)))
            folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out.with_suffix('.npz'),X_history=H,y_history=hy,X_reference=R,y_reference=ry,
                X_source=A,y_source=b,X_target=X[ids],y_target=y[ids],calibration_indices=S[ids],original_test_indices=T)
            record.update(calibration_status='READY',calibration_indices_sha256=protocol.array_sha256(S[ids]),
                          original_history_reference_source_and_test_excluded=True)
        atomic_json(out,record);records.append(record)
        print('REGRESSION_CALIBRATION_RESERVED',meta['task_id'],record['calibration_status'],len(unused),flush=True)
    atomic_json(RUN/'configs/seven_severity_calibration_manifest.json',{'status':'READY','datasets':records,
        'ready_datasets':sum(r['calibration_status']=='READY' for r in records),'original_test_labels_used_for_severity_selection':False})


if __name__=='__main__':main()
