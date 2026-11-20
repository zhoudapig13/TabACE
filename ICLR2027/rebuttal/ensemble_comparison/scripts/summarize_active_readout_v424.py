"""Truth-based numerical summaries; never substitutes for the full audit."""
import argparse
import importlib
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from prepare_data import atomic_json
from run_tabicl import metrics


def main(module):
    run=importlib.import_module(module)
    track,task,family=run.features.schedule.endpoint.ENDPOINT
    cases=(run.split64.endpoint_cases() if hasattr(run,'split64')
           else run.features.schedule.endpoint.endpoint_cases())
    execution=json.loads((run.OUT/'execution.json').read_text())
    assert execution['status']=='COMPLETE'
    rows=[]
    for c in cases:
        assert run.source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
        with np.load(c['heldout_truth_file']) as z:
            y=z['y_target']
        for rho in run.source.novel.STRENGTHS:
            path=run.OUT/'predictions'/str(c['task_id'])/f'{family}_{rho}.npz'
            meta=json.loads(path.with_suffix('.json').read_text())
            assert run.source.sha(path)==meta['output_sha256'] and not meta['evaluation_labels_opened']
            apos,epos=meta['adaptation_positions'],meta['evaluation_positions']
            assert set(apos).isdisjoint(epos)
            original=run.source.OUT/'predictions'/track/task/str(c['task_id'])/path.name
            assert run.source.sha(original)==meta['original_prediction_sha256']
            with np.load(path) as prediction,np.load(original) as original_prediction:
                np.testing.assert_array_equal(prediction['Base'],original_prediction['Base'][epos])
                for method in ('Base','Native_probe','Shared_probe'):
                    assert len(prediction[method])==len(y) and np.isfinite(prediction[method]).all()
                    rows.append({'data_id':c['data_id'],'task_id':c['task_id'],'rho':rho,'method':method,
                                 **metrics(y,prediction[method],task,max(float(np.std(y)),1e-12))})
    frame=pd.DataFrame(rows)
    names=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
    table=frame.groupby(['data_id','method'])[names].mean().groupby('method').mean().reset_index()
    primary=names[0]
    get=lambda name:float(table.loc[table.method==name,primary].iloc[0])
    gain=get('Shared_probe')-get('Base')
    value={'status':'NUMERICAL_SUMMARY_PENDING_FULL_PROTOCOL_AUDIT','active_endpoint':[track,task,family],
           'datasets':len(cases),'conditions':4*len(cases),'table':table.to_dict(orient='records'),
           'gain':gain,'gain_vs_Native_same_supervision':get('Shared_probe')-get('Native_probe'),
           'required_gain_strictly_greater_than':.03 if track=='perturbation' else .02,
           'does_not_mark_endpoint_passed':True,'no_training_or_inference_rerun':True}
    atomic_json(run.OUT/'pre_audit_metrics.json',value)
    frame.to_csv(run.OUT/'condition_metrics_pre_audit.csv',index=False)
    print(table.to_string(index=False),flush=True)
    print('GAIN',gain,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('module')
    main(parser.parse_args().module)
