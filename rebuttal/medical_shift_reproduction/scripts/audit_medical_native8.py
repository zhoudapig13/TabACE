"""Check a cached Native-8 Base against an independently loaded checkpoint."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import hashlib
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tabicl import TabICLClassifier
import tune_medical_development as main
import tune_medical_local as local
from tune_expanded_medical import ExpandedProbability
from run_public_shift import CHECKPOINT, write_json


def audit():
    main.DATA=main.ROOT/'data/processed/expanded_medical_development_v18';local.DATA=main.DATA
    w=ExpandedProbability('sepsis_A_to_B', 0, 'raw')
    w.reference_domain('id')
    f=json.loads((main.ROOT/'results/medical_common_native8_v24/sepsis_A_to_B/frozen_selection.json').read_text())
    conf=f['configurations']['id']['Base']
    rows=w.rows(conf['context_fraction'],np.ones(len(w.y)),0)
    columns=np.arange(w.X.shape[1]);context=w.X[rows].copy();y=w.y[rows]
    key=hashlib.sha256(context.tobytes()+y.tobytes()+columns.tobytes()).hexdigest()
    cache=main.DATA/w.task/'member_cache_known_domain_native8'/(key+'.npz')
    assert cache.is_file(), 'The evaluated control must use an existing Native-8 cache'
    with np.load(cache) as z:
        cached=z['p'];assert np.array_equal(z['rows'],rows)
    fresh=TabICLClassifier(n_estimators=8,device='cuda:0',model_path=str(CHECKPOINT),
        allow_auto_download=False,n_jobs=1,batch_size=1,random_state=0)
    fresh.fit(context,y)
    actual=np.concatenate([fresh.predict_proba(w.queries[a:a+1024])[:,1]
                           for a in range(0,len(w.queries),1024)])
    delta=float(np.max(np.abs(actual-cached)))
    disagreements=int(np.count_nonzero((actual>=conf['policy']['threshold'])!=(cached>=conf['policy']['threshold'])))
    report=dict(task=w.task,domain='id',native_estimators=8,context_rows=len(rows),
        queries=len(actual),max_abs_probability_difference=delta,classification_disagreements=disagreements,
        fresh_checkpoint_load_compared_to_cached_predictions=True,test_truth_opened=False,
        passed=bool(delta<1e-5 and disagreements==0))
    write_json(main.ROOT/'results/native8_checkpoint_consistency_audit.json',report)
    print(report,flush=True)
    assert report['passed']


if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):audit()
