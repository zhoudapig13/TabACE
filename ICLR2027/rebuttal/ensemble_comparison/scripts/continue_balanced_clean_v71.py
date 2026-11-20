"""Run the changed-source smoke check, then all source banks and frozen fits."""
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_data import RUN, atomic_json

OUT = RUN/'results/tabace_balanced_clean_calibration_v71'
SCRIPT = RUN/'scripts/balanced_clean_calibration_v71.py'
PYTHON = str(RUN/'.venv/bin/python')


def verify_smoke():
    import numpy as np
    paths = list((OUT/'source').glob('*/*.npz'))
    assert paths
    for path in paths:
        info = json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            n, k = z['base'].shape
            assert z['members'].shape == (21,n,k)
            assert len(np.unique(z['validation_original_ids'])) == n
            assert np.isfinite(z['members']).all()
            assert np.allclose(z['members'].sum(-1), 1., atol=1e-12)
            assert np.all(z['member_label_counts'].sum(1) <= info['requested_context'])
            held_ids=z['validation_original_ids'].copy()
        split_path = Path(info['calibration_partition'])
        with np.load(split_path) as split:
            assert not np.intersect1d(split['validation_original_ids'],split['source_original_ids']).size
            assert np.array_equal(held_ids,split['validation_original_ids'])
    atomic_json(OUT/'smoke_audit.json',dict(status='PASS',cases=len(paths),all21_probabilities_finite_and_normalized=True,
        unique_source_calibration_rows=True,source_original_IDs_disjoint=True,per_branch_row_caps=True))


def run():
    assert json.loads((OUT/'partition_audit.json').read_text())['status']=='PASS'
    state=dict(status='SMOKE_RUNNING',pid=os.getpid(),gpu=4,source_workers=[],fits=[])
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def launch(args,name,gpu=True):
        log=RUN/'logs'/f'balanced_clean_v71_{name}.log'
        with log.open('w') as handle:
            child=subprocess.Popen([PYTHON,'-u',str(SCRIPT)]+args,stdout=handle,stderr=subprocess.STDOUT,env=env if gpu else os.environ.copy())
        return child,dict(pid=child.pid,log=str(log),arguments=args)
    def store():
        state['updated_unix']=time.time();atomic_json(OUT/'execution.json',state)
    child,job=launch(['source','--limit','1'],'smoke');state['smoke']=job;store()
    while child.poll() is None:
        time.sleep(5)
    job['exit_code']=child.returncode
    if child.returncode:
        state['status']='SMOKE_FAILED';store();return
    verify_smoke();state['status']='SOURCE_INFERENCE_RUNNING'
    child,job=launch(['source'],'source_gpu4');state['source_workers'].append(job);store()
    while child.poll() is None:
        time.sleep(10)
    job['exit_code']=child.returncode
    if child.returncode:
        state['status']='SOURCE_INFERENCE_FAILED';store();return
    state['status']='SOURCE_COMPLETE_FITTING';store();active=[]
    for budget in [1024,4096,8192]:
        proc,job=launch(['fit','--budget',str(budget)],f'fit_{budget}',gpu=False)
        job['budget']=budget;state['fits'].append(job);active.append((proc,job))
    store()
    while any(proc.poll() is None for proc,_ in active):
        time.sleep(10)
    for proc,job in active:
        job['exit_code']=proc.returncode
    state['status']='FIT_FAILED' if any(proc.returncode for proc,_ in active) else 'ALL_GROUPS_COMPLETE_REQUIRES_RAW_AND_FULL48_AUDIT'
    store()


if __name__=='__main__':
    run()
