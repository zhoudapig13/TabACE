"""Sixteen resumable workers on eight GPUs; no Shared jobs."""
import os
import subprocess
import sys
from prepare_data import RUN

workers = []
for shard in range(16):
    env = os.environ | {'CUDA_VISIBLE_DEVICES': str(shard % 8), 'OMP_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    log = (RUN/'logs'/f'canonical_regression_{shard}.log').open('a')
    worker = subprocess.Popen([sys.executable, str(RUN/'scripts/run_canonical_regression.py'),
                               '--shard', str(shard), '--shards', '16'],
                              env=env, stdout=log, stderr=subprocess.STDOUT)
    workers.append((shard, worker, log))
    print('LAUNCHED', shard, worker.pid, flush=True)
failed = []
for shard, worker, log in workers:
    code = worker.wait()
    log.close()
    print('FINISHED', shard, code, flush=True)
    if code:
        failed.append(shard)
if failed:
    raise SystemExit('Failed workers: '+str(failed))
