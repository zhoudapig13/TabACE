"""Redistribute not-yet-started datasets from the last serial CPU shard.

The original process is stopped temporarily while strictly later datasets are
computed. Its current incomplete dataset is excluded from auxiliary work; the
original process is always resumed before this driver exits. No fitter changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import column_desync_supervised_probe_v375 as run
from prepare_data import RUN, atomic_json

OUT = RUN / 'results/column_desync_tail_v392'


def worker(tid):
    cases = run.features.schedule.endpoint.endpoint_cases()
    chosen = [c for c in cases if c['task_id'] == tid]
    assert len(chosen) == 1
    run.features.schedule.endpoint.endpoint_cases = lambda: chosen
    original = run.atomic_json
    def write(path, value):
        if path == run.OUT/'execution.json':
            path = OUT / f'dataset_{tid}_execution.json'
        return original(path, value)
    run.atomic_json = write
    run.main(pilot=False, require_complete=True)


def launch():
    candidates = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = [b.decode() for b in (p/'cmdline').read_bytes().split(b'\0') if b]
            if (len(args) > 2 and args[1].endswith('column_desync_probe_parallel_v376.py')
                    and 'worker' in args and '--shard' in args
                    and args[args.index('--shard')+1] == '1'):
                candidates.append(int(p.name))
        except (OSError, UnicodeError):
            continue
    if not candidates:
        atomic_json(OUT/'execution.json', {'status':'NO_LIVE_WORKER_TO_REBALANCE'})
        return
    assert len(candidates) == 1
    pid = candidates[0]
    stopped = False
    try:
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        # Confirm that execution has stopped before inferring its in-flight case.
        import time
        for _ in range(100):
            if (Path('/proc')/str(pid)/'stat').read_text().split()[2] in ('T', 't'):
                break
            time.sleep(.01)
        else:
            raise RuntimeError('Could not confirm stopped original worker')
        cases = [c for i,c in enumerate(run.features.schedule.endpoint.endpoint_cases()) if i%6 == 1]
        complete = lambda c: all((run.OUT/'predictions'/str(c['task_id'])/f'column_desynchronization_{rho}.json').exists() for rho in run.source.novel.STRENGTHS)
        first = next((i for i,c in enumerate(cases) if not complete(c)), len(cases))
        # Excluding the first incomplete case avoids concurrent writes even if
        # the original worker was stopped partway through a fit or file write.
        todo = [c['task_id'] for c in cases[first+1:] if not complete(c)]
        plan = {'original_PID':pid, 'original_shard':1,
                'inflight_excluded_task_id':cases[first]['task_id'] if first<len(cases) else None,
                'auxiliary_task_ids':todo, 'same_fitter_sha256':run.source.sha(run.__file__),
                'GPU_workers':0, 'CPU_workers':min(6,len(todo)),
                'benchmark_and_labels_and_parameters_unchanged':True,
                'original_process_resumed_on_exit':True}
        atomic_json(OUT/'plan.json', plan)
        atomic_json(OUT/'execution.json', {'status':'RUNNING', **plan})
        def execute(tid):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''
            log = OUT / f'dataset_{tid}.log'
            with log.open('a') as handle:
                p = subprocess.run([sys.executable, __file__, 'worker', '--task-id', str(tid)],
                                   env=env, stdout=handle, stderr=subprocess.STDOUT)
            return {'task_id':tid, 'returncode':p.returncode, 'log':str(log)}
        if todo:
            with ThreadPoolExecutor(max_workers=min(6,len(todo))) as pool:
                results = list(pool.map(execute, todo))
        else:
            results = []
        ok = all(r['returncode']==0 for r in results)
        atomic_json(OUT/'execution.json', {'status':'COMPLETE' if ok else 'FAILED',
                    'results':results, 'original_PID':pid, 'original_will_resume':True})
        assert ok
    finally:
        if stopped:
            os.kill(pid, signal.SIGCONT)
            print('ORIGINAL_WORKER_RESUMED',pid,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['launch','worker'])
    parser.add_argument('--task-id',type=int)
    args = parser.parse_args()
    launch() if args.phase=='launch' else worker(args.task_id)
