"""Launch resumable data-parallel workers; all retain outer seed zero."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
PYTHON = RUN.parents[1].parent / 'PFN-Research/.tabpfn3-venv/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    assert command
    workers = []
    state_path = RUN / 'results/tabpfn_official_comparison_v2' / f'{args.stage}_workers.json'
    for shard in range(args.workers):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(shard % 8)
        for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
            env[key] = '1'
        actual = [value.replace('{shard}', str(shard)).replace('{shards}', str(args.workers)) for value in command]
        path = RUN / 'logs' / f'tabpfn_official_v2_{args.stage}_gpu{shard}.log'
        handle = path.open('a', buffering=1)
        process = subprocess.Popen([str(PYTHON), '-u', *actual], cwd=RUN, env=env, stdout=handle, stderr=subprocess.STDOUT)
        workers.append((shard, process, handle, path))
        print('STAGE_WORKER_STARTED', args.stage, shard, process.pid, flush=True)
    while True:
        records = [{'shard': i, 'gpu': i % 8, 'pid': p.pid, 'exit_code': p.poll(), 'log': str(path.relative_to(RUN))}
                   for i, p, h, path in workers]
        done = all(row['exit_code'] is not None for row in records)
        value = {'stage': args.stage, 'status': ('COMPLETE' if all(row['exit_code'] == 0 for row in records) else 'FAILED') if done else 'RUNNING',
                 'seed': 0, 'outer_repetitions': 1, 'workers': records, 'updated_unix': time.time()}
        temp = state_path.with_suffix('.tmp.json')
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(state_path)
        if done:
            break
        time.sleep(5)
    for _, process, handle, _ in workers:
        handle.close()
    if any(process.returncode for _, process, _, _ in workers):
        raise SystemExit('Worker failed; inspect the saved logs before resuming')
    print('TABPFN_STAGE_COMPLETE', args.stage, flush=True)


if __name__ == '__main__':
    main()
