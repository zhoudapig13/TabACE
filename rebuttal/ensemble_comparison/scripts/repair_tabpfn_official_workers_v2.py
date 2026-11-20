"""Repair failed shards without duplicating their still-running peers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / 'results/tabpfn_official_comparison_v2'
PYTHON = RUN.parents[1].parent / 'PFN-Research/.tabpfn3-venv/bin/python'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--task', required=True, choices=['classification', 'regression'])
    args = parser.parse_args(); repairs = {}
    path = OUT / f'{args.task}_methods_workers.json'
    while True:
        old = json.loads(path.read_text())
        for worker in old['workers']:
            shard = worker['shard']
            if worker['exit_code'] in [None, 0] or shard in repairs:
                continue
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(worker['gpu'])
            for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
                env[key] = '1'
            log = RUN / 'logs' / f'tabpfn_official_v2_{args.task}_constant_compat_shard{shard}.log'
            handle = log.open('a')
            process = subprocess.Popen([str(PYTHON), '-u', 'scripts/run_tabpfn_official_compat_v2.py',
                '--task', args.task, '--shard', str(shard), '--shards', str(len(old['workers']))], cwd=RUN,
                env=env, stdout=handle, stderr=subprocess.STDOUT)
            repairs[shard] = process, handle, log
            print('FAILED_SHARD_COMPAT_REPAIR_STARTED', args.task, shard, process.pid, flush=True)
        status = {'task': args.task, 'repairs': {str(shard): {'pid': p.pid, 'exit_code': p.poll(), 'log': str(log.relative_to(RUN))}
                  for shard, (p, h, log) in repairs.items()}}
        (OUT / f'{args.task}_compat_repair_status.json').write_text(json.dumps(status, indent=2))
        if any(p.poll() not in [None, 0] for p, h, log in repairs.values()):
            raise SystemExit('Compatibility repair failed; inspect the dedicated shard log')
        if all(w['exit_code'] is not None for w in old['workers']) and all(p.poll() == 0 for p, h, log in repairs.values()):
            (OUT / f'{args.task}_workers_original_attempt.json').write_text(json.dumps(old, indent=2))
            for worker in old['workers']:
                if worker['shard'] in repairs:
                    p, h, log = repairs[worker['shard']]
                    worker['original_attempt'] = dict(worker)
                    worker.update(pid=p.pid, exit_code=0, log=str(log.relative_to(RUN)), constant_context_compatibility=True)
            old['status'] = 'COMPLETE'; old['compatibility_repair_applied'] = bool(repairs)
            temp = path.with_suffix('.tmp.json'); temp.write_text(json.dumps(old, indent=2)); temp.replace(path)
            print('OFFICIAL_METHOD_WORKERS_COMPLETE_WITH_REPAIRS', args.task, flush=True)
            break
        time.sleep(5)
