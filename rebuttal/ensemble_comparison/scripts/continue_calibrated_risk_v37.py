"""Complete the predeclared TRAIN-only regression calibration study."""
import json
import os
import time
from pathlib import Path
from continue_tabace_v28 import launch, save, RUN

ROOT = RUN / 'results/tabace_calibrated_risk_v37'
SOURCES = {'fixed_pool': 'tabace_context_adaptation_v29', 'full_pool': 'tabace_full_pool_v34'}
EXPECTED = {'clean': 35, 'five': 700, 'seven': 924}

if __name__ == '__main__':
    assert json.loads((ROOT / 'grouped_cv_audit.json').read_text())['status'] == 'PASS'
    path = ROOT / 'execution.json'
    assert not path.exists(), 'Avoid duplicate controllers'
    state = dict(status='RUNNING', pid=os.getpid(), fits=[])
    children = {}
    while True:
        for proc, row in children.values():
            if proc.poll() is not None:
                row['exit_code'] = proc.returncode
        for version, dirname in SOURCES.items():
            source = RUN / 'results' / dirname
            for budget in [256, 1024, 4096, 8192]:
                for block, count in EXPECTED.items():
                    key = version, budget, block
                    if key in children:
                        continue
                    completion = ROOT / version / 'policies' / str(budget) / 'regression' / block / 'completion.json'
                    if completion.exists():
                        continue
                    if sum(p.poll() is None for p, _ in children.values()) >= 4:
                        continue
                    if any(sum(p.with_suffix('.npz').exists() for p in (source / 'banks' / role / str(budget) / 'regression' / block).glob('*/*.json') if '.tmp.' not in p.name) != count for role in ['validation', 'query']):
                        continue
                    log = RUN / 'logs' / f'calibrated_risk_v37_{version}_{budget}_{block}.log'
                    proc = launch('calibrated_risk_fusion_v37.py', ['--source', str(source), '--out', str(ROOT / version), '--budget', str(budget), '--block', block], log)
                    row = dict(version=version, budget=budget, block=block, pid=proc.pid, log=str(log))
                    children[key] = proc, row
                    state['fits'].append(row)
        done = sum((ROOT / version / 'policies' / str(budget) / 'regression' / block / 'completion.json').exists() for version in SOURCES for budget in [256,1024,4096,8192] for block in EXPECTED)
        state['completed_groups'] = done
        state['updated_unix'] = time.time()
        save(path, state)
        if done == 24:
            state['status'] = 'COMPLETE_REQUIRES_METRIC_AUDIT'
            save(path, state)
            break
        if any(row.get('exit_code', 0) != 0 for _, row in children.values()):
            state['status'] = 'WORKER_FAILURE_REQUIRES_REPAIR'
            save(path, state)
            raise SystemExit(1)
        time.sleep(20)
