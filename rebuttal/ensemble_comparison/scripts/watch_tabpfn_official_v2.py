"""Periodically publish live JSON/CSV and run verification when stages finish."""
import json
import os
from pathlib import Path
import subprocess
import time

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / 'results/tabpfn_official_comparison_v2'
PYTHON = RUN.parents[1].parent / 'PFN-Research/.tabpfn3-venv/bin/python'


def main():
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = '1'
    audits = {}; last_report = 0.
    while True:
        if time.monotonic() - last_report >= 60:
            with (RUN / 'logs/tabpfn_official_v2_live_report.log').open('w') as handle:
                subprocess.run([str(PYTHON), '-u', 'scripts/report_tabpfn_official_v2.py'], cwd=RUN, env=env,
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
            last_report = time.monotonic()
        for task in ['classification', 'regression']:
            path = OUT / f'{task}_methods_workers.json'
            if task in audits or not path.exists():
                continue
            if json.loads(path.read_text())['status'] == 'COMPLETE':
                # Refresh global averages before checking this completed task.
                with (RUN / 'logs/tabpfn_official_v2_live_report.log').open('w') as handle:
                    subprocess.run([str(PYTHON), '-u', 'scripts/report_tabpfn_official_v2.py'], cwd=RUN, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
                handle = (RUN / 'logs' / f'tabpfn_official_v2_audit_{task}.log').open('w')
                process = subprocess.Popen([str(PYTHON), '-u', 'scripts/audit_tabpfn_official_v2.py', task], cwd=RUN,
                                           env=env, stdout=handle, stderr=subprocess.STDOUT)
                audits[task] = process, handle
                print('INDEPENDENT_AUDIT_STARTED', task, process.pid, flush=True)
        workflow = OUT / 'regression_oligo_workflow.json'
        if 'oligo' not in audits and workflow.exists() and json.loads(workflow.read_text())['status'] == 'COMPLETE':
            handle = (RUN / 'logs/tabpfn_official_v2_audit_oligo.log').open('w')
            process = subprocess.Popen([str(PYTHON), '-u', 'scripts/audit_tabpfn_official_v2.py', 'oligo'], cwd=RUN,
                                       env=env, stdout=handle, stderr=subprocess.STDOUT)
            audits['oligo'] = process, handle
            print('INDEPENDENT_AUDIT_STARTED', 'oligo', process.pid, flush=True)
        for name, (process, handle) in audits.items():
            if process.poll() not in [None, 0]:
                raise RuntimeError(f'Audit failed: {name}; inspect its dedicated log')
        state = {'status': 'RUNNING', 'updated_unix': time.time(),
            'audits': {name: {'pid': p.pid, 'exit_code': p.poll()} for name, (p, h) in audits.items()}}
        if len(audits) == 3 and all(p.poll() == 0 for p, h in audits.values()):
            subprocess.run([str(PYTHON), 'scripts/report_tabpfn_official_v2.py'], cwd=RUN, env=env, check=True)
            subprocess.run([str(PYTHON), 'scripts/audit_tabpfn_official_v2.py', 'final'], cwd=RUN, env=env, check=True)
            state['status'] = 'COMPLETE_AND_AUDITED'
        path = OUT / 'verification_workflow.json'; temp = path.with_suffix('.tmp.json')
        temp.write_text(json.dumps(state, indent=2)); temp.replace(path)
        if state['status'] == 'COMPLETE_AND_AUDITED':
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
