"""Resume validation when each GPU's query shard exits, then fit complete banks.

This supervisor never marks the research goal complete and never tunes on query
scores. Exit codes, expected bank counts, and downstream failures are recorded.
"""
import json
import os
from pathlib import Path
import subprocess
import time

RUN = Path(__file__).resolve().parent.parent
OUT = RUN / 'results/tabace_context_reproduction_v28'
PYTHON = RUN / '.venv/bin/python'
SCRIPTS = RUN / 'scripts'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ')[1][0] != 'Z'


def launch(script, arguments, logfile, gpu=None):
    env = os.environ | {k: '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with logfile.open('a') as stream:
        process = subprocess.Popen([str(PYTHON), '-u', str(SCRIPTS / script), *arguments],
                                   env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    return process


def main():
    original = json.loads((OUT / 'query_higher_execution.json').read_text())
    statefile = OUT / 'higher_supervisor.json'
    assert not statefile.exists(), 'Do not duplicate an existing supervisor'
    state = dict(status='RUNNING', supervisor_pid=os.getpid(), validation=[], fusion=[])
    validation = {}
    fits = {}
    finished = set()
    budgets = [1024, 4096, 8192]
    expected = {('classification', 'clean'): 72, ('classification', 'five'): 1440,
                ('classification', 'seven'): 980, ('regression', 'clean'): 35,
                ('regression', 'five'): 700, ('regression', 'seven'): 924}
    while True:
        for row in original:
            gpu = row['gpu']
            if gpu not in validation and not alive(row['pid']):
                log = RUN / 'logs' / f'tabace_validation_v28_higher_gpu{gpu}.log'
                proc = launch('tabace_validation_banks_v28.py',
                              ['--budgets', '1024,4096,8192', '--shard', str(gpu), '--shards', '8'], log, gpu)
                validation[gpu] = proc
                state['validation'].append(dict(gpu=gpu, pid=proc.pid, log=str(log)))
            if gpu in validation:
                status = validation[gpu].poll()
                if status is not None:
                    state['validation'][next(i for i,r in enumerate(state['validation']) if r['gpu']==gpu)]['exit_code'] = status
        for budget in budgets:
            for (task, block), count in expected.items():
                key = (budget, task, block)
                if key in fits:
                    proc, record = fits[key]
                    code = proc.poll()
                    if code is not None:
                        record['exit_code'] = code
                        finished.add(key)
                    continue
                if sum(p.poll() is None for p, _ in fits.values()) >= 6:
                    continue
                complete = True
                for role in ['query', 'validation']:
                    folder = OUT / 'banks' / role / str(budget) / task / block
                    files = [p for p in folder.glob('*/*.json') if '.tmp.' not in p.name]
                    if len(files) != count or any(not p.with_suffix('.npz').exists() for p in files):
                        complete = False
                        break
                if not complete:
                    continue
                log = RUN/'logs'/f'tabace_fusion_v28_{budget}_{task}_{block}.log'
                proc = launch('tabace_fusion_v28.py', ['--budget', str(budget), '--task', task, '--block', block], log)
                record = dict(budget=budget, task=task, block=block, pid=proc.pid, log=str(log))
                state['fusion'].append(record)
                fits[key] = proc, record
        state['updated_unix'] = time.time()
        save(statefile, state)
        if len(finished) == 18:
            state['status'] = 'INFERENCE_AND_FUSION_FINISHED_REQUIRES_GOAL_AUDIT'
            save(statefile, state)
            log = RUN/'logs'/'tabace_v28_higher_goal_audit.log'
            proc = launch('audit_tabace_goals_v28.py', ['report'], log)
            state['audit_exit_code'] = proc.wait()
            save(statefile, state)
            return
        if len(validation) == 8 and all(p.poll() is not None for p in validation.values()) and len(finished)==len(fits):
            state['status'] = 'STOPPED_FOR_INCOMPLETE_BANK_OR_FAILED_WORKER'
            save(statefile, state)
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
