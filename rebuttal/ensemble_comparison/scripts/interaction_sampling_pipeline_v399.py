"""Continue one active endpoint through features, heads, audit and freeze.

Failed or nonpassing stages stop the driver with the active endpoint unchanged.
This program never launches another endpoint or changes benchmark parameters.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
from prepare_data import RUN, atomic_json
from serial_endpoint_state_v371 import STATE

OUT = RUN / 'results/interaction_sampling_pipeline_v399'
FEATURES = RUN / 'results/interaction_sampling_probe_features_v394'
HEADS = RUN / 'results/interaction_sampling_supervised_probe_v395'
ENDPOINT = ['shift', 'classification', 'interaction_sampling']


def status(phase, **extra):
    value = {'active_endpoint': ENDPOINT, 'phase': phase, 'seed': 0,
             'required_gain_strictly_greater_than': .02, **extra}
    atomic_json(OUT / 'execution.json', value)
    state = json.loads(STATE.read_text())
    assert state['active_endpoint'] == ENDPOINT
    state['serial_driver'] = {'script': str(Path(__file__)), 'PID': os.getpid(), **value}
    atomic_json(STATE, state)
    print('SERIAL_STAGE', phase, flush=True)


def command(script, args, log):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    with log.open('a') as handle:
        p = subprocess.run([sys.executable, str(RUN/'scripts'/script), *args],
                           env=env, cwd=RUN, stdout=handle, stderr=subprocess.STDOUT)
    if p.returncode:
        status('FAILED', script=script, returncode=p.returncode, log=str(log))
        raise RuntimeError(f'{script} failed, see {log}')


def main():
    assert json.loads(STATE.read_text())['active_endpoint'] == ENDPOINT
    status('WAITING_FOR_FEATURE_EXTRACTION', GPU_workers=6, reserved_GPUs=[6, 7])
    deadline = time.monotonic() + 7200
    while True:
        p = FEATURES / 'full_execution.json'
        if p.exists():
            state = json.loads(p.read_text())
            if state['status'] == 'FAILED':
                status('FAILED', source=str(p))
                raise RuntimeError('Feature extraction failed')
            if state['status'] == 'COMPLETE':
                break
        if time.monotonic() > deadline:
            status('FEATURE_WAIT_TIMEOUT', does_not_restart_or_kill_feature_workers=True)
            return
        time.sleep(5)
    assert len(list((FEATURES/'features').glob('*/*.json'))) == 140
    status('FITTING_HEADS', CPU_workers=6, GPU_workers=0)
    HEADS.mkdir(parents=True, exist_ok=True)
    command('interaction_sampling_probe_parallel_v396.py', ['launch'], HEADS/'full_fit.log')
    status('INDEPENDENT_FULL_AUDIT', CPU_workers=6, GPU_workers=0)
    command('audit_interaction_sampling_parallel_v398.py', ['launch'], HEADS/'full_audit.log')
    acceptance = json.loads((HEADS/'numerical_acceptance.json').read_text())
    audit = json.loads((HEADS/'independent_audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['conditions'] == 140
    if not acceptance['active_endpoint_pass']:
        status('NEEDS_DIAGNOSIS', numerical_result=acceptance, next_endpoint_not_started=True)
        return
    status('FREEZING_PASSING_ENDPOINT', numerical_result=acceptance)
    command('freeze_remaining_shift_v389.py', ['interaction_sampling'], OUT/'freeze.log')
    # Freeze advances the global active endpoint, so this driver must not write
    # through status() again or overwrite the new endpoint's state.
    atomic_json(OUT/'execution.json', {'phase':'COMPLETE_FROZEN', 'endpoint':ENDPOINT,
                'numerical_result':acceptance, 'next_endpoint_not_started_by_this_driver':True})
    print('INTERACTION_SAMPLING_COMPLETE_FROZEN', acceptance['gain'], flush=True)


if __name__ == '__main__':
    main()
