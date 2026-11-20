"""Run the two remaining endpoints in order, only after the previous passes."""
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

OUT = RUN / 'results/remaining_classification_shifts_v402'
SPECS = [
    ('nonlinear_sensor', 32, 'nonlinear_sensor_probe_features_v384.py',
     'nonlinear_sensor_pipeline_v390.py', 'nonlinear_sensor_pipeline_v390', 391),
    ('interaction_sampling', 35, 'interaction_sampling_probe_features_v394.py',
     'interaction_sampling_pipeline_v399.py', 'interaction_sampling_pipeline_v399', 401),
]


def status(phase, **extra):
    atomic_json(OUT/'execution.json', {'phase':phase, 'seed':0,
                'strictly_sequential_endpoints':True, 'PID':os.getpid(), **extra})
    print('REMAINING_SHIFT_STAGE', phase, flush=True)


def run(script, args, name):
    log = OUT / name
    with log.open('a') as handle:
        p = subprocess.run([sys.executable, str(RUN/'scripts'/script), *args],
                           cwd=RUN, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
    if p.returncode:
        status('FAILED', script=script, returncode=p.returncode, log=str(log))
        raise RuntimeError(f'{script} failed; see {log}')


def main():
    status('WAITING_FOR_COLUMN_DESYNC_FULL_AUDIT')
    deadline = time.monotonic()+7200
    initial = RUN/'results/column_desync_pipeline_v380/execution.json'
    while True:
        value = json.loads(initial.read_text())
        phase = value['phase']
        if phase=='COMPLETE_FROZEN':
            break
        if phase in ('FAILED','NEEDS_DIAGNOSIS','FEATURE_WAIT_TIMEOUT'):
            status('PREVIOUS_ENDPOINT_NOT_PASSED', previous_result=value)
            return
        if time.monotonic()>deadline:
            status('WAIT_TIMEOUT', does_not_restart_or_kill_other_jobs=True)
            return
        time.sleep(5)
    for family, n, feature_script, pipeline_script, pipeline_dir, registry in SPECS:
        state = json.loads(STATE.read_text())
        assert state['active_endpoint']==['shift','classification',family]
        status('EXTRACTING_FEATURES', active_endpoint=state['active_endpoint'], datasets=n,
               expected_conditions=4*n, allowed_GPUs=[0,1,2,3,4,5], reserved_GPUs=[6,7])
        state.update(current_full_experiment=None, active_datasets=n,
                     feature_extraction={'status':'RUNNING', 'script':feature_script,
                       'expected_conditions':4*n, 'GPU_workers':6, 'reserved_GPUs':[6,7]},
                     head_fitting={'status':'NOT_STARTED'}, current_full_audit=None)
        atomic_json(STATE,state)
        run(feature_script, ['launch'], f'{family}_features.log')
        status('HEADS_AND_INDEPENDENT_AUDIT', active_endpoint=['shift','classification',family])
        run(pipeline_script, [], f'{family}_pipeline.log')
        result = json.loads((RUN/'results'/pipeline_dir/'execution.json').read_text())
        if result['phase']!='COMPLETE_FROZEN':
            status('ACTIVE_ENDPOINT_NEEDS_DIAGNOSIS', active_endpoint=['shift','classification',family], result=result)
            return
        assert (RUN/f'results/frozen_endpoints_v{registry}/registry.json').exists()
    status('ALL_TWELVE_FROZEN_PENDING_FINAL_COMPLETION_AUDIT')
    audit = RUN/'scripts/audit_complete_unseen_goal_v410.py'
    if audit.exists():
        run(audit.name, [], 'full_completion_audit.log')
        status('COMPLETE_ALL_TWELVE_AUDITED')


if __name__=='__main__':
    main()
