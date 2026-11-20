"""Serial gate and input manifest for interaction-sampling classification."""
import json
import sys
import numpy as np
import task_adaptation_bank_v312 as oldbank
from prepare_data import RUN, ROOT, atomic_json
from serial_endpoint_state_v371 import verify_registry, STATE

source = oldbank.source
OUT = RUN / 'results/interaction_sampling_schedule_v393'
ENDPOINT = ('shift', 'classification', 'interaction_sampling')
endpoint = sys.modules[__name__]


def endpoint_cases():
    cases = json.loads((oldbank.split.OUT / 'case_manifest.json').read_text())
    return [c for c in cases if (c['track'], c['task']) == ENDPOINT[:2] and ENDPOINT[2] in c['families']]


def verify_frozen():
    return verify_registry(RUN / 'results/frozen_endpoints_v391/registry.json')['registry_sha256']


def prepare():
    state = json.loads(STATE.read_text())
    assert state['active_endpoint'] == list(ENDPOINT) and state['frozen_endpoint_count'] >= 11
    frozen_sha = verify_frozen()
    cases = endpoint_cases()
    assert len(cases) == 35
    values = {}
    for c in cases:
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            y = z['y_source'].astype(int)
        with np.load(c['reference_file']) as z:
            r = z['y_train_reference'].astype(int)
        K = int(y.max()) + 1
        p = np.bincount(y, minlength=K) / len(y)
        q = np.bincount(r, minlength=K) / len(r)
        assert len(p) == len(q)
        values[c['task_id']] = float(.5 * np.abs(p-q).sum())
    ordered = sorted(cases, key=lambda c: (values[c['task_id']], c['task_id']))
    plan = {'active_endpoint': list(ENDPOINT), 'seed': 0, 'cases': 35, 'conditions': 140,
            'pilot_task_ids': [ordered[i]['task_id'] for i in (0, len(ordered)//2, len(ordered)-1)],
            'pilot_selection': 'minimum, median and maximum observable source/reference class-prior discrepancy',
            'frozen_registry_sha256': frozen_sha,
            'GPU_policy': json.loads(oldbank.training.implementation.GPU_POLICY.read_text()),
            'script_sha256': source.sha(__file__), 'Markdown_written': False}
    p = OUT / 'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0


if __name__ == '__main__':
    prepare()
