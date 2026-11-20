"""Isolate Native8 replay drift without selecting on query outcomes."""
import json
import os
import time
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import numpy as np
import torch
import native_context_specific_v67 as native
import tabace_validation_banks_v28 as validation
from prepare_data import atomic_json, protocol, RUN
from run_tabicl import Predictor

OUT = RUN / 'results/tabace_native_replay_diagnosis_v70'


def inputs(tid, budget):
    meta = next(m for m in native.RAW.frozen.cohort('classification', 'clean') if int(m['task_id']) == tid)
    pool = native.RAW.scaling.prepare('classification', 'five', meta)
    split = validation.validation_split('classification', 'five', meta, pool)
    data, _ = validation.materialize('classification', 'clean', meta, 'original', 0, budget, pool, split)
    origin = native.OUT / 'banks/validation' / str(budget) / f'{tid}.npz'
    row = json.loads(origin.with_suffix('.json').read_text())
    assert {k: protocol.array_sha256(v) for k, v in data.items()} == row['input_hashes']
    with np.load(origin) as z:
        saved = z['prediction']
    return data, saved


def run():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    data, saved = inputs(167124, 4096)
    small, _ = inputs(167124, 1024)
    prior = []
    records = []
    predictor = Predictor('classification', 'cuda:0')

    def call(name, current, d, compare=True):
        start = time.monotonic()
        p, _ = current(d['X_source'], d['y_source'], d['X_target'], members=8, random_state=0, n_classes=10)
        entry = dict(name=name, seconds=time.monotonic()-start,
                     max_error_vs_saved=float(np.max(np.abs(p-saved))) if compare else None,
                     output_hash=protocol.array_sha256(p),
                     class_shuffles=str(current.models[8].ensemble_generator_.class_shuffles_),
                     feature_shuffles_hash=protocol.array_sha256(np.asarray(next(iter(current.models[8].ensemble_generator_.feature_shuffles_.values())))),
                     model_training=bool(current.models[8].model_.training),
                     tf32=torch.backends.cuda.matmul.allow_tf32)
        if compare:
            entry['max_error_vs_previous_calls'] = [float(np.max(np.abs(p-old))) for old in prior]
            prior.append(p.copy())
        native.RAW.scaling.npwrite(OUT / f'{name}.npz', prediction=p)
        records.append(entry)
        atomic_json(OUT / 'diagnosis.json', dict(status='RUNNING', task_id=167124, role='TRAIN_validation_only',
                                               budget=4096, tolerance_unchanged=1e-5, records=records,
                                               predictions_in_v67_modified=False, query_labels_used=False))
        print('NATIVE_REPLAY_DIAGNOSTIC', json.dumps(entry), flush=True)

    call('first_fresh', predictor, data)
    call('same_instance_repeat', predictor, data)
    call('intervening_1024', predictor, small, False)
    call('after_intervening_1024', predictor, data)
    del predictor
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    np.random.seed(0)
    call('new_instance_reset_seed', Predictor('classification', 'cuda:0'), data)
    d = json.loads((OUT / 'diagnosis.json').read_text())
    d.update(status='COMPLETE_REQUIRES_ANALYSIS', time_unix=time.time())
    atomic_json(OUT / 'diagnosis.json', d)


if __name__ == '__main__':
    run()
