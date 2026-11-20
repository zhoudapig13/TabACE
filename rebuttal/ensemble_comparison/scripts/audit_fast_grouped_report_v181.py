"""Sealed-policy replay plus direct verification of fast scoring on real data."""
import argparse
import importlib
import json
import numpy as np
import audit_shared_adaptation_v140 as replay


def main(module):
    run = importlib.import_module(module)
    plan = json.loads((run.OUT/'plan.json').read_text())
    for key, implementation in [('underlying_group_risk_sha256', run.calibration.grouped),
        ('underlying_heads_sha256', run.calibration.original), ('fast_metrics_sha256', run.calibration.fast)]:
        assert run.original.sha(implementation.__file__) == plan[key]
    original_metric = run.metrics
    calls = 0
    def checked(y, prediction, task, scale):
        nonlocal calls
        values = original_metric(y, prediction, task, scale)
        expected = [values['auc'] if values['auc'] is not None else values['accuracy'], values['accuracy']] \
            if task == 'classification' else [values['r2'], -values['nrmse']]
        actual = run.calibration.fast.metric({'task': task, 'y': y}, prediction)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        calls += 1
        return values
    run.metrics = checked
    replay.audit(module)
    path = run.OUT/'independent_replay_audit.json'
    result = json.loads(path.read_text()) | {'fast_metrics_checked_against_original_on_real_predictions': calls,
        'underlying_implementation_hashes_checked': True}
    run.atomic_json(path, result)
    print('FAST_METRIC_REPLAY_PASS', calls, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('module')
    main(parser.parse_args().module)
