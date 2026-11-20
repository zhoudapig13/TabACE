"""Reuse audited official-v3 Native6 raw predictions; never use adapted outputs."""
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import RUN, atomic_json
from tabpfn_official_benchmark_v2 import OUT as OLD, tasks, conditions, path_for
from run_tabicl import metrics

OUT = RUN / 'results/tabpfn_v3_native6_seed0_v19'


@lru_cache(None)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import tabpfn
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    import importlib.metadata

    probe = json.loads((OUT / 'official_default_probe.json').read_text())
    assert len(probe) == 6 and all(r['max_error_vs_saved_raw'] < 1e-6 for r in probe)
    assert all(r['resolved_estimators'] == '6' and r['resolved_temperature'] == '0.9' for r in probe)
    dependencies = json.loads((OLD / 'dependency_provenance.json').read_text())
    for path, expected in dependencies['checkpoint_sha256'].items():
        assert sha(path) == expected
    for path, expected in dependencies['dependencies_sha256'].items():
        assert sha(RUN / path) == expected
    frozen = {str(p.relative_to(RUN)): sha(p) for p in OLD.glob('frozen_*.json')}
    rows, contexts, analytical, native_calls = [], {}, 0, 0
    for task in ['classification', 'regression']:
        for track in ['five', 'seven']:
            for meta in tasks(task, track):
                grid = ([('original', 0)] if track == 'five' else []) + conditions(track)
                for family, rho in grid:
                    block = 'clean' if rho == 0 else track
                    cfg = json.loads((OLD / f'frozen_{task}_{block}.json').read_text())['settings']
                    base_path = path_for(task, block, cfg['name'], meta, family, rho)
                    base_meta = json.loads(base_path.with_suffix('.json').read_text())
                    path = OLD / 'official_predictions' / task / track / str(meta['task_id']) / f'{family}_{rho}__Native6.npz'
                    m = json.loads(path.with_suffix('.json').read_text())
                    assert m['status'] == 'COMPLETE' and m['seed'] == 0
                    assert not m['backbone_gradient_finetuning'] and not m['benchmark_query_labels_used_for_training_or_selection']
                    assert m['input_hashes'] == base_meta['input_hashes']
                    if m['method_details'].get('analytical_case'):
                        analytical += 1
                    else:
                        assert m['method_details']['native_estimators'] == 6
                        assert m['measured_PFN_calls']['sum_n_estimators'] == 6
                        native_calls += 1
                    # Older metadata retains the original pool's source_rows.
                    # Recover the actual inference budget from the frozen cap or
                    # the seven-regression materializer's explicit row count.
                    actual_rows = m['source_rows']
                    if task == 'regression' and track == 'five':
                        clean_cfg = json.loads((OLD / 'frozen_regression_clean.json').read_text())['settings']
                        actual_rows = min(meta['source_rows'], clean_cfg['source_cap'])
                    elif task == 'regression':
                        actual_rows = m['eligible_source_rows']
                    contexts.setdefault((task, block), set()).add(actual_rows)
                    with np.load(path) as z, np.load(base_path) as b:
                        assert np.array_equal(z['original_Base'], b['prediction'])
                        assert np.array_equal(z['y_target'], b['y_target'])
                        for name, key in [('Native-1', 'original_Base'), ('Native-6', 'raw_prediction')]:
                            p = z[key]
                            assert np.isfinite(p).all()
                            if task == 'classification':
                                assert (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-6)
                            observed = metrics(z['y_target'], p, task, m['scale'])
                            expected = base_meta['metrics'] if name == 'Native-1' else m['raw_metrics']
                            for metric, value in observed.items():
                                if value is not None:
                                    assert abs(value - expected[metric]) < 1e-10
                            rows.append({'task': task, 'method': name, 'block': block,
                                         'task_id': meta['task_id'], 'family': family, 'rho': rho,
                                         'context_rows': actual_rows, 'seed': 0,
                                         'artifact': str(path.relative_to(RUN)), 'prediction_key': key,
                                         **observed})
                print('VERIFIED', task, track, meta['task_id'], flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == 8302
    assert not frame.duplicated(['task', 'method', 'block', 'task_id', 'family', 'rho']).any()
    summaries = []
    for (task, method, block), group in frame.groupby(['task', 'method', 'block'], sort=False):
        expected = {'classification': {'clean': 72, 'five': 1440, 'seven': 980},
                    'regression': {'clean': 35, 'five': 700, 'seven': 924}}[task][block]
        assert len(group) == expected
        keys = ['accuracy', 'auc'] if task == 'classification' else ['r2', 'nrmse']
        summaries.append({'task': task, 'method': method, 'block': block,
                          'conditions': len(group), 'datasets': group.task_id.nunique(),
                          'seed': 0, **{key: float(group[key].mean()) for key in keys}})
    frame.to_csv(OUT / 'detail.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT / 'averages.csv', index=False)
    atomic_json(OUT / 'summary.json', {'status': 'COMPLETE', 'rows': summaries,
                'seed': 0, 'checkpoint_version': 'v3', 'raw_only': True,
                'extra_supervision': False, 'adaptation': False, 'finetuning': False,
                'benchmark_is_frozen_internal_reconstruction': True,
                'exact_original_paper_protocol': False})
    atomic_json(OUT / 'audit.json', {'status': 'PASS', 'conditions_checked': 4151,
                'metrics_rows': len(frame), 'nondegenerate_native6_conditions': native_calls,
                'analytical_constant_conditions': analytical,
                'same_input_hashes_for_Base_and_Native6': True,
                'native_outputs_used': 'raw_prediction, never adapted_prediction',
                'checkpoint_sha256': dependencies['checkpoint_sha256'],
                'frozen_benchmark_sha256': frozen, 'official_fresh_inference_checks': probe,
                'tabpfn_package_version': importlib.metadata.version('tabpfn'),
                'official_class_sources': {inspect.getfile(c): sha(inspect.getfile(c))
                                          for c in [TabPFNClassifier, TabPFNRegressor]},
                'context_rows': {f'{t}/{b}': sorted(v) for (t, b), v in contexts.items()},
                'Markdown_written': False})
    print('FULL_NATIVE6_AUDIT_PASS', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
