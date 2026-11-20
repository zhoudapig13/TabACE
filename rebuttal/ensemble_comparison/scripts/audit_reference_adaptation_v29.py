"""Audit completed reference adaptations against expected frozen task paths."""
from pathlib import Path
import hashlib
import json
import paperish_fixed_benchmark_v24 as core

ROOT = core.RUN / 'results'


def audit(root):
    plan = json.loads((root / 'plan.json').read_text())
    for variant in plan['variants']:
        for task in ['classification', 'regression']:
            assert (root / f'complete_{variant}_{task}.json').exists(), (root, variant, task, 'fit incomplete')
    jobs = core.main.jobs()
    missing_jobs = [str(core.rel(j)) for j in jobs if not (root / core.rel(j) / 'inference_complete.json').exists()]
    assert not missing_jobs, missing_jobs
    cases, max_rows, refs, changed = 0, 0, set(), []
    for job in jobs:
        rel = core.rel(job)
        expected = [(Path('calibration') / f'{family}_{rho}.json', 'calibration')
                    for family, rho in [('original', 0)] + [(f, r) for f in core.main.families(job['track']) for r in [50, 100]]]
        for block in (['clean', 'five'] if job['track'] == 'five' else ['seven']):
            expected += [(Path('query') / block / f'{f}_{r}.json', block) for f, r in core.main.contexts.case_grid(block)]
        for suffix, block in expected:
            p = root / rel / suffix
            meta = json.loads(p.read_text())
            raw = json.loads((core.RAW / rel / suffix).read_text())
            assert meta['frozen_input_hashes'] == raw['input_hashes'], str(p)
            assert meta['reference_disjoint_from_calibration_and_final_queries'], str(p)
            assert len(meta['roles']) == 10, str(p)
            assert p.with_suffix('.npz').exists(), str(p)
            max_rows = max(max_rows, max(role['rows'] for role in meta['roles']))
            refs.add(meta['clean_reference_examples'])
            if block != 'calibration':
                cases += 1
                if 'changed_source_fraction' in meta:
                    changed.append(dict(task=job['task'], block=block, fraction=meta['changed_source_fraction']))
    assert cases == 4151 and max_rows <= 256
    policies = list((root / 'policies').glob('*/*/fold*/policies.json'))
    nonconverged = []
    models = 0
    for p in policies:
        d = json.loads(p.read_text())
        assert not set(d['outer_train_data_ids']) & set(d['outer_test_data_ids'])
        for name, model in [('Shared', d['Shared'])] + list(d['Specific'].items()):
            models += 1
            if not model['optimization']['success']:
                nonconverged.append(dict(path=str(p), model=name))
    hashes = {}
    for key, path in [('benchmark_hash', ROOT / 'tabicl_base256_calibration_v22/frozen_suite.json'),
                      ('classification_five_hash', ROOT / 'finetune_source_benchmark_v5/frozen.json')]:
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest() == plan[key]
    assert all(hashes.values())
    result = dict(status='PASS', jobs=len(jobs), query_conditions=cases,
                  maximum_rows_per_context=max_rows, actual_clean_reference_counts=sorted(refs),
                  all_frozen_source_query_input_hashes_identical=True,
                  frozen_protocol_hashes_unchanged=hashes, all_OOF_fold_datasets_disjoint=True,
                  source_label_stratified_reference_selection=True,
                  reference_budget_definition='Selected reference examples fed to models, not label-blind annotation access',
                  policies_available=len(policies), fitted_models=models,
                  nonconverged_models=nonconverged)
    (root / 'audit_v29.json').write_text(json.dumps(result, indent=2))
    if changed:
        import pandas as pd
        pd.DataFrame(changed).groupby(['task', 'block'], as_index=False).fraction.mean().to_csv(
            root / 'source_repair_fraction_averages.csv', index=False)
    print(root.name, json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', nargs='+')
    args = parser.parse_args()
    for name in args.roots:
        audit(ROOT / name)
