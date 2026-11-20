"""Precompute outcome-free evidence before fitting a new fusion gate."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
import shared_dual_canonical_banks_v182 as bank
import domain_evidence_v186 as evidence
from prepare_data import RUN, atomic_json, protocol

OUT = RUN/'results/shared_domain_evidence_v187'


def worker(c):
    source = bank.source
    assert source.sha(c['input_file']) == c['input_sha256']
    with np.load(c['input_file']) as data:
        assert 'y_target' not in data.files
        X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
    n = 0
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
        if rho:
            X, _, Q, _ = source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
        else:
            X, Q = X0, Q0
        values = evidence.descriptors(X, Q, numeric)
        dest = OUT/'features'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
        source.npwrite(dest, domain_evidence=values)
        atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
            {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
            'source_input_sha256': c['input_sha256'], 'observed_source_X_sha256': protocol.array_sha256(X),
            'observed_query_X_sha256': protocol.array_sha256(Q),
            'task_labels_or_Base_predictions_used_for_descriptors': False,
            'family_or_severity_received_by_descriptors': False})
        n += 1
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'status': 'FEATURES_FROZEN_BEFORE_NEW_FUSION_CALIBRATION', 'seed': 0,
        'source_bank': str(bank.OUT), 'source_bank_plan_sha256': bank.source.sha(bank.OUT/'plan.json'),
        'feature_names': list(evidence.NAMES), 'only_observed_source_and_query_features': True,
        'no_task_labels_and_no_generator_or_severity_identifiers': True,
        'no_new_TFM_inference': True, 'Base_and_benchmark_unchanged': True,
        'explicit_new_fusion_evidence_extension': True, 'not_original_Shared': True,
        'already_scored_development_cohort': True, 'independent_confirmation': False,
        'expected_conditions': 1327, 'script_sha256': bank.source.sha(__file__),
        'evidence_sha256': bank.source.sha(evidence.__file__), 'Markdown_written': False}
    path = OUT/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)
    cases = [c for c in json.loads((bank.source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    counts = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(worker, c) for c in cases]):
            counts.append(future.result())
    assert sum(counts) == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'conditions': sum(counts)})
    print('DOMAIN_EVIDENCE_FEATURES_COMPLETE', sum(counts), flush=True)


if __name__ == '__main__':
    main()
