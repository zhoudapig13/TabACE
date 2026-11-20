"""Combine observable domain mismatch with source-only validation evidence.

Source validation is imperfect under measurement shift. Its four descriptors
are therefore combined with three physical distribution descriptors, rather
than used as a rule that automatically trusts the source-CV winner.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
import shared_dual_canonical_banks_v182 as bank
import shared_domain_evidence_features_v187 as domain
import source_native_reliability_v191 as reliability
from prepare_data import RUN, atomic_json

OUT = RUN/'results/shared_source_evidence_v193'
INDICES = [2, 4, 6]
NAMES = tuple(domain.evidence.NAMES[i] for i in INDICES)+reliability.NAMES


def worker(c):
    source = bank.source
    count = 0
    for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in source.novel.STRENGTHS]:
        name = f'{family}_{rho}.npz'
        dpath = domain.OUT/'features'/c['task']/str(c['task_id'])/name
        rpath = reliability.OUT/'features'/c['task']/str(c['task_id'])/name
        dmeta = json.loads(dpath.with_suffix('.json').read_text())
        rmeta = json.loads(rpath.with_suffix('.json').read_text())
        assert source.sha(dpath) == dmeta['output_sha256']
        assert source.sha(rpath) == rmeta['output_sha256']
        assert dmeta['source_input_sha256'] == rmeta['original_input_sha256'] == c['input_sha256']
        assert dmeta['family'] == rmeta['family'] == family and dmeta['rho'] == rmeta['rho'] == rho
        with np.load(dpath) as d, np.load(rpath) as r:
            values = np.r_[d['domain_evidence'][INDICES], r['source_reliability']]
        assert values.shape == (7,) and np.isfinite(values).all()
        dest = OUT/'features'/c['task']/str(c['task_id'])/name
        source.npwrite(dest, domain_evidence=values)
        atomic_json(dest.with_suffix('.json'), {k: c[k] for k in ('task', 'task_id', 'data_id', 'fold')} |
            {'family': family, 'rho': rho, 'output_sha256': source.sha(dest),
            'source_input_sha256': c['input_sha256'], 'domain_parent': str(dpath),
            'domain_parent_sha256': source.sha(dpath), 'source_reliability_parent': str(rpath),
            'source_reliability_parent_sha256': source.sha(rpath),
            'existing_observed_source_labels_used_by_reliability': True,
            'final_query_outcomes_used_by_features': False, 'feature_function_receives_family_or_strength': False})
        count += 1
    return count


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = bank.source
    for parent in (domain.OUT, reliability.OUT):
        assert json.loads((parent/'execution.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((reliability.OUT/'independent_source_audit.json').read_text())['status'] == 'PASS'
    plan = {'status': 'FROZEN_SOURCE_VALIDATION_AND_DOMAIN_EVIDENCE_EXTENSION', 'seed': 0,
        'feature_names': list(NAMES), 'domain_feature_indices': INDICES,
        'domain_parent_plan_sha256': source.sha(domain.OUT/'plan.json'),
        'reliability_parent_plan_sha256': source.sha(reliability.OUT/'plan.json'),
        'source_reliability_implementation_sha256': source.sha(reliability.__file__),
        'domain_descriptor_implementation_sha256': source.sha(domain.evidence.__file__),
        'script_sha256': source.sha(__file__), 'context_bank': str(bank.OUT),
        'context_bank_and_Base_and_benchmark_unchanged': True,
        'own_observed_source_labels_used_for_local_reliability': True,
        'extra_labeled_rows': 0, 'own_final_query_labels_excluded_from_all_fitting': True,
        'source_CV_estimator_forward_passes': 'up to27',
        'final_query_prediction_estimator_forward_passes': 'up to13',
        'total_calibration_and_query_estimator_forward_passes': 'up to40',
        'new_TFM_inference_in_this_merge_stage': False, 'expected_conditions': 1327,
        'known_source_to_query_validation_mismatch_explicit': True,
        'previously_scored_development_cohort': True, 'independent_confirmation': False,
        'Markdown_written': False}
    p = OUT/'plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan
    else:
        atomic_json(p, plan)
    cases = [c for c in json.loads((source.OUT/'case_manifest.json').read_text()) if c['track'] == 'perturbation']
    counts = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(worker, c) for c in cases]):
            counts.append(future.result())
    assert sum(counts) == 1327
    atomic_json(OUT/'execution.json', {'status': 'COMPLETE', 'conditions': sum(counts)})
    print('SOURCE_EVIDENCE_FEATURES_COMPLETE', sum(counts), flush=True)


if __name__ == '__main__':
    main()
