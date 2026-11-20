"""Immutable passing-endpoint registries and serial execution guards."""
import json
from pathlib import Path
from datetime import datetime, timezone
from prepare_data import RUN, ROOT, atomic_json
import heavy_source_prior_probe_v358 as previous

source = previous.source
STATE = RUN / 'results/serial_workflow_state_v348/state.json'


def verify_registry(path):
    seen_files, seen_registries = {}, set()
    def visit(registry):
        registry = Path(registry)
        if str(registry) in seen_registries:
            return
        seen_registries.add(str(registry))
        data = json.loads(registry.read_text())
        if 'acceptance_contract_sha256' in data:
            assert source.sha(data['acceptance_contract_file']) == data['acceptance_contract_sha256']
        if 'parent_registry_file' in data:
            parent = Path(data['parent_registry_file'])
            assert source.sha(parent) == data['parent_registry_sha256']
            visit(parent)
        for key in ('endpoints', 'unchanged_parent_endpoints', 'additional_archived_endpoints', 'newly_frozen_endpoints'):
            for entry in data.get(key, []):
                for row in entry['conditions']:
                    for prefix in ('prediction', 'metadata'):
                        p, expected = row[prefix + '_file'], row[prefix + '_sha256']
                        if p in seen_files:
                            assert seen_files[p] == expected
                        else:
                            assert source.sha(p) == expected
                            seen_files[p] = expected
    visit(path)
    assert (ROOT / 'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    return {'registry_sha256': source.sha(path), 'prediction_and_metadata_files': len(seen_files)}


def freeze_heavy():
    experiment = RUN / 'results/heavy_domain_validated_probe_v366'
    report = json.loads((experiment / 'independent_audit.json').read_text())
    acceptance = json.loads((experiment / 'numerical_acceptance.json').read_text())
    assert report['status'] == 'PASS' and report['conditions'] == 256 and not report['pilot_only']
    assert acceptance['active_endpoint_pass'] and acceptance['datasets'] == 64
    assert acceptance['gain'] > .03 and acceptance['required_gain_strictly_greater_than'] == .03
    parent = RUN / 'results/frozen_endpoints_v363/registry.json'
    preservation = verify_registry(parent)
    entries = []
    for meta in sorted((experiment / 'predictions').glob('*/*.json')):
        pred = meta.with_suffix('.npz')
        data = json.loads(meta.read_text())
        assert source.sha(pred) == data['output_sha256'] and not data['evaluation_labels_opened']
        entries.append({'prediction_file': str(pred), 'prediction_sha256': source.sha(pred),
                        'metadata_file': str(meta), 'metadata_sha256': source.sha(meta)})
    assert len(entries) == 256
    endpoint = ['perturbation', 'classification', 'cell_heavy_tail']
    parent_data = json.loads(parent.read_text())
    remaining = [e for e in parent_data['unresolved_endpoints'] if e != endpoint]
    registry = {'status': 'FROZEN_VERIFIED_ENDPOINTS_UNDER_LATEST_USER_THRESHOLDS',
                'parent_registry_file': str(parent), 'parent_registry_sha256': preservation['registry_sha256'],
                'frozen_endpoint_count': 9,
                'newly_frozen_endpoints': [{'endpoint': endpoint, 'directory': str(experiment),
                    'method': 'Shared_probe', 'numerical_result': acceptance, 'conditions': entries,
                    'annotation': 'Source-finetuned encoder plus supervised source-prior linear readout; extra clean source references, independent64 source validation labels and32 original-test adaptation labels; evaluation96 excludes adaptation rows; repeated development cohort'}],
                'these_are_separately_adapted_recipes_not_one_global_Shared_configuration': True,
                'acceptance_contract_file': str(RUN / 'results/acceptance_contract_v361/contract.json'),
                'acceptance_contract_sha256': source.sha(RUN / 'results/acceptance_contract_v361/contract.json'),
                'active_endpoint': remaining[0], 'unresolved_endpoints': remaining, 'Markdown_written': False}
    path = RUN / 'results/frozen_endpoints_v372/registry.json'
    if path.exists():
        assert json.loads(path.read_text()) == registry
    else:
        atomic_json(path, registry)
    verified = verify_registry(path)
    assert verified['prediction_and_metadata_files'] == 1444 * 2
    state = json.loads(STATE.read_text())
    frozen = [e for e in state['frozen_endpoints'] if e != endpoint] + [endpoint]
    state.update(updated_UTC=datetime.now(timezone.utc).isoformat(), frozen_registry=str(path),
                 frozen_endpoint_count=9, frozen_endpoints=frozen,
                 active_endpoint=remaining[0], unresolved_endpoints=remaining,
                 active_datasets=72, current_full_experiment=None, current_full_audit=None,
                 current_best_completed_full_cohort=acceptance,
                 next_actions=['Extract unchanged Shared features for the next endpoint only: column-desynchronization classification',
                               'Fit supervised Native/Shared readouts with the same disjoint label budget; independently audit the entire cohort'])
    atomic_json(STATE, state)
    print('FROZEN_ENDPOINTS', 9, 'NEXT', remaining[0], flush=True)
    return path


if __name__ == '__main__':
    freeze_heavy()
