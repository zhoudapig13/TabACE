"""Freeze fully audited remaining classification shifts with strict >.02 gain."""
import argparse
from datetime import datetime, timezone
import json
from prepare_data import RUN, atomic_json
from serial_endpoint_state_v371 import verify_registry, STATE, source

CONFIGS = {
    'nonlinear_sensor': {'experiment':'nonlinear_sensor_supervised_probe_v385',
        'datasets':32, 'parent':381, 'registry':391, 'frozen_count':11, 'total_conditions':1860},
    'interaction_sampling': {'experiment':'interaction_sampling_supervised_probe_v395',
        'datasets':35, 'parent':391, 'registry':401, 'frozen_count':12, 'total_conditions':2000},
}


def main(family, experiment_override=None):
    config = dict(CONFIGS[family])
    if experiment_override is not None:
        assert experiment_override.startswith(family+'_')
        config['experiment'] = experiment_override
    experiment = RUN / 'results' / config['experiment']
    report = json.loads((experiment/'independent_audit.json').read_text())
    acceptance = json.loads((experiment/'numerical_acceptance.json').read_text())
    n = config['datasets'] * 4
    endpoint = ['shift', 'classification', family]
    assert report['status']=='PASS' and report['conditions']==n and not report['pilot_only']
    assert acceptance['active_endpoint']==endpoint and acceptance['datasets']==config['datasets']
    assert acceptance['active_endpoint_pass'] and acceptance['gain']>.02
    assert acceptance['required_gain_strictly_greater_than']==.02
    parent = RUN / f'results/frozen_endpoints_v{config["parent"]}/registry.json'
    preservation = verify_registry(parent)
    entries = []
    target_budgets = set()
    for meta in sorted((experiment/'predictions').glob('*/*.json')):
        pred = meta.with_suffix('.npz')
        data = json.loads(meta.read_text())
        assert source.sha(pred)==data['output_sha256'] and not data['evaluation_labels_opened']
        assert data['family']==family and data['track']=='shift' and data['task']=='classification'
        target_budgets.add(len(data['adaptation_positions']))
        entries.append({'prediction_file':str(pred), 'prediction_sha256':source.sha(pred),
                        'metadata_file':str(meta), 'metadata_sha256':source.sha(meta)})
    assert len(entries)==n
    parent_data = json.loads(parent.read_text())
    remaining = [e for e in parent_data['unresolved_endpoints'] if e!=endpoint]
    registry = {
        'status':'FROZEN_VERIFIED_ENDPOINTS_UNDER_LATEST_USER_THRESHOLDS',
        'parent_registry_file':str(parent), 'parent_registry_sha256':preservation['registry_sha256'],
        'frozen_endpoint_count':config['frozen_count'],
        'newly_frozen_endpoints':[{'endpoint':endpoint, 'directory':str(experiment),
            'method':'Shared_probe', 'numerical_result':acceptance, 'conditions':entries,
            'annotation':f'Source-finetuned encoder and supervised TFM readout; extra clean source references, reserved source validation labels and {sorted(target_budgets)} original-test adaptation labels per case; adaptation rows excluded from evaluation; repeatedly scored development cohort',
            'target_adaptation_row_budgets':sorted(target_budgets)}],
        'these_are_separately_adapted_recipes_not_one_global_Shared_configuration':True,
        'acceptance_contract_file':str(RUN/'results/acceptance_contract_v361/contract.json'),
        'acceptance_contract_sha256':source.sha(RUN/'results/acceptance_contract_v361/contract.json'),
        'active_endpoint':remaining[0] if remaining else None,
        'unresolved_endpoints':remaining, 'Markdown_written':False,
    }
    path = RUN / f'results/frozen_endpoints_v{config["registry"]}/registry.json'
    if path.exists():
        assert json.loads(path.read_text())==registry
    else:
        atomic_json(path, registry)
    verified = verify_registry(path)
    assert verified['prediction_and_metadata_files']==config['total_conditions']*2
    state = json.loads(STATE.read_text())
    frozen = [e for e in state['frozen_endpoints'] if e!=endpoint]+[endpoint]
    state.update(updated_UTC=datetime.now(timezone.utc).isoformat(), frozen_registry=str(path),
                 frozen_endpoint_count=config['frozen_count'], frozen_endpoints=frozen,
                 active_endpoint=registry['active_endpoint'], unresolved_endpoints=remaining,
                 active_datasets=35 if remaining else None,
                 current_full_experiment=None, current_full_audit=None,
                 current_best_completed_full_cohort=acceptance,
                 next_actions=['Run only the last classification shift: interaction sampling'] if remaining
                              else ['Independently recompute all twelve endpoint metrics and audit the complete goal before claiming completion'])
    atomic_json(STATE, state)
    print('FROZEN_ENDPOINTS',config['frozen_count'],'NEXT',registry['active_endpoint'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('family',choices=list(CONFIGS))
    parser.add_argument('--experiment')
    args = parser.parse_args()
    main(args.family,args.experiment)
