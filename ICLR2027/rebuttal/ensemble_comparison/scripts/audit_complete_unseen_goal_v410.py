"""Recompute every frozen endpoint from current predictions and held-out truth.

This verifies the latest user contract (> .03 perturbations, > .02 shifts), not
the superseded .05 perturbation contract. Recipes differ across endpoints and
their extra-supervision annotations are retained in the output. No models run.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import unseen_shared_robustness_v117 as source
from prepare_data import RUN, ROOT, atomic_json
from run_tabicl import metrics
from serial_endpoint_state_v371 import verify_registry

FAMILIES = {
    'perturbation': ('cell_heavy_tail', 'column_desynchronization', 'feature_quantization'),
    'shift': ('cross_feature_rotation', 'nonlinear_sensor', 'interaction_sampling'),
}
OUT = RUN / 'results/unseen_goal_completion_v410'


def collect(path):
    entries, visited = {}, set()
    def visit(p):
        p = Path(p)
        if str(p) in visited:
            return
        visited.add(str(p))
        data = json.loads(p.read_text())
        if 'parent_registry_file' in data:
            assert source.sha(data['parent_registry_file']) == data['parent_registry_sha256']
            visit(data['parent_registry_file'])
        for key in ('endpoints','unchanged_parent_endpoints','additional_archived_endpoints','newly_frozen_endpoints'):
            for entry in data.get(key, []):
                endpoint = tuple(entry['endpoint'])
                if endpoint in entries:
                    assert entries[endpoint] == entry
                else:
                    entries[endpoint] = entry
    visit(path)
    return entries


def experiment_directory(entry):
    if 'directory' in entry:
        return Path(entry['directory'])
    path = Path(entry['conditions'][0]['prediction_file'])
    return next(p.parent for p in path.parents if p.name == 'predictions')


def main(registry, partial=False):
    assert tuple(source.novel.STRENGTHS)==(20,50,80,100)
    entries = collect(registry)
    expected_endpoints = {(track,task,family) for track,fs in FAMILIES.items()
                          for task in ('classification','regression') for family in fs}
    assert set(entries) <= expected_endpoints
    if not partial:
        assert set(entries) == expected_endpoints
        assert json.loads(Path(registry).read_text())['unresolved_endpoints'] == []
    preservation = verify_registry(registry)
    contract_path = RUN/'results/acceptance_contract_v361/contract.json'
    contract = json.loads(contract_path.read_text())
    assert contract['perturbation_required_gain_strictly_greater_than']==.03
    assert contract['shift_required_gain_strictly_greater_than']==.02
    cases = json.loads((source.OUT/'case_manifest.json').read_text())
    rows, summary, audits = [], [], []
    conditions = 0
    for endpoint,entry in sorted(entries.items()):
        track,task,family = endpoint
        selected = [c for c in cases if c['track']==track and c['task']==task and family in c['families']]
        assert len({c['data_id'] for c in selected}) == len(selected)
        case_map = {c['task_id']:c for c in selected}
        required = {(c['task_id'],rho) for c in selected for rho in source.novel.STRENGTHS}
        observed = set()
        method = entry.get('method','Shared_adapted')
        native_method = method.replace('Shared','Native',1)
        directory = experiment_directory(entry)
        plan_path = directory/'plan.json'
        plan = json.loads(plan_path.read_text())
        assert plan['seed']==0
        assert source.sha(RUN/'scripts'/f'{directory.name}.py')==plan['script_sha256']
        report_path = directory/'independent_audit.json'
        report = json.loads(report_path.read_text())
        assert report['status']=='PASS' and not report.get('pilot_only',False)
        if 'upstream_full_audit_sha256' in entry:
            assert source.sha(report_path)==entry['upstream_full_audit_sha256']
        if 'original_plan_sha256' in entry:
            assert source.sha(plan_path)==entry['original_plan_sha256']
        endpoint_rows = []
        for frozen in entry['conditions']:
            path = Path(frozen['prediction_file'])
            meta_path = Path(frozen['metadata_file'])
            assert source.sha(path)==frozen['prediction_sha256']
            assert source.sha(meta_path)==frozen['metadata_sha256']
            meta = json.loads(meta_path.read_text())
            assert source.sha(path)==meta['output_sha256']
            assert meta['family']==family and meta['task']==task
            assert meta.get('track',track)==track
            key = (meta['task_id'],meta['rho'])
            assert key in required and key not in observed
            observed.add(key)
            c = case_map[meta['task_id']]
            assert meta['data_id']==c['data_id']
            assert source.sha(c['input_file'])==c['input_sha256']
            assert source.sha(c['truth_file'])==c['truth_sha256']
            if 'original_input_sha256' in meta:
                assert meta['original_input_sha256']==c['input_sha256']
            with np.load(c['truth_file']) as z:
                original_y = z['y_target']
            if 'evaluation_positions_in_original_query' in meta:
                epos = np.asarray(meta['evaluation_positions_in_original_query'],dtype=int)
                apos = np.asarray(meta['target_adaptation_positions'],dtype=int)
            elif 'evaluation_positions' in meta:
                epos = np.asarray(meta['evaluation_positions'],dtype=int)
                apos = np.asarray(meta['adaptation_positions'],dtype=int)
            else:
                epos = np.arange(len(original_y))
                apos = np.array([],dtype=int)
            assert len(set(epos))==len(epos) and len(set(apos))==len(apos)
            assert set(apos).isdisjoint(epos)
            assert set(np.r_[apos,epos])==set(range(len(original_y)))
            flags = [meta[k] for k in ('final_eval_labels_opened','final_eval_scores_used_for_selection',
                     'evaluation_labels_opened','query_labels_opened','query_labels_opened_for_inference') if k in meta]
            assert flags and not any(flags)
            y = original_y[epos]
            original_path = source.OUT/'predictions'/track/task/str(c['task_id'])/path.name
            original_meta = json.loads(original_path.with_suffix('.json').read_text())
            assert source.sha(original_path)==original_meta['output_sha256']
            if 'original_prediction_sha256' in meta:
                assert source.sha(original_path)==meta['original_prediction_sha256']
            with np.load(path) as prediction, np.load(original_path) as original:
                np.testing.assert_array_equal(prediction['Base'],original['Base'][epos])
                methods = ['Base',method]+([native_method] if native_method in prediction.files else [])
                for name in methods:
                    pred = prediction[name]
                    assert len(pred)==len(y) and np.isfinite(pred).all()
                    endpoint_rows.append({'track':track,'task':task,'family':family,
                        'data_id':c['data_id'],'task_id':c['task_id'],'rho':meta['rho'],
                        'method':name,'evaluation_rows':len(y),'target_adaptation_rows':len(apos),
                        **metrics(y,pred,task,max(float(np.std(y)),1e-12))})
            conditions += 1
        assert observed==required and len(observed)==4*len(selected)
        frame = pd.DataFrame(endpoint_rows)
        metric_names = ['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        table = frame.groupby(['data_id','method'])[metric_names].mean().groupby('method').mean()
        primary = 'accuracy' if task=='classification' else 'r2'
        base,adapted = float(table.loc['Base',primary]),float(table.loc[method,primary])
        gain = adapted-base
        threshold = .03 if track=='perturbation' else .02
        assert gain>threshold
        recorded = entry.get('numerical_result',entry)
        np.testing.assert_allclose(recorded['gain'],gain,atol=1e-10,rtol=0)
        np.testing.assert_allclose(recorded['Base'],base,atol=1e-10,rtol=0)
        old_table = pd.read_csv(directory/'family_averages.csv')
        if 'track' in old_table:
            old_table = old_table[old_table.track==track]
        if 'task' in old_table:
            old_table = old_table[old_table.task==task]
        if 'family' in old_table:
            old_table = old_table[old_table.family==family]
        for name in table.index:
            old_row = old_table[old_table.method==name]
            assert len(old_row)==1
            np.testing.assert_allclose(old_row[metric_names].iloc[0].to_numpy(dtype=float),
                                       table.loc[name].to_numpy(dtype=float),atol=1e-10,rtol=0)
        summary.append({'track':track,'task':task,'family':family,'datasets':len(selected),
            'conditions':len(required),'method':method,'recipe':directory.name,
            'Base':base,'Shared_adapted':adapted,'gain':gain,'threshold_strict':threshold,'passed':True,
            **{f'Base_{m}':float(table.loc['Base',m]) for m in metric_names},
            **{f'Shared_{m}':float(table.loc[method,m]) for m in metric_names},
            **({f'Native_{m}':float(table.loc[native_method,m]) for m in metric_names} if native_method in table.index else {}),
            'annotation':entry.get('annotation','Quarter-target-label guarded supervised residual adaptation on source-finetuned TFM; extra source/reference labels; original-test adaptation rows excluded from evaluation'),
            'target_adaptation_rows_min':int(frame.target_adaptation_rows.min()),
            'target_adaptation_rows_max':int(frame.target_adaptation_rows.max()),
            'evaluation_rows_min':int(frame.evaluation_rows.min()),
            'evaluation_rows_max':int(frame.evaluation_rows.max())})
        rows.extend(endpoint_rows)
        audits.append({'endpoint':list(endpoint),'audit_file':str(report_path),
                       'audit_sha256':source.sha(report_path),'plan_sha256':source.sha(plan_path)})
        print('COMPLETION_ENDPOINT_VERIFIED',endpoint,len(selected),gain,flush=True)
    expected_total = sum(4*sum(c['track']==track and c['task']==task and family in c['families'] for c in cases)
                         for track,task,family in expected_endpoints)
    if not partial:
        assert conditions==expected_total==2000 and len(summary)==12
    out = OUT/f'partial_{len(summary)}' if partial else OUT
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'recomputed_condition_metrics.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'endpoint_averages.csv',index=False)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(out/'completion_audit.json',{
        'status':'PARTIAL_VERIFIED' if partial else 'PASS','complete_goal_verified':not partial,
        'endpoints':len(summary),'conditions':conditions,'expected_endpoints':12,'expected_conditions':expected_total,
        'contract_sha256':source.sha(contract_path),'registry_sha256':preservation['registry_sha256'],
        'all_Base_predictions_bitwise_original':True,'all_adaptation_rows_excluded_from_evaluation':True,
        'all_four_frozen_positive_strengths_and_full_eligible_cohorts_verified':True,
        'all_thresholds_strict_unrounded':True,'single_experiment_seed':0,
        'supervised_extensions_explicitly_annotated':True,
        'different_endpoint_recipes_and_supervision_budgets':True,
        'not_original_zero_shot_Shared_or_blind_replication':True,
        'this_audit_did_not_delete_or_modify_prior_experiment_results':True,'Markdown_bytes':0,
        'upstream_full_audits':audits,'script_sha256':source.sha(__file__)})
    print('COMPLETE_GOAL_VERIFIED' if not partial else 'PARTIAL_ONLY',len(summary),conditions,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--registry',default=str(RUN/'results/frozen_endpoints_v401/registry.json'))
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    main(Path(args.registry),args.partial)
