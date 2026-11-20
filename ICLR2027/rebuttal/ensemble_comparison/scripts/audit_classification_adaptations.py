"""Check complete cohorts, OOF separation, budgets and sampled saved metrics."""
import hashlib
import json
import numpy as np
from fit_combiners import fold_mapping
from prepare_data import RUN, atomic_json
from run_tabicl import metrics


def main():
    count_meta = count_policy = count_metric = 0
    roots = [('robust_six_combiners', 'robust_six_predictions'),
             ('robust_six_combiners_acc_auc', 'robust_six_predictions_acc_auc'),
             ('robust_six_combiners_stable', 'robust_six_predictions_stable')]
    for track, n, families in [('five_feature_calibrated',72,5), ('seven_reimplemented',35,7)]:
        mapping, _ = fold_mapping('classification', track)
        for phase in ['robust_six', 'robust_six_all']:
            root = RUN/'results'/phase/'tabicl'/track/'classification'
            paths = sorted(root.glob('*/*.json'))
            assert len(paths) == n*(1+4*families)
            for path in paths:
                m = json.loads(path.read_text())
                assert m['status'] == 'COMPLETE' and m['seed'] == 0
                assert m['logical_candidate_count'] == 6 and len(m['roles']) == 6
                assert max(m['context_rows']) <= 256
                assert not m['feature_adapter_uses_target_labels']
                assert not m['feature_adapter_uses_family_or_strength']
                assert not m['Shared_or_paper_scores_used']
                count_meta += 1
        for rootname, prediction_root in roots:
            for method in ['Native','Random','DENSE','GES','Convex','Loglinear']:
                paths = sorted((RUN/'results'/rootname/track/'classification').glob(f'fold_*_{method}.json'))
                assert len(paths) == 5
                grid = set()
                for path in paths:
                    p = json.loads(path.read_text())
                    assert not p['outer_test_labels_used_for_fit_or_selection']
                    assert not p['paper_Shared_values_used_for_fit_or_selection'] and not p['Shared_run']
                    assert set(p['train_data_ids']).isdisjoint(p['test_data_ids'])
                    assert all(mapping[i] == p['fold'] for i in p['test_data_ids'])
                    assert all(mapping[i] != p['fold'] for i in p['train_data_ids'])
                    assert p['inner_validation_fold'] != p['fold']
                    assert p['candidate_budget'] == 6
                    if rootname.endswith('_stable'):
                        assert p['clean_tolerance'] == .005
                        # Selection is checked on calibration data; final Clean
                        # feasibility is a reported outcome, not an assumption.
                        if p['selected_mode'] != 3:
                            match = [r for r in p['inner_scores'] if r['mode'] == p['selected_mode'] and r['regularization'] == p['regularization']]
                            assert len(match) == 1 and match[0]['clean_feasible']
                    rows = p['rows']
                    for row in rows:
                        key = (row['task_id'],row['family'],row['rho'])
                        assert key not in grid
                        grid.add(key)
                    for idx in sorted({0,len(rows)//2,len(rows)-1}):
                        row = rows[idx]
                        npz = RUN/'results'/prediction_root/track/'classification'/method/str(row['task_id'])/f"{row['family']}_{row['rho']}.npz"
                        with np.load(npz) as f:
                            pred, y = f['prediction'], f['y_target']
                        assert np.isfinite(pred).all() and (pred >= 0).all()
                        assert np.max(np.abs(pred.sum(1)-1)) < 1e-9
                        observed = metrics(y,pred,'classification',1.)
                        for k in observed:
                            assert abs(observed[k]-row[k]) < 1e-10
                        count_metric += 1
                    count_policy += 1
                assert len(grid) == n*(1+4*families)
    summary = json.loads((RUN/'results/classification_adaptation_summary.json').read_text())
    checks = {r['method']:{b:r[b+'_clean_decline_at_most_005'] for b in ['five','seven']} for r in summary['primary']}
    atomic_json(RUN/'results/classification_adaptation_audit.json', {
        'status':'PASS', 'raw_condition_metadata_checked':count_meta,
        'OOF_policies_checked':count_policy, 'saved_prediction_metrics_recomputed':count_metric,
        'complete_dataset_family_strength_grids':True, 'seed':0, 'candidate_budget':6,
        'Shared_run':False, 'ranking_bounds_used_for_selection':False,
        'final_Clean_within_user_limit':checks,
        'all_primary_Clean_within_limit':all(v for d in checks.values() for v in d.values()),
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            RUN/'scripts/fit_robust_six.py', RUN/'scripts/robust_six_views.py',
            RUN/'scripts/report_classification_adaptations.py',
            RUN/'scripts/audit_classification_adaptations.py']}})
    print('PASS',count_meta,count_policy,count_metric,checks,flush=True)


if __name__ == '__main__':
    main()
