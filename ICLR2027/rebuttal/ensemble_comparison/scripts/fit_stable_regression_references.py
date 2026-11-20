"""Finite, calibration-only Clean-stability selection for regression references.

No historical Shared scores enter selection. Unconstrained and input-access
variants remain reported separately, including any stronger results.
"""
import json
import math
import numpy as np
from fit_combiners import fold_mapping
from fit_convex_combiners import design, fit_convex, fit_ges
from prepare_data import RUN, ROOT, atomic_json
from run_tabicl import metrics


def load(track, regime):
    root = (RUN/'results/canonical_input/tabicl' if regime == 'canonical' else RUN/'results/tabicl')/track/'regression'
    mapping, _ = fold_mapping('regression',track)
    cells = []
    for path in sorted(root.glob('*/*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as f:
            bank, y = f['bootstrap_bank'], f['y_target']
            predictions = {k:f['prediction__'+k] for k in ['base_protocol_check','native_6','bootstrap_mean_6','oligoicp_iqr_6_keep1_adapted']}
            mu, sigma = float(f['source_y_mean']),float(f['source_y_std'])
        cells.append({'meta':meta,'bank':bank,'y':y,'mu':mu,'sigma':sigma,
                      'predictions':predictions,'fold':mapping[meta['data_id']],
                      'clean_weight':(7. if track.startswith('seven') else 5.) if meta['rho']==0 else 1.})
    expected = 957 if track.startswith('seven') else 735
    assert len(cells) == expected
    return cells


def predict(c, method, model):
    if model.get('base_backoff'):
        return c['predictions']['base_protocol_check']
    if method in ['GES-6','Convex Stacking-6']:
        return np.asarray(model['weights']) @ c['bank']
    key = {'Native-6':'native_6','Random-6':'bootstrap_mean_6',
           'OligoICP 6→1':'oligoicp_iqr_6_keep1_adapted'}[method]
    return c['predictions'][key]


def evaluate(cells, method, model, clean_only=False):
    rows = []
    for c in cells:
        if clean_only and c['meta']['rho'] != 0:
            continue
        scale = max(float(c['y'].std()),1e-12) if c['meta']['track'].startswith('seven') else c['meta']['scale']
        rows.append(metrics(c['y'],predict(c,method,model),'regression',scale))
    return {k:math.fsum(r[k] for r in rows)/len(rows) for k in ['r2','nrmse']}


def objective(cells, method, model):
    # Same per-dataset/condition normalized squared prediction loss as design().
    from fit_combiners import sample_weights
    total = 0.
    for c,w in zip(cells,sample_weights(cells)):
        scale = max(float(c['y'].std()),1e-12) if c['meta']['track'].startswith('seven') else c['meta']['scale']
        total += float(w @ ((predict(c,method,model)-c['y'])/scale)**2)
    return total


def main():
    all_averages = []
    for track,regime in [('five','canonical'),('seven_calibrated_v2','canonical'),('seven_calibrated_v2','full_pool')]:
        # The original five track already has exactly Base's canonical pool.
        cells = load(track,'full_pool' if track=='five' else regime)
        all_rows = []
        for fold in range(5):
            train = [c for c in cells if c['fold'] != fold]
            test = [c for c in cells if c['fold'] == fold]
            inner = min(c['fold'] for c in train)
            inner_train = [c for c in train if c['fold'] != inner]
            val = [c for c in train if c['fold'] == inner]
            train_ids = {c['meta']['data_id'] for c in train}
            test_ids = {c['meta']['data_id'] for c in test}
            assert train_ids.isdisjoint(test_ids)
            base_clean = evaluate(val,'Random-6',{'base_backoff':True},True)
            for method in ['Native-6','Random-6','OligoICP 6→1','GES-6','Convex Stacking-6']:
                candidates = [(None,{'base_backoff':True})]
                if method == 'GES-6':
                    weights,_ = fit_ges(*design(inner_train,'regression'),'regression')
                    candidates.append((0.,{'weights':weights.tolist()}))
                elif method == 'Convex Stacking-6':
                    for reg in [0.,.001,.01,.1,1.]:
                        weights = fit_convex(*design(inner_train,'regression'),'regression',reg)
                        candidates.append((reg,{'weights':weights.tolist()}))
                else:
                    candidates.append((0.,{}))
                trials = []
                best = None
                for reg,model in candidates:
                    clean = evaluate(val,method,model,True)
                    feasible = clean['r2'] >= base_clean['r2']-.005 and clean['nrmse'] <= base_clean['nrmse']+.005
                    loss = objective(val,method,model)
                    trials.append({'regularization':reg,'model':model,'clean':clean,'clean_feasible':feasible,'loss':loss})
                    if feasible and (best is None or loss < best[0]):
                        best = (loss,reg,model)
                assert best is not None
                _,reg,model = best
                if not model.get('base_backoff'):
                    if method == 'GES-6':
                        weights,_ = fit_ges(*design(train,'regression'),'regression')
                        model = {'weights':weights.tolist()}
                    elif method == 'Convex Stacking-6':
                        weights = fit_convex(*design(train,'regression'),'regression',reg)
                        model = {'weights':weights.tolist()}
                rows = []
                for c in test:
                    scale = max(float(c['y'].std()),1e-12) if track.startswith('seven') else c['meta']['scale']
                    pred = predict(c,method,model)
                    rows.append(c['meta'] | {'method':method,'outer_fold':fold,'access_regime':regime,
                        'base_backoff':bool(model.get('base_backoff'))} | metrics(c['y'],pred,'regression',scale))
                all_rows += rows
                atomic_json(RUN/'results/regression_Clean_stable'/regime/track/f"fold_{fold}_{method.replace('/','_')}.json",{
                    'status':'COMPLETE','seed':0,'fold':fold,'method':method,'track':track,'access_regime':regime,
                    'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),
                    'inner_validation_fold':inner,'base_clean_validation':base_clean,'trials':trials,
                    'selected_regularization':reg,'model':model,'rows':rows,
                    'clean_absolute_decline_limit':.005,'candidate_budget':6,
                    'base_fallback_uses_one_forward':True,
                    'outer_test_labels_used_for_fit_or_selection':False,
                    'paper_Shared_used_for_selection':False,'Shared_run':False})
        for method in ['Native-6','Random-6','OligoICP 6→1','GES-6','Convex Stacking-6']:
            item = {'track':track,'access_regime':regime,'method':method}
            for name,nonzero in [('clean',False),('shift',True)]:
                rows = [r for r in all_rows if r['method']==method and (r['rho']>0)==nonzero]
                count = (33 if track.startswith('seven') else 35)*(28 if track.startswith('seven') else 20) if nonzero else (33 if track.startswith('seven') else 35)
                assert len(rows)==count
                item[name] = {k:math.fsum(r[k] for r in rows)/len(rows) for k in ['r2','nrmse']}
                base = evaluate([c for c in cells if (c['meta']['rho']>0)==nonzero],method,{'base_backoff':True})
                item[name+'_difference_vs_backbone'] = {k:item[name][k]-base[k] for k in base}
                if name=='clean':
                    item['outer_Clean_within_005'] = item[name]['r2']>=base['r2']-.005 and item[name]['nrmse']<=base['nrmse']+.005
            item['backoff_condition_fraction'] = sum(r['base_backoff'] for r in all_rows if r['method']==method)/sum(r['method']==method for r in all_rows)
            all_averages.append(item)
        print('COMPLETE',track,regime,flush=True)
    atomic_json(RUN/'results/regression_Clean_stable_summary.json',{
        'status':'COMPLETE','seed':0,'Shared_run':False,'unconstrained_variants_retained':True,
        'ranking_not_enforced':True,'selection':'validation Clean R2 decrease <= .005 and nRMSE increase <= .005; then minimize validation prediction loss',
        'averages':all_averages})
    print(json.dumps(all_averages),flush=True)


if __name__=='__main__':main()
