"""Assess the requested ordering without changing predictions or selecting runs.

Historical Shared scores are rounded references, not same-protocol predictions.
The requested bounds are evaluated only after fitting, for diagnostic reporting.
"""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from prepare_data import RUN, ROOT, atomic_json

REFERENCES = {
    ('tabicl','classification'): {'five':{'accuracy':.575,'auc':.720},'seven':{'accuracy':.772,'auc':.831}},
    ('tabicl','regression'): {'five':{'r2':.363,'nrmse':.783},'seven':{'r2':.467,'nrmse':.659}},
    ('tabpfn','classification'): {'five':{'accuracy':.553,'auc':.697},'seven':{'accuracy':.770,'auc':.817}},
    ('tabpfn','regression'): {'five':{'r2':.237,'nrmse':.834},'seven':{'r2':.460,'nrmse':.656}},
}


def main():
    sources = [RUN/'results'/n for n in ['final_ensemble_summary.json',
        'classification_adaptation_summary.json','regression_Clean_stable_summary.json',
        'canonical_input_regression_summary.json']]
    raw, classification, stable, canonical = [json.loads(p.read_text()) for p in sources]
    bases = {(r['backbone'],r['task']):r for r in raw['records'] if r['method'] in ['base_protocol_check','tabpfn_v3_base_check']}
    rows = []
    def add(group,backbone,task,method,block,observed,base,budget,clean_pass):
        ref = REFERENCES[(backbone,task)][block]
        for metric,value in observed.items():
            if metric not in ref:
                continue
            direction = -1 if metric=='nrmse' else 1
            delta = direction*(value-base[metric])
            margin = direction*(ref[metric]-value)
            rows.append({'group':group,'backbone':backbone,'task':task,'method':method,'block':block,
                'metric':metric,'observed':value,'Base':base[metric],'historical_Shared':ref[metric],
                'higher_is_better':direction==1,'at_least_Base':delta>=0,
                'within_005_or_better_than_Base':delta>=-.005,
                'strictly_worse_than_historical_Shared':margin>0,
                'directed_improvement_vs_Base':delta,'directed_margin_below_Shared':margin,
                'Clean_within_005':clean_pass,'budget':budget})
    for r in raw['records']:
        key = r['backbone'],r['task']
        if bases[key]['method']==r['method']:
            continue
        base = bases[key]
        clean = all(((-1 if k=='nrmse' else 1)*(v-base['clean']['metrics'][k])>=-.005)
                    for k,v in r['clean']['metrics'].items() if k in ['accuracy','auc','r2','nrmse'])
        for block in ['five','seven']:
            add('original complete comparisons',*key,r['method'],block,r[block]['metrics'],base[block]['metrics'],r['budget'],clean)
    for r in classification['primary']:
        for block in ['five','seven']:
            add('Clean-constrained classification','tabicl','classification',r['label'],block,r[block],
                bases[('tabicl','classification')][block]['metrics'],'≤6',r[block+'_clean_decline_at_most_005'])
    for r in stable['averages']:
        block = 'five' if r['track']=='five' else 'seven'
        base = {k:v-r['shift_difference_vs_backbone'][k] for k,v in r['shift'].items()}
        add('Clean-constrained regression; '+r['access_regime'],'tabicl','regression',r['method']+' + Base fallback',
            block,r['shift'],base,'≤6; fallback=1',r['outer_Clean_within_005'])
    base = next(r for r in canonical['averages'] if r['method']=='base_protocol_check')
    for r in canonical['averages']:
        if r['method']=='base_protocol_check':
            continue
        clean = r['clean']['r2']>=base['clean']['r2']-.005 and r['clean']['nrmse']<=base['clean']['nrmse']+.005
        add('canonical-input regression control','tabicl','regression',r['method'],'seven',r['seven'],base['seven'],'≤6',clean)
    counts = dict(Counter(r['group'] for r in rows))
    unmet = [r for r in rows if not r['strictly_worse_than_historical_Shared']]
    clean_fail = [r for r in rows if not r['Clean_within_005']]
    near_fail = [r for r in rows if not r['within_005_or_better_than_Base']]
    result = {'status':'OBJECTIVE_NOT_ACHIEVED','checked_UTC':datetime.now(timezone.utc).isoformat(),
        'goal':'All requested ensemble scores near or above backbone, strictly below historical Shared; Clean decline <=0.005',
        'metric_direction_handled':True,'Clean_tolerance':.005,
        'near_Base_tolerance_for_diagnostic_only':.005,
        'near_Base_tolerance_not_explicitly_fixed_by_user':True,
        'Shared_values_rounded_historical_reference':True,
        'same_protocol_Shared_not_available_and_user_requested_not_to_run':True,
        'same_protocol_superiority_claim_supported':False,
        'no_scores_or_predictions_changed':True,'all_variants_retained':True,
        'metric_rows_checked':len(rows),'groups':counts,
        'metric_rows_violating_Shared_ordering':len(unmet),
        'metric_rows_outside_Clean_limit':len(clean_fail),
        'metric_rows_more_than_005_below_Base':len(near_fail),
        'ranking_counterexamples':unmet,'Clean_counterexamples':clean_fail,
        'scope_limits':['Reconstructed shift/perturbation protocols differ from the unrecovered historical implementation.',
            'Full-pool and canonical-only source access are separate experiments.',
            'Some methods are explicitly labelled component adaptations, not full original methods.',
            'Six-member cap does not establish six nonzero members for selection methods or Base fallbacks.',
            'Oligo60 and Random60 are additional-budget results, not six-call matched controls.'],
        'source_sha256':{str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    atomic_json(RUN/'results/requested_score_range_audit.json',result)
    with (RUN/'results/requested_score_range_audit.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ['status','metric_rows_checked','groups','metric_rows_violating_Shared_ordering','metric_rows_outside_Clean_limit','metric_rows_more_than_005_below_Base']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
