"""Freeze and audit all target-calibrated policies before opening fresh truth."""
import json
import os
from datetime import datetime, timezone
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import torch

import target_supervised_adaptation_v99 as study
from prepare_data import atomic_json, RUN
from run_tabicl import metrics
from report_abcd_v84 import paired_statistics

METHODS=['Base_zero_target','Base_target','Target_only','Single_matched','A','B','C','D']
CONTROLS=['Base_target','Base_zero_target','Target_only']


def sha(path):return study.banks.split.parent.original.sha(path)


def preflight():
    torch.set_num_threads(1)
    execution=json.loads((study.OUT/'execution.json').read_text())
    assert execution['status']=='COMPLETE' and execution['datasets']==72
    assert json.loads((study.OUT/'plan.json').read_text())['code_sha256']==sha(study.__file__)
    assert json.loads((study.banks.OUT/'plan.json').read_text())['code_sha256']==sha(study.banks.__file__)
    assert json.loads((study.banks.OUT/'execution.json').read_text())['status']=='COMPLETE'
    paths=sorted((study.banks.split.OUT/'inputs').glob('*.npz'))
    assert len(paths)==72
    prediction_files=[];policy_files=[];audits=[];total_banks=0
    for path in paths:
        tid=int(path.stem);meta=json.loads(path.with_suffix('.json').read_text())
        assert sha(path)==meta['model_inputs_sha256']
        with np.load(path) as z:data={k:z[k] for k in z.files}
        assert 'y_evaluation' not in data
        data['data_id']=int(meta['data_id'])
        assert set(data['source_ids']).isdisjoint(data['evaluation_raw_ids'])
        assert set(data['calibration_raw_ids']).isdisjoint(data['evaluation_raw_ids'])
        assert len(data['y_calibration'])==64
        folder=study.OUT/'tasks'/str(tid)
        assert json.loads((folder/'completion.json').read_text())['all_four_policies_frozen']
        for family,rho in study.banks.CONDITIONS:
            for stage,fit_ids,query_ids in study.banks.stage_specs(data,family):
                p=study.banks.OUT/'banks'/str(tid)/f'{family}_{rho}'/f'{stage}.npz'
                m=json.loads(p.with_suffix('.json').read_text())
                assert not m['uses_evaluation_labels'] and m['input_file_sha256']==sha(path)
                assert max(m['branch_context_rows']+[m['base_context_rows']])<=256
                with np.load(p) as z:
                    assert np.array_equal(fit_ids,z['fit_cal_positions'])
                    assert np.array_equal(query_ids,z['query_positions'])
                    assert z['members'].shape[0]==5
                    for a in (z['base_target'],z['members'],z['bootstrap']):
                        assert np.isfinite(a).all() and np.allclose(a.sum(-1),1.,atol=1e-6)
                assert set(fit_ids).isdisjoint(query_ids)
                total_banks+=1
        for method in ['A','B','C','D']:
            policy_path=folder/f'{method}_policy.json';saved=json.loads(policy_path.read_text())
            policy_files.append(policy_path)
            seen,matched=method in ('A','C'),method in ('A','B')
            validation=study.cells_for(tid,'validation',matched,seen,data)
            refit=study.cells_for(tid,'refit',matched,seen,data)
            assert all('shortcut' not in c['meta']['family'] for c in validation+refit) or seen
            for trial in saved['trials']:
                values=study.scores(validation,trial['model'])
                assert np.isclose(values['auc'],trial['validation_scores']['auc'],atol=1e-12)
            best=max(saved['trials'],key=lambda t:t['validation_scores']['auc'])
            assert best['candidate']==saved['selected_candidate']
            assert best.get('regularization')==saved['selected_regularization']
            prior,origin=study.source_prior(meta,method)
            assert origin==saved['source_prior']
            if best['candidate']=='Target_fit':
                model=saved['policy'];t=study.rank.pair_data(refit)
                p=torch.as_tensor(model['parameters'],dtype=torch.float64)
                pos=study.rank.mapped_probability(p,t,'positive').clamp(1e-7,1-1e-7)
                neg=study.rank.mapped_probability(p,t,'negative').clamp(1e-7,1-1e-7)
                loss=(torch.nn.functional.softplus(-(torch.logit(pos)-torch.logit(neg)))*t['weights']).sum()
                loss+=model['regularization']*(p-torch.as_tensor(prior)).square().mean()
                assert abs(float(loss)-model['training_objective'])<1e-8
            else:assert saved['policy']==best['model']
            for c in study.cells_for(tid,'evaluation',matched,seen,data):
                prefix=f"{c['meta']['family']}_{c['meta']['rho']}"
                expected={method:study.predict(c,saved['policy'])}
                if method=='A':expected.update(Base_target=c['base'],Base_zero_target=c['zero'],
                    Target_only=c['bank'][1],Single_matched=c['bank'][3])
                for name,p in expected.items():
                    pp=folder/'evaluation_predictions'/f'{prefix}_{name}.npz'
                    with np.load(pp) as z:assert np.allclose(z['prediction'],p,atol=1e-12,rtol=1e-12)
                    prediction_files.append(pp)
            audits.append({'task_id':tid,'method':method,'selection':best['candidate'],
                'validation_auc_recomputed':True,'target_label_budget':64,
                'training_families':saved['allowed_families']})
    assert total_banks==3600 and len(prediction_files)==2880 and len(policy_files)==288
    assert (RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    seal_payload={
        'status':'PASS_BEFORE_FRESH_TRUTH_READ','UTC':datetime.now(timezone.utc).isoformat(),
        'banks_verified':total_banks,'policies_verified':len(policy_files),
        'predictions_verified':len(prediction_files),'evaluation_truth_opened_by_this_audit':False,
        'code_sha256':sha(study.__file__),'reporter_sha256':sha(__file__),
        'policy_sha256':{str(p.relative_to(study.OUT)):sha(p) for p in policy_files},
        'prediction_sha256':{str(p.relative_to(study.OUT)):sha(p) for p in prediction_files},
        'calibration_audits':audits,'Markdown_bytes':0}
    seal_path=study.OUT/'pre_evaluation_seal.json'
    if seal_path.exists():
        prior=json.loads(seal_path.read_text())
        assert prior['policy_sha256']==seal_payload['policy_sha256']
        assert prior['prediction_sha256']==seal_payload['prediction_sha256']
        atomic_json(study.OUT/'post_seal_reaudit.json',seal_payload)
    else:atomic_json(seal_path,seal_payload)
    print('PREFLIGHT_PASS',len(audits),'policies',flush=True)
    return paths


def main():
    paths=preflight();rows=[]
    for path in paths:
        tid=int(path.stem);meta=json.loads(path.with_suffix('.json').read_text())
        truth=study.banks.split.OUT/'evaluation_truth'/f'{tid}.npz'
        assert sha(truth)==meta['evaluation_truth_sha256']
        with np.load(truth) as z:y=z['y_evaluation'];raw_ids=z['raw_ids']
        with np.load(path) as z:assert np.array_equal(raw_ids,z['evaluation_raw_ids'])
        for family,rho in [('clean',0)]+[('shortcut',r) for r in (20,50,80,100)]:
            for method in METHODS:
                p=study.OUT/'tasks'/str(tid)/'evaluation_predictions'/f'{family}_{rho}_{method}.npz'
                with np.load(p) as z:prediction=z['prediction']
                rows.append({'task_id':tid,'data_id':meta['data_id'],'dataset':meta['dataset'],
                    'family':family,'rho':rho,'method':method,'query_rows':len(y),'evaluation_classes':len(np.unique(y)),
                    **metrics(y,prediction,'classification',1.)})
    detail=pd.DataFrame(rows);assert len(detail)==2880
    detail.to_csv(study.OUT/'fresh_evaluation_detail.csv',index=False)
    g=detail[detail.rho>0].groupby(['task_id','method'])[['accuracy','auc']].mean()
    clean=detail[detail.rho==0].set_index(['task_id','method'])[['accuracy','auc']]
    table=[];tests=[]
    for method in METHODS:
        v=g.xs(method,level='method');c=clean.xs(method,level='method')
        b=g.xs('Base_target',level='method');bc=clean.xs('Base_target',level='method')
        table.append({'task':'classification','method':method,'datasets':72,
            'datasets_accuracy':int(v.accuracy.notna().sum()),'datasets_auc':int(v.auc.notna().sum()),
            'accuracy':v.accuracy.mean(),'auc':v.auc.mean(),
            'clean_accuracy':c.accuracy.mean(),'clean_auc':c.auc.mean(),
            'clean_accuracy_change_vs_target_Base':(c.accuracy-bc.accuracy).mean(),
            'clean_auc_change_vs_target_Base':(c.auc-bc.auc).mean(),
            'extra_target_labels':0 if method=='Base_zero_target' else 64})
        if method in ('A','B','C','D'):
            for control in CONTROLS:
                base=g.xs(control,level='method')
                delta=(v.auc-base.auc).dropna()
                assert np.isfinite(delta).all()
                tests.append({'task':'classification','method':method,'control':control,'datasets':len(delta),
                    'primary_metric':'auc',**paired_statistics(delta)})
    tables=pd.DataFrame(table)
    reg=pd.read_csv(RUN/'results/abcd_concept_calibrated_v87/tables.csv')
    for row in reg[reg.method.isin(['A','B','C','D'])].to_dict('records'):
        tests.append({k:row[k] for k in ('task','method','datasets','gain','gain_ci_low','gain_ci_high',
            'p_two_sided','wins','ties','losses')}|{'control':'Base','primary_metric':'r2'})
    stats=pd.DataFrame(tests);assert len(stats)==16
    previous=0.
    for rank,i in enumerate(sorted(stats.index,key=lambda j:stats.loc[j,'p_two_sided'])):
        previous=max(previous,min(1.,(len(stats)-rank)*stats.loc[i,'p_two_sided']))
        stats.loc[i,'p_holm']=previous
    stats['positive_significant']=(stats.gain_ci_low>0)&(stats.p_holm<.05)
    tables.to_csv(study.OUT/'classification_tables.csv',index=False)
    stats.to_csv(study.OUT/'paired_tests.csv',index=False)
    reg.to_csv(study.OUT/'frozen_concept_table.csv',index=False)
    choices=pd.DataFrame(json.loads((study.OUT/'pre_evaluation_seal.json').read_text())['calibration_audits'])
    choices.groupby(['method','selection']).size().unstack(fill_value=0).to_csv(study.OUT/'selected_candidates.csv')
    primary=stats[(stats.task=='regression')|(stats.control=='Base_target')]
    undefined=detail[(detail.method=='Base_target')&(detail.rho==0)&detail.auc.isna()]
    assert (undefined.evaluation_classes==1).all()
    undefined.to_csv(study.OUT/'undefined_auc_single_class_tasks.csv',index=False)
    atomic_json(study.OUT/'reporting_amendment.json',{'statistics_only':True,
        'reason':'Undefined single-class-cohort AUC must be excluded pairwise before bootstrap and sign-flip; initial NaN statistics invalidated and retained.',
        'models_predictions_data_and_cohorts_changed':False,'original_pre_evaluation_seal_preserved':True,
        'accuracy_datasets':72,'auc_datasets':int(tables.datasets_auc.min()),
        'reporter_sha256':sha(__file__)})
    atomic_json(study.OUT/'audit.json',{'status':'PASS','classification_tasks':72,'new_query_rows':int(detail[(detail.method=='A')&(detail.rho==0)].query_rows.sum()),
        'accuracy_datasets':72,'auc_datasets':int(tables.datasets_auc.min()),'undefined_single_class_auc_tasks':len(undefined),
        'classification_prediction_metric_rows':len(detail),'seed':0,'alternative_contexts':5,'context_cap':256,
        'extra_target_labels_per_dataset':64,'B_D_Shortcut_calibration_excluded':True,
        'partial_original_test_reassigned_to_calibration':True,'same_target_labels_Base_control_included':True,
        'target_only_control_included':True,'evaluation_labels_used_only_after_all72_policies_frozen':True,
        'target_labels_used_for_semisynthetic_feature_generation':True,'unlabeled_transduction':True,
        'global_historical_evaluation_data_freshness_claim':False,'original_paper_replication':False,
        'independent_zero_target_robustness_claim':False,'all_eight_primary_ABCD_positive_significant':bool(primary.positive_significant.all()),
        'Holm_family':16,'bootstrap_unit':'dataset after averaging four severities; conditional on frozen policies',
        'Concept_unchanged_no_target_supervision_added':True,'Markdown_bytes':0})
    print(tables.to_string(index=False),flush=True)
    print(stats.to_string(index=False),flush=True)


if __name__=='__main__':main()
