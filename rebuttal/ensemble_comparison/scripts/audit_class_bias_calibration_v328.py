"""Verify source-context audit coverage, LOO class bias fits and held-out scores."""
import argparse
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
import class_bias_calibration_v327 as run
from prepare_data import ROOT,atomic_json
from run_tabicl import metrics


def main(pilot=False,partial=False):
    source=run.source;out=run.OUT;plan=json.loads((out/'plan.json').read_text())
    assert source.sha(run.__file__)==plan['script_sha256']
    assert source.sha(run.calibration.__file__)==plan['calibration_sha256']
    assert source.sha(run.parent.selector.__file__)==plan['selector_sha256']
    assert source.sha(run.parent.OUT/'plan.json')==plan['parent_plan_sha256']
    allcases=json.loads((run.parent.oldbank.split.OUT/'case_manifest.json').read_text())
    cases=[allcases[i] for i in run.parent.PILOT_POSITIONS] if pilot else allcases
    prefix='pilot_' if pilot else ''
    auditpath=run.parent.OUT/f'{prefix}independent_audit.json'
    if partial and not auditpath.exists():auditpath=run.parent.OUT/f'{prefix}partial_independent_audit.json'
    upstream=json.loads(auditpath.read_text());assert upstream['status']=='PASS'
    coveragefile=run.parent.OUT/auditpath.name.replace('independent_audit.json','condition_metrics.csv')
    coverage=pd.read_csv(coveragefile)
    covered=set(coverage[['track','task','task_id','family','rho']].itertuples(index=False,name=None))
    rows=[];count=0;class_count=0
    for c in cases:
        track,task,tid=c['track'],c['task'],c['task_id']
        assert source.sha(c['adaptation_file'])==c['adaptation_sha256']
        with np.load(c['adaptation_file']) as z:apos,epos,ay=(z[k] for k in ('adaptation_positions','evaluation_positions','adaptation_y'))
        assert set(apos).isdisjoint(epos)
        for family,rho in [('clean',0)]+[(f,r) for f in c['families'] for r in source.novel.STRENGTHS]:
            path=out/'predictions'/track/task/str(tid)/f'{family}_{rho}.npz'
            key=(track,task,tid,family,rho)
            if not path.with_suffix('.json').exists() or key not in covered:
                assert partial,(key,'missing prediction or upstream context audit coverage')
                continue
            meta=json.loads(path.with_suffix('.json').read_text());assert source.sha(path)==meta['output_sha256']
            assert meta['adaptation_sha256']==c['adaptation_sha256'] and meta['adaptation_positions']==apos.tolist() and meta['evaluation_positions']==epos.tolist()
            assert not meta['class_bias_features_depend_on_target_labels'] and not meta['evaluation_labels_opened']
            parentpath=run.parent.OUT/'predictions'/track/task/str(tid)/path.name
            pm=json.loads(parentpath.with_suffix('.json').read_text());assert source.sha(parentpath)==meta['parent_output_sha256']==pm['output_sha256']
            with np.load(path) as actual,np.load(parentpath) as upstream_pred:
                if task=='classification':
                    oldfile=run.parent.oldbank.OUT/'predictions'/track/task/str(tid)/path.name
                    oldmeta=json.loads(oldfile.with_suffix('.json').read_text())
                    assert not oldmeta['target_labels_opened'] and oldmeta['target_labels_used_in_context']==0
                    originalfile=source.OUT/'predictions'/track/task/str(tid)/path.name
                    assert source.sha(oldfile)==pm['previous_bank_sha256'] and source.sha(originalfile)==pm['original_prediction_sha256']
                    with np.load(oldfile) as old,np.load(originalfile) as original:
                        expected,policies=run.adapt_classification(c,upstream_pred,old,original,ay,apos,epos)
                    for name,v in expected.items():np.testing.assert_allclose(actual[name],v,atol=1e-12,rtol=0)
                    for name in ['Native','Shared']:
                        for suffix in ['_policy','_bias_policy']:
                            p=policies[name+suffix];saved=meta[name+suffix]
                            if suffix=='_bias_policy':
                                assert p['kind']==saved['kind'] and p['source']==saved['source']
                                np.testing.assert_allclose(p['offset'],saved['offset'],atol=1e-12,rtol=0)
                                if p['kind']=='class_intercept':assert p['penalty']==saved['penalty']
                            else:assert p['weights']==saved['weights']
                            np.testing.assert_allclose(p['score'],saved['score'],atol=1e-12,rtol=0)
                    class_count+=1
                else:
                    assert actual.files==upstream_pred.files
                    for name in actual.files:np.testing.assert_array_equal(actual[name],upstream_pred[name])
                assert source.sha(c['heldout_truth_file'])==c['heldout_truth_sha256']
                with np.load(c['heldout_truth_file']) as z:truth=z['y_target']
                names=['Base','Native_finetuned','Shared_finetuned','Native_reference_only','Shared_reference_only','Native_target_context','Shared_target_context','Native_adapted','Shared_adapted']
                if task=='classification':names+=['Native_calibrated','Shared_calibrated']
                for name in names:
                    assert np.isfinite(actual[name]).all() and len(actual[name])==len(truth)
                    rows.append({'track':track,'task':task,'task_id':tid,'data_id':c['data_id'],'family':family,'rho':rho,'method':name,
                        'target_adaptation_labels_available':len(ay) if name.endswith('_adapted') or name.endswith('_target_context') or name.endswith('_calibrated') else 0,
                        **metrics(truth,actual[name],task,max(float(truth.std()),1e-12))})
            count+=1
        print('BIAS_CALIBRATION_AUDIT_CASE',track,task,tid,flush=True)
    expected=sum(1+4*len(c['families']) for c in cases)
    if not partial:assert count==expected
    assert count>0
    prefix=('pilot_' if pilot else '')+('partial_' if count<expected else '')
    df=pd.DataFrame(rows);df.to_csv(out/f'{prefix}condition_metrics.csv',index=False)
    keys=['track','task','family','method'];columns=['accuracy','auc','r2','nrmse','target_adaptation_labels_available']
    valid=df[df.rho>0].groupby(keys+['data_id']).rho.nunique().reset_index(name='strengths')
    valid=valid[valid.strengths==4][keys+['data_id']]
    df=pd.concat([df[df.rho==0],df[df.rho>0].merge(valid,on=keys+['data_id'])],ignore_index=True)
    ds=df.groupby(keys+['data_id'])[columns].mean().reset_index()
    table=ds.groupby(keys)[columns].mean().reset_index()
    table=table.merge(ds.groupby(keys).size().rename('datasets').reset_index(),on=keys)
    table.to_csv(out/f'{prefix}family_averages.csv',index=False)
    gains=[]
    for (track,task,family),g in table[table.family!='clean'].groupby(['track','task','family']):
        metric='accuracy' if task=='classification' else 'r2';value=lambda name:float(g.loc[g.method==name,metric].iloc[0])
        baseline=value('Base');adapted=value('Shared_adapted');threshold=.05 if track=='perturbation' else .02
        gains.append({'track':track,'task':task,'family':family,'datasets':int(g.datasets.iloc[0]),'Base':baseline,'Shared_adapted':adapted,
            'gain':adapted-baseline,'gain_vs_same_budget_Native':adapted-value('Native_adapted'),
            'required_gain_strictly_greater_than':threshold,'passed':adapted-baseline>threshold})
    atomic_json(out/f'{prefix}numerical_acceptance.json',{'pilot_only':pilot,'conditions':count,'family_results':gains,
        'passed_count':sum(g['passed'] for g in gains),'all_twelve_pass':not pilot and count==2175 and len(gains)==12 and all(g['passed'] for g in gains),
        'extra_clean_source_and_target_subset_supervision':True,'classification_bias_fit_uses_label_independent_features':True,
        'not_original_zero_shot_Shared':True,'seed':0})
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(out/f'{prefix}independent_audit.json',{'status':'PASS','pilot_only':pilot,'complete':count==expected,'conditions':count,
        'class_bias_conditions_recomputed':class_count,'upstream_context_audit_sha256':source.sha(auditpath),
        'class_bias_features_independent_of_target_labels':True,'regression_arrays_bitwise_unchanged':True,
        'all_new_head_and_deployment_policies_reselected_on_adaptation_labels_only':True,
        'source_vs_adaptation_vs_evaluation_row_separation_inherited_from_upstream_audit':True,
        'Markdown_bytes':0,'audit_script_sha256':source.sha(__file__)})
    print(table.to_string(index=False),flush=True);print(json.dumps(gains,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.pilot,a.partial)
