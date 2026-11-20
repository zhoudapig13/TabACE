"""Report calibration-selected transfer results with frozen B/D audits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import abcd_transfer_calibration_v109 as study
import abcd_fresh_confirmation_v101 as fresh
from prepare_data import atomic_json,RUN
from run_tabicl import metrics
from report_abcd_v84 import paired_statistics


def main():
    OUT=study.OUT
    assert json.loads((OUT/'execution.json').read_text())['status']=='COMPLETE'
    assert json.loads((OUT/'plan.json').read_text())['code_sha256']==study.classification.origin.sha(study.__file__)
    seal=json.loads((OUT/'policy_seal.json').read_text())
    assert len(seal)==308 and all(study.classification.origin.sha(OUT/p)==h for p,h in seal.items())
    # Compatibility alias for the shared regression reporter, without changing
    # any fitted policy file or the existing seal paths.
    alias=OUT/'folds'
    if not alias.exists():alias.symlink_to('regression',target_is_directory=True)
    study.algorithm.OUT=OUT;study.algorithm.cells=study.inputs.load_cells
    study.algorithm.report()
    rows=[];class_checks=0;control_checks=0
    for path in (study.classification.cls.banks.split.OUT/'inputs').glob('*.npz'):
        tid=int(path.stem);meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:data={k:z[k] for k in z.files}
        data['data_id']=meta['data_id']
        for method,seen,matched in study.classification.METHODS:
            saved=json.loads((OUT/'classification'/str(tid)/f'{method}_policy.json').read_text());p=saved['policy']
            if not seen:
                old=json.loads((study.classification.OUT/'classification'/str(tid)/f'{method}_policy.json').read_text())
                assert p==old['policy'];control_checks+=1
            else:
                validation=[c for c in study.classification.cls.cells_for(tid,'validation',matched,True,data) if c['meta']['family'] in ('clean','shortcut')]
                for trial in saved['trials']:
                    value=study.classification.score(validation,trial['policy'])
                    assert np.isclose(value['primary'],trial['score']['primary'],atol=1e-12)
                    assert np.isclose(value['secondary'],trial['score']['secondary'],atol=1e-12)
                best=max(saved['trials'],key=lambda t:(t['score']['primary'],t['score']['secondary']))
                assert best['candidate']==saved['selected_candidate'];class_checks+=1
            for cohort in ('old_development','additional_development'):
                if cohort=='old_development':
                    truth=study.classification.cls.banks.split.OUT/'evaluation_truth'/f'{tid}.npz'
                    cells=study.classification.cls.cells_for(tid,'evaluation',matched,seen,data)
                else:
                    truth=fresh.OUT/'truth/classification'/f'{tid}.npz'
                    if not truth.exists():continue
                    cells=[]
                    for rho in (0,20,50,80,100):
                        with np.load(fresh.OUT/'banks/classification'/str(tid)/f'{rho}.npz') as z:
                            bank=z['members'].copy()
                            if not matched:bank[3]=z['bootstrap']
                            cells.append({'task':'classification','base':z['base'],'bank':bank,'zero':z['zero'],'meta':{'rho':rho}})
                with np.load(truth) as z:y=z['y_evaluation']
                for c in cells:
                    rho=c['meta']['rho'];prediction=study.classification.predict(c,p)
                    if not seen:
                        oldpred=(study.classification.OUT/'development_predictions/classification' if cohort=='old_development' else fresh.OUT/'predictions/classification')/str(tid)/f'{rho}_{method}.npz'
                        with np.load(oldpred) as z:assert np.array_equal(prediction,z['prediction'])
                    preds={method:prediction}
                    if method=='A':preds.update(Base=c['base'],Base_zero=c['zero'],Target_only=c['bank'][1])
                    for name,q in preds.items():
                        dest=OUT/'classification_predictions'/cohort/str(tid)/f'{rho}_{name}.npz'
                        dest.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(dest,prediction=q)
                        rows.append({'cohort':cohort,'task_id':tid,'data_id':meta['data_id'],'rho':rho,'method':name,
                            **metrics(y,q,'classification',1.)})
    detail=pd.DataFrame(rows);detail.to_csv(OUT/'classification_detail.csv',index=False)
    tables=[];tests=[]
    for cohort,g in detail.groupby('cohort'):
        values=g[g.rho>0].groupby(['data_id','method'])[['accuracy','auc']].mean()
        clean=g[g.rho==0].groupby('method')[['accuracy','auc']].mean()
        for method in g.method.unique():
            v=values.xs(method,level='method');tables.append({'cohort':cohort,'method':method,'datasets':len(v),
                'accuracy':v.accuracy.mean(),'auc':v.auc.mean(),'clean_accuracy':clean.loc[method,'accuracy'],'clean_auc':clean.loc[method,'auc']})
        for a,b in study.classification.CONTRASTS:
            delta=values.xs(a,level='method').accuracy-values.xs(b,level='method').accuracy
            tests.append({'cohort':cohort,'contrast':a+'-'+b,**paired_statistics(delta)})
    table=pd.DataFrame(tables);stats=pd.DataFrame(tests)
    table.to_csv(OUT/'classification_tables.csv',index=False);stats.to_csv(OUT/'classification_contrasts.csv',index=False)
    reg_controls=0
    for f in (OUT/'regression').glob('*/*_policy.json'):
        if f.stem not in ('B_policy','D_policy'):continue
        original=study.regression.OUT/'folds'/f.parent.name/f.name
        assert json.loads(f.read_text())['policy']==json.loads(original.read_text())['policy'];reg_controls+=1
    old=pd.read_csv(study.regression.OUT/'tables.csv');new=pd.read_csv(OUT/'tables.csv')
    for row in new[new.method.isin(['B','D'])].to_dict('records'):
        previous=old[(old.cohort==row['cohort'])&(old.method==row['method'])].iloc[0]
        assert np.isclose(previous.r2,row['r2'],atol=1e-12)
    assert (RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(OUT/'transfer_audit.json',{'status':'PASS','classification_validation_selections_recomputed':class_checks,
        'classification_frozen_control_models':control_checks,'regression_frozen_control_models':reg_controls,
        'B_D_classification_predictions_bitwise_equal_to_previous':True,
        'B_D_regression_metrics_equal_to_previous':True,'all_policy_hashes_unchanged':True,
        'no_requested_order_in_loss_or_selection':True,'previously_evaluated_development_cohorts':True,
        'independent_confirmation':False,'Markdown_bytes':0})
    print(table.to_string(index=False),flush=True);print(stats.to_string(index=False),flush=True)


if __name__=='__main__':main()
