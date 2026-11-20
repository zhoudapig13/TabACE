"""Independent exported-input/prediction/metric audit for the development suite."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import RUN,atomic_json,protocol
import benchmark256_v22 as suite
import tabicl_base256_calibration_v22 as base
import tabicl_semisynthetic256_v22 as semi

OUT=suite.OUT


def teacher_audit():
    from sklearn.linear_model import Ridge
    cfg=suite.configuration()['regression_clean'];rows=[]
    for meta in base.manifest():
        data,_=base.clean_pool(meta)
        g=semi.prepare(meta);p=semi.OUT/'generators'/f'{meta["task_id"]}.json';m=json.loads(p.read_text())
        assert not m['original_query_labels_used'] and not m['TabICL_predictions_used_for_labels']
        assert not np.intersect1d(g['source_row_ids'],g['query_row_ids']).size
        X=data['X_source'].astype(np.float64);Q=data['X_target'].astype(np.float64)
        mean=X.mean(0);sd=np.maximum(X.std(0),1e-6)
        Z=np.clip((X-mean)/sd,-5,5);V=np.clip((Q-mean)/sd,-5,5)
        cols=np.asarray(m['selected_original_columns']);active=np.asarray(m['teacher_active_original_columns'])
        y=data['y_source'];target=(y-y.mean())/max(float(y.std()),1e-8)
        fit=Ridge(alpha=1.).fit(Z[:,active],target)
        assert np.allclose(fit.coef_,m['teacher_coefficients'],atol=1e-12,rtol=1e-12)
        signal=fit.predict(Z[:,active]);qs=fit.predict(V[:,active])
        assert np.array_equal(g['X_source_pool'],Z[:,cols].astype(np.float32))
        assert np.array_equal(g['X_query'],V[:,cols].astype(np.float32))
        assert np.allclose(g['signal_source'],(signal-signal.mean())/signal.std(),atol=1e-12,rtol=1e-12)
        assert np.allclose(g['signal_query'],(qs-signal.mean())/signal.std(),atol=1e-12,rtol=1e-12)
        streams=np.random.SeedSequence([0,int(meta['task_id']),2201]).spawn(3)
        assert np.array_equal(g['epsilon_source'],np.random.default_rng(streams[0]).normal(size=len(signal)))
        assert np.array_equal(g['epsilon_query'],np.random.default_rng(streams[1]).normal(size=len(qs)))
        assert float(g['dataset_noise_draw'])==float(np.random.default_rng(streams[2]).normal())
        sigma=cfg['mu']*np.exp(cfg['h']*float(g['dataset_noise_draw']))
        scale=float((g['signal_source']+sigma*g['epsilon_source']).std())
        rows.append({'task_id':meta['task_id'],'context_rows':len(g['context_indices']),
                     'source_query_disjoint':True,'teacher_refit_matches':True,'nrmse_scale':scale})
    atomic_json(OUT/'teacher_generator_audit.json',{'status':'PASS','datasets':rows,
                'original_query_labels_not_used_for_teacher':True,'TabICL_predictions_not_used_for_labels':True})
    return {r['task_id']:r['nrmse_scale'] for r in rows}


def audit():
    from run_tabicl import metrics
    cfg=suite.configuration();scales=teacher_audit();rows=[]
    files=[p for p in (OUT/'input_banks').glob('*/*/*/*.json') if not p.name.endswith('.tmp.json')]
    assert len(files)==4151
    for path in files:
        m=json.loads(path.read_text())
        assert m['context_rows']==256 and m['n_estimators']==1 and m['seed']==0
        with np.load(path.with_suffix('.npz')) as z:
            assert len(z['y_source'])==256
            for key in ['X_source','y_source','X_target','y_target']:
                assert np.isfinite(z[key]).all() and protocol.array_sha256(z[key])==m['input_hashes'][key]
            p=z['Base_prediction'];assert np.isfinite(p).all()
            if m['task']=='classification':assert (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6)
            if m['task']=='regression' and m['block'] in ['clean','five']:
                assert np.isclose(m['scale'],scales[m['task_id']],atol=1e-12,rtol=1e-12)
            if m['task']=='regression' and m['block']=='seven':assert m['scale']==max(float(z['y_target'].std()),1e-12)
            calculated=metrics(z['y_target'],p,m['task'],m['scale'])
            for key,value in calculated.items():
                if value is not None:assert abs(value-m['metrics'][key])<1e-10
            origin=RUN/m['prediction_origin']
            with np.load(origin) as old:
                assert np.array_equal(p,old[m['prediction_key']])
                assert np.array_equal(z['y_target'],old['y_target'])
        rows.append({k:m[k] for k in ['task','block','task_id','family','rho','task_provenance']}|calculated)
    frame=pd.DataFrame(rows)
    assert not frame.duplicated(['task','block','task_id','family','rho']).any()
    summary=[];checks=[]
    expected={'classification':{'clean':72,'five':1440,'seven':980},'regression':{'clean':35,'five':700,'seven':924}}
    for (task,block),g in frame.groupby(['task','block'],sort=False):
        assert len(g)==expected[task][block]
        keys=['accuracy','auc'] if task=='classification' else ['r2','nrmse']
        row={'task':task,'block':block,'conditions':len(g),'datasets':g.task_id.nunique(),'context_rows':256,'seed':0}
        for key in keys:
            value=float(g[key].mean());target=cfg['targets'][task][block][key];delta=value-target
            checks.append({'task':task,'block':block,'metric':key,'paper_reference':target,
                           'observed':value,'difference':delta,'within_005':abs(delta)<=.005})
            row[key]=value
        summary.append(row)
    assert len(checks)==12 and all(r['within_005'] for r in checks)
    for relative,expected_sha in cfg['reference_files_sha256'].items():assert base.sha(RUN/relative)==expected_sha
    assert (RUN.parents[1]/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    frame.to_csv(OUT/'Base_detail.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'Base_averages.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'Base_paper_comparison.csv',index=False)
    atomic_json(OUT/'metric_audit.json',{'status':'PASS','input_conditions':4151,
                'checks':checks,'all_12_within_005':True,'records':summary,
                'teacher_generator_verified':True,'reference_files_unchanged':True,
                'model_weights_not_trained':True,'Shared_Specific_not_run':True,
                'result_role':'development calibration, real-label and explicitly semisynthetic tracks'})
    print('METRIC_AUDIT_PASS',json.dumps(summary),flush=True)


def fresh():
    import torch
    from run_tabicl import Predictor,CHECKPOINTS
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    results=[]
    for task in ['classification','regression']:
        predictor=Predictor(task,'cuda:0')
        for block in ['clean','five','seven']:
            meta=suite.cohort(task,block)[0]
            grid=[('original',0)] if block=='clean' else [(f,20) for f in protocol.FAMILIES] if block=='five' else [('covariate',20)]
            for family,rho in grid:
                path=OUT/'input_banks'/task/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                m=json.loads(path.with_suffix('.json').read_text())
                with np.load(path) as z:
                    p,_=predictor(z['X_source'],z['y_source'],z['X_target'],members=1,random_state=0,
                                  n_classes=len(np.unique(z['y_source'])) if task=='classification' else None)
                    scale=1. if task=='classification' else max(1.,float(z['y_target'].std()))
                    err=float(np.max(np.abs(p-z['Base_prediction'])))/scale
                    assert err<1e-5,(task,block,family,err)
                results.append({'task':task,'block':block,'family':family,'rho':rho,'normalized_max_error':err})
    checkpoints={str(CHECKPOINTS/f'tabicl-{kind}-v2-20260212.ckpt'):base.sha(CHECKPOINTS/f'tabicl-{kind}-v2-20260212.ckpt')
                 for kind in ['classifier','regressor']}
    atomic_json(OUT/'fresh_inference_audit.json',{'status':'PASS','n_estimators':1,'context_rows':256,
                'seed':0,'cases':results,'checkpoint_sha256':checkpoints,'backbone_weights_updated':False})
    print('FRESH_INFERENCE_PASS',len(results),'cases; max error',max(r['normalized_max_error'] for r in results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','fresh']);a=p.parse_args()
    if a.phase=='audit':audit()
    else:fresh()
