"""Recompute saved regression predictions and verify preserved artifacts."""
import argparse
import hashlib
import json
import time
import numpy as np
from prepare_data import RUN,ROOT,atomic_json
from run_tabicl import metrics


def check(actual,expected,where):
    for k in ['r2','nrmse']:
        assert np.isclose(actual[k],expected[k],rtol=1e-10,atol=1e-10),(str(where),k,actual[k],expected[k])


def audit(name):
    out=RUN/'results'/name;summary=json.loads((out/'latest_summary.json').read_text());count=0;rows_by_method={}
    for track,expected in [('five',735),('seven',957)]:
        metadata={};base_rows=[]
        paths=sorted((out/'ensemble_banks'/track).glob('*/*.npz'));assert len(paths)==expected
        for p in paths:
            meta=json.loads(p.with_suffix('.json').read_text());key=(int(meta['task_id']),meta['family'],int(meta['rho']))
            assert key not in metadata;metadata[key]=meta
            assert meta['context_count']==6 and meta['backbone_weights_frozen']
            assert not meta['additional_history_reference_rows_used']
            with np.load(p) as z:
                assert np.array_equal(z['feature_bank'][0],z['prediction__Base'])
                assert np.array_equal(z['iqr_bank'][0],z['prediction__Base'])
                for method,expected_values in meta['methods'].items():
                    actual=metrics(z['y_target'],z['prediction__'+method],'regression',meta['scale'])
                    check(actual,expected_values,p);count+=1
                    if method=='Base':base_rows.append({'rho':meta['rho'],**actual})
        frozen=json.loads((out/f'frozen_{track}.json').read_text())
        actual={k:float(np.mean([r[k] for r in base_rows if r['rho']>0])) for k in ['r2','nrmse']}
        check(actual,frozen['metrics'],track+' Base')
        if track=='five':
            clean={k:float(np.mean([r[k] for r in base_rows if r['rho']==0])) for k in ['r2','nrmse']}
            check(clean,json.loads((out/'frozen_clean.json').read_text())['metrics'],'Clean Base')
        for folder,predfolder,refined in [('policies','predictions',False),('refined_policies','refined_predictions',True)]:
            for p in sorted((out/folder/track).glob('*.json')):
                policy=json.loads(p.read_text());assert set(policy['train_data_ids']).isdisjoint(policy['test_data_ids'])
                if refined:
                    assert not policy['test_labels_used_for_fit'] and not policy['Shared_bounds_used_for_fit']
                else:
                    assert not policy['test_labels_used_for_method_fitting'] and not policy['historical_Shared_bounds_used_for_selection']
                method=policy['method']
                for stage,keyrows in [('adapted','rows')]+([] if refined else [('raw','raw_rows')]):
                    for row in policy[keyrows]:
                        key=(int(row['task_id']),row['family'],int(row['rho']));meta=metadata[key]
                        path=out/predfolder/track/(method if refined else method+'_'+stage)/str(row['task_id'])/f"{row['family']}_{row['rho']}.npz"
                        with np.load(path) as z:actual=metrics(z['y_target'],z['prediction'],'regression',meta['scale'])
                        check(actual,row,path);count+=1
                        if stage=='adapted':rows_by_method.setdefault((track,method),[]).append({'rho':row['rho'],**actual})
                        if stage=='raw' and method=='Native6':rows_by_method.setdefault((track,'Native6_original'),[]).append({'rho':row['rho'],**actual})
        for method in ['PHE6','RF_PFN6']:
            paths=list((out/'references'/method/track).glob('*/*.npz'))
            if not paths:continue
            assert len(paths)==expected
            for path in paths:
                meta=json.loads(path.with_suffix('.json').read_text())
                assert not meta['target_labels_used_for_fit'] and meta['backbone_weights_frozen']
                with np.load(path) as z:actual=metrics(z['y_target'],z['prediction'],'regression',meta['scale'])
                check(actual,meta['metrics'],path);count+=1
                rows_by_method.setdefault((track,method),[]).append({'rho':meta['rho'],**actual})
    for row in summary['records'][1:]:
        for track,expected in [('five',735),('seven',957)]:
            values=rows_by_method[(track,row['method'])];assert len(values)==expected
            check({k:float(np.mean([r[k] for r in values if r['rho']>0])) for k in ['r2','nrmse']},row[track],row['method']+' '+track)
            if track=='five':check({k:float(np.mean([r[k] for r in values if r['rho']==0])) for k in ['r2','nrmse']},row['clean'],row['method']+' clean')
    lock=json.loads((RUN/'results/internal_regression_component_v1/classification_snapshot_lock.json').read_text())
    for path,sha in lock['files_sha256'].items():assert hashlib.sha256((RUN/path).read_bytes()).hexdigest()==sha,path
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('frozen_*.json')}
    atomic_json(out/'prediction_audit.json',{'status':'PASS','recomputed_prediction_vectors':count,
        'full_grid_counts':{'five':735,'seven':957},'reported_means_recomputed_from_saved_predictions':True,
        'six_candidate_bank_shapes_and_Base_membership_verified':True,'dataset_fit_test_disjointness_verified':True,
        'classification_snapshot_unchanged':True,'Markdown_bytes':0,'frozen_config_sha256':frozen,
        'paper_protocol_reproduction':False,'historical_Shared_not_recomputed':True,'audited_unix':time.time()})
    print('PREDICTION_AUDIT_PASS',name,count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='internal_regression_component_v1');a=p.parse_args();audit(a.name)
