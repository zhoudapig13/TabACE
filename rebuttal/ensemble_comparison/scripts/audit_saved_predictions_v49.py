"""Recompute all 16604 predictions/metric records in the v47 comparison matrix.

Verifies the actual saved arrays against their frozen models, not just summary
CSVs. This establishes consistency; numerical goal checks remain independent.
"""
import json
import numpy as np
import calibration_head_v45 as head
import sparse_specific_v47 as sparse
import clean_role_head_v46 as clean
import reliability_fusion_v33 as reliability
from prepare_data import RUN,atomic_json


def policy(path):return json.loads(path.read_text())['model']


if __name__=='__main__':
    root=sparse.OUT;count=0;cache={};sparse_cache={}
    for budget in [256,1024,4096,8192]:
        for task in ['classification','regression']:
            for block in ['clean','five','seven']:
                folder=head.source.OUT/'banks/query'/str(budget)/task/block
                cohort={str(m['task_id']):m for m in head.source.adaptation.raw.frozen.cohort(task,block)}
                if task=='classification':
                    original=head.source.full.OUT/'policies'/str(budget)/task/block
                    shared=policy(original/'Shared.json')
                    specific={f:policy(original/f'Specific_{f}.json') for f,_ in head.source.adaptation.raw.case_grid(block)}
                    order=json.loads((original/'generic_scale_selection.json').read_text())['permutation'] if block=='clean' else list(range(21))
                else:
                    original=(clean.OUT if block=='clean' else head.OUT)/'policies'/str(budget)/task/block
                    shared=policy(original/'Shared/policy.json')
                    specific={f:policy(original/f'Specific_{f}/policy.json') for f,_ in head.source.adaptation.raw.case_grid(block)}
                    order=json.loads((original/'role_permutation.json').read_text())['permutation'] if block=='clean' else list(range(21))
                    extra={f:policy(root/'policies'/str(budget)/task/block/f'Specific_{f}/policy.json') for f in specific} if block!='clean' else None
                seen=0
                for path in sorted(folder.glob('*/*.npz')):
                    if '.tmp.' in path.name:continue
                    meta=json.loads(path.with_suffix('.json').read_text());meta['data_id']=cohort[str(meta['task_id'])]['data_id']
                    with np.load(path) as z:c=dict(base=z['base'],bank=z['members'][order],y=z['y_target'],mu=float(z['source_mean']),sigma=float(z['source_scale']))
                    c.update(meta=meta,task=task)
                    if task=='classification':
                        sh=reliability.predict(c,shared);sp=reliability.predict(c,specific[meta['family']])
                        frozen=(root/'policies'/str(budget)/task/block)
                        freeze_time=max((frozen/'Shared.json').stat().st_mtime,*[(frozen/f'Specific_{f}.json').stat().st_mtime for f in specific])
                    else:
                        data,_=head.source.adaptation.raw.scaling.bank(task,block,meta,meta['family'],meta['rho']);X=data['X_target']
                        sh=head.predict(c,shared,c['base'],X,cache);sp=head.predict(c,specific[meta['family']],sh,X,cache)
                        if block!='clean':
                            refinement=extra[meta['family']];chosen=refinement['chosen']
                            if chosen:
                                modelpath=refinement['models'][str(meta['task_id'])]
                                if modelpath not in sparse_cache:sparse_cache[modelpath]=sparse.joblib.load(modelpath)
                                expert=head.head_predict(sparse.matrix(X,c,chosen['profile']),sparse_cache[modelpath]);a=chosen['blend'];sp=(1-a)*sp+a*expert
                        freeze_time=(root/'policies'/str(budget)/task/block/'all_policies_frozen.json').stat().st_mtime
                    out=root/'predictions'/str(budget)/task/block/path.relative_to(folder)
                    result=json.loads(out.with_suffix('.json').read_text())
                    assert out.stat().st_mtime>=freeze_time
                    assert result['input_hashes']==meta['source_input_hashes']
                    with np.load(out) as z:
                        assert np.array_equal(z['Base'],c['base']) and np.array_equal(z['y_target'],c['y'])
                        assert np.allclose(z['Shared'],sh,rtol=0,atol=1e-12)
                        assert np.allclose(z['Specific'],sp,rtol=0,atol=1e-12)
                        for name in ['Base','Shared','Specific']:
                            values=head.risk.metrics(c['y'],z[name],task,meta['scale'])
                            for key,value in values.items():
                                stored=result['methods'][name][key]
                                assert value is None and stored is None or value is not None and stored is not None and abs(value-stored)<1e-12
                    seen+=1;count+=1
                expected=len(cohort)*len(head.source.adaptation.raw.case_grid(block));assert seen==expected
                print('SAVED_GROUP_VERIFIED',budget,task,block,seen,flush=True)
    assert count==16604
    atomic_json(root/'saved_prediction_audit.json',dict(status='PASS',verified_conditions=count,
        frozen_models_reproduce_every_saved_prediction=True,all_metrics_recomputed=True,
        query_outputs_written_after_policy_freeze=True,Base_arrays_equal_paired_bank=True,
        full_goal_metric_checks_required=48,consistency_audit_does_not_imply_goal_satisfied=True,
        extra_clean_TRAIN_supervision=True,development_only=True,original_paper_replication=False))
    print('FULL_SAVED_PREDICTION_AUDIT_PASS',count,flush=True)
