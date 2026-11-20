"""Verify every changed bank against its paired, frozen development Base."""
import json
import numpy as np
import recovery_contexts_v44 as p
from prepare_data import atomic_json,protocol

if __name__=='__main__':
    p.verify_registry();count=0;calls=0;aliases=0
    for block,meta in p.adaptation.raw.repaired.prior.jobs():
        split=p.split(block,meta);pool=p.adaptation.raw.scaling.prepare('regression',block,meta)
        assert not np.intersect1d(split['source_original_ids'],split['validation_original_ids']).size
        assert not np.intersect1d(split['validation_original_ids'],pool['query_ids']).size
        for role in ['validation','query']:
            for budget in [256,1024,4096,8192]:
                for family,rho in p.adaptation.raw.case_grid(block):
                    path=p.OUT/'banks'/role/str(budget)/'regression'/block/str(meta['task_id'])/f'{family}_{rho}.npz'
                    record=json.loads(path.with_suffix('.json').read_text())
                    if role=='validation':
                        expected,truth,_,_=p.validation_base(block,meta,family,rho,budget)
                    else:
                        root=p.five.OUT if block=='five' else p.seven.OUT
                        with np.load(root/'query'/str(budget)/block/str(meta['task_id'])/path.name) as z:expected=z['prediction'];truth=z['y_target']
                    with np.load(path) as z:
                        assert np.array_equal(z['base'],expected) and np.array_equal(z['y_target'],truth)
                        assert z['members'].shape==(21,len(truth)) and np.isfinite(z['members']).all()
                        assert np.isfinite(z['source_mean']) and float(z['source_scale'])>0
                    assert record['source_input_hashes']['y_target']==protocol.array_sha256(truth)
                    assert record['Specific_contains_all_Shared'] and not record['query_labels_used_by_constructors']
                    assert len(record['shared_roles'])==10 and len(record['specific_additional_roles'])==11
                    assert len(record['member_input_hashes'])==21 and all(len(key)==3 for key in record['member_input_hashes'])
                    assert len(record['context_rows'])==21 and all(0<n<=record['actual_context']<=budget for n in record['context_rows'])
                    assert record['provisional_Base_calibration'] is False
                    calls+=record['new_TFM_calls'];aliases+=sum(record['aliases']);count+=1
    assert count==12992
    atomic_json(p.OUT/'bank_audit.json',dict(status='PASS',verified_changed_bank_conditions=count,
        all_Base_predictions_equal_frozen_registry=True,all_truths_and_row_caps_verified=True,
        shared_members=10,specific_members=21,all_TRAIN_calibration_splits_source_query_disjoint=True,
        new_TFM_calls=calls,exact_input_aliases=aliases,original_query_labels_used_for_construction=False,
        development_only=True,original_paper_replication=False))
    print('RECOVERY_BANK_AUDIT_PASS',count,'TFM_calls',calls,'exact_aliases',aliases,flush=True)
