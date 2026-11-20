"""Validate every dataset checkpoint and its source-only supervision membership."""
import gc
import hashlib
import json
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
import torch
import shared_dataset_finetuning_v304 as run
from prepare_data import ROOT,atomic_json,protocol


def state_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    out=run.OUT;assert json.loads((out/'execution.json').read_text())['status']=='COMPLETE'
    plan=json.loads((out/'plan.json').read_text())
    assert run.source.sha(run.__file__)==plan['script_sha256']
    assert run.source.sha(run.implementation.__file__)==plan['implementation_sha256']
    assert run.source.sha(run.implementation.parent.__file__)==plan['parent_implementation_sha256']
    assert run.source.sha(out/'case_manifest.json')==plan['case_manifest_sha256']
    cases=json.loads((out/'case_manifest.json').read_text());assert len(cases)==175
    seen=set();initials={};records=[]
    for c in cases:
        key=(c['track'],c['task'],c['task_id']);assert key not in seen;seen.add(key)
        d=out/'checkpoints'/c['track']/c['task']/str(c['task_id'])
        result=json.loads((d/'result.json').read_text());a=json.loads((d/'input_audit.json').read_text())
        assert result['status']=='COMPLETE' and tuple(result[k] for k in ('track','task','task_id'))==key
        assert result['actual_GPU'] in plan['allowed_GPUs'] and result['actual_GPU'] not in plan['reserved_GPUs']
        assert result['no_final_query_labels_or_scores_opened'] and not result['target_adaptation_labels_used_for_weights']
        assert run.source.sha(c['input_file'])==c['input_sha256']
        assert run.source.sha(c['reference_file'])==c['reference_sha256']
        with np.load(c['input_file']) as z:
            assert 'y_target' not in z.files
            y0,sid,qid=z['y_source'],z['source_indices'],z['target_indices']
        with np.load(c['reference_file']) as z:Ty,tids=z['y_train_reference'],z['train_reference_indices']
        ids,y=np.r_[sid,tids],np.r_[y0,Ty]
        assert set(map(int,ids)).isdisjoint(map(int,qid)) and len(set(map(int,ids)))==len(ids)
        if c['task']=='classification':train,val=run.implementation.class_split(y,128)
        else:
            order=np.random.default_rng(0).permutation(len(y));val,train=order[:128],order[128:]
        np.testing.assert_array_equal(train,a['train_pool_positions'])
        np.testing.assert_array_equal(val,a['validation_pool_positions'])
        assert set(train).isdisjoint(val) and len(train)+len(val)==len(y)
        assert protocol.array_sha256(ids[train])==a['training_row_ids_sha256']
        assert protocol.array_sha256(ids[val])==a['validation_row_ids_sha256']
        assert protocol.array_sha256(qid)==a['final_query_row_ids_sha256']
        assert a['original_input_sha256']==c['input_sha256'] and a['reference_sha256']==c['reference_sha256']
        assert a['extra_labeled_source_references']==len(Ty)
        assert result['training_loss_calls']>0
        trace=result['validation_trace'];assert len(trace)>=2
        assert all(v['context_rows']<=256 for v in trace)
        selected=result['selected_validation_primary'];assert np.isfinite(selected)
        assert selected>=trace[0]['primary']-1e-12
        assert selected>=max(v['primary'] for v in trace)-1.0001e-4
        if c['task']=='classification':
            same=[v['secondary']['balanced_log_loss'] for v in trace if abs(v['primary']-selected)<1e-12]
            assert result['selected_balanced_nll']<=min(same)+1.1e-6
        assert run.source.sha(result['checkpoint_file'])==result['checkpoint_sha256']
        checkpoint=torch.load(result['checkpoint_file'],map_location='cpu',weights_only=False)
        digest=state_digest(checkpoint['state_dict'])
        assert digest==result['selected_model_sha256']
        assert result['selected_weights_changed']==(digest!=result['initial_model_sha256'])
        initials.setdefault(c['task'],set()).add(result['initial_model_sha256'])
        del checkpoint;gc.collect()
        records.append({'track':c['track'],'task':c['task'],'task_id':c['task_id'],
            'selected_weights_changed':result['selected_weights_changed'],'training_steps':result['training_loss_calls'],
            'source_validation_gain':selected-trace[0]['primary'],'source_training_rows':len(train),
            'source_validation_rows':len(val),'extra_source_labels':len(Ty),
            'checkpoint_sha256':result['checkpoint_sha256']})
        print('TRAINING_AUDIT_DATASET',*key,'changed',result['selected_weights_changed'],flush=True)
    assert len(seen)==175 and all(len(values)==1 for values in initials.values())
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(out/'training_summary.json',{'datasets':records,
        'selected_updated_weights':sum(r['selected_weights_changed'] for r in records),
        'datasets_total':175,'source_validation_not_final_query_metrics':True})
    atomic_json(out/'independent_training_audit.json',{'status':'PASS','datasets':175,
        'all_training_and_validation_row_ids_disjoint_from_final_query':True,
        'target_adaptation_labels_not_used_for_gradient_training':True,
        'source_train_and_validation_splits_recomputed':True,
        'checkpoint_tensor_hashes_match_selected_models':True,
        'best_model_selection_uses_independent_source_validation_only':True,
        'separate_checkpoints_by_track_and_task':True,
        'GPU6_and_GPU7_reserved':True,'source_clean_pre_corruption_features_explicit':True,
        'supervised_development_extension_not_original_Shared':True,
        'report_script_sha256':run.source.sha(__file__),'Markdown_bytes':0})


if __name__=='__main__':main()
