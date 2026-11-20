"""Fixed label-adaptation and evaluation partitions for all six mechanisms."""
import json
import numpy as np
import shared_dataset_finetuning_v307 as training
from prepare_data import RUN,ROOT,atomic_json

source=training.source
OUT=RUN/'results/task_adaptation_split_v311'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan={'seed':0,'scope':'all175 track/task pairs, three perturbations and three shifts',
        'adaptation_labels':'min(64,max(16,floor(original_query_count*.25)))',
        'split':'fixed seed0 permutation independent of labels, predictions and scores',
        'same_split_across_mechanisms_and_strengths':True,
        'adaptation_rows_removed_from_final_evaluation':True,
        'per_dataset_mechanism_strength_hyperparameters_allowed':True,
        'selection_uses_adaptation_labels_only':True,
        'Base_and_Shared_same_remaining_eval_rows':True,'script_sha256':source.sha(__file__),'Markdown_written':False}
    p=OUT/'plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)
    manifest=[]
    for c in json.loads((training.OUT/'case_manifest.json').read_text()):
        assert source.sha(c['input_file'])==c['input_sha256']
        with np.load(c['input_file']) as z:qid=z['target_indices'];assert 'y_target' not in z.files
        assert source.sha(c['truth_file'])==c['truth_sha256']
        with np.load(c['truth_file']) as z:y=z['y_target'];np.testing.assert_array_equal(qid,z['target_indices'])
        n=min(64,max(16,len(qid)//4));order=np.random.default_rng(0).permutation(len(qid))
        adapt,evalpos=order[:n],order[n:]
        assert set(adapt).isdisjoint(evalpos) and len(evalpos)>16
        dest=OUT/'adaptation_labels'/c['track']/c['task']/f'{c["task_id"]}.npz'
        source.npwrite(dest,adaptation_positions=adapt,adaptation_y=y[adapt],
            adaptation_indices=qid[adapt],evaluation_positions=evalpos,evaluation_indices=qid[evalpos])
        truth=OUT/'evaluation_truth'/c['track']/c['task']/f'{c["task_id"]}.npz'
        source.npwrite(truth,y_target=y[evalpos],target_indices=qid[evalpos])
        row=dict(c,adaptation_file=str(dest),adaptation_sha256=source.sha(dest),
            heldout_truth_file=str(truth),heldout_truth_sha256=source.sha(truth),
            adaptation_rows=n,evaluation_rows=len(evalpos))
        manifest.append(row)
    assert len(manifest)==175
    atomic_json(OUT/'case_manifest.json',manifest)
    assert (ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size==0
    atomic_json(OUT/'audit.json',{'status':'PASS','cases':175,'conditions':2175,
        'adaptation_and_evaluation_row_ids_disjoint':True,'all_methods_same_eval_subset':True,
        'both_tracks_use_label_adaptation_explicitly':True,'Markdown_bytes':0})


if __name__=='__main__':main()
