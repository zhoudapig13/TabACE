"""Apply the registered train-only reliability adapter to restored pool banks."""
import argparse
import json
import reliability_fusion_v33 as reliability
import full_pool_contexts_v34 as full
from prepare_data import atomic_json


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);p.add_argument('--task',choices=['classification','regression'],required=True);p.add_argument('--block',choices=['clean','five','seven'],required=True);a=p.parse_args()
    reliability.OUT=full.OUT
    reliability.banks.OUT=full.OUT
    reliability.configure()
    reliability.fusion.run(a.budget,a.task,a.block,shared_members=10)
    # Carry the larger selection-pool provenance into final prediction records.
    folder=full.OUT/'predictions'/str(a.budget)/a.task/a.block
    for path in folder.glob('*/*.json'):
        if '.tmp.' in path.name:continue
        relative=path.relative_to(folder)
        bank=full.OUT/'banks/query'/str(a.budget)/a.task/a.block/relative
        source=json.loads(bank.read_text());result=json.loads(path.read_text())
        result.update(selection_scope=source['selection_scope'],
            available_observed_pool_rows=source['available_observed_pool_rows'],
            Base_consumed_rows=source['Base_consumed_rows'],pool_input_hashes=source['pool_input_hashes'],
            source_bank=str(bank),adaptation_version='v34',
            same_selected_rows_as_Base=False,extra_pre_corruption_labels_used_by_constructors=False,
            original_paper_replication=False)
        atomic_json(path,result)
