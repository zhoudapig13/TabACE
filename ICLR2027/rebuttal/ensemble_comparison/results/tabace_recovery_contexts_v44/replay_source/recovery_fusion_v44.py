"""Refit the registered reliability rule on matched v43-observation banks."""
import argparse
import json
import reliability_fusion_v33 as fusion
import recovery_contexts_v44 as banks
from prepare_data import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);p.add_argument('--block',choices=['five','seven'],required=True);a=p.parse_args()
    fusion.OUT=banks.OUT;fusion.banks.OUT=banks.OUT;fusion.configure()
    fusion.fusion.run(a.budget,'regression',a.block,shared_members=10)
    folder=banks.OUT/'predictions'/str(a.budget)/'regression'/a.block
    for path in folder.glob('*/*.json'):
        if '.tmp.' in path.name:continue
        source=banks.OUT/'banks/query'/str(a.budget)/'regression'/a.block/path.relative_to(folder)
        meta=json.loads(source.read_text());out=json.loads(path.read_text())
        out.update(source_bank=str(source),Base_registry=str(banks.REGISTRY),
            selection_scope=meta['selection_scope'],available_observed_pool_rows=meta['available_observed_pool_rows'],
            extra_clean_TRAIN_information_in_benchmark=True,query_feedback_used_for_benchmark_design=True,
            original_paper_replication=False,independent_test_claim=False,adaptation_version='v44')
        atomic_json(path,out)
