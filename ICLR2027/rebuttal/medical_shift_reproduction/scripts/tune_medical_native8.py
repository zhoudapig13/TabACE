"""Common Native-8 backbone control: equal native estimator count for all methods."""
import argparse
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import json
import torch
from threadpoolctl import threadpool_limits
import tune_medical_development as main
import tune_medical_local as local
from tune_expanded_medical import ExpandedProbability


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();main.DATA=main.ROOT/'data/processed/expanded_medical_development_v18';local.DATA=main.DATA
    torch.set_num_threads(1);torch.manual_seed(0)
    with threadpool_limits(limits=1):
        w=ExpandedProbability(args.task,args.gpu,'raw')
        w.model.n_estimators=8
        w.results_tag='medical_common_native8_v24'
        w.predictions_filename='medical_common_native8_predictions.npz'
        w.cache_tag='member_cache_known_domain_native8'
        w.protocol_path=main.ROOT/'configs/medical_common_native8_v24.json'
        assert w.model.get_params()['n_estimators']==8
        w.run_domain();w.supplement_snapshot()
        path=main.ROOT/'results'/w.results_tag/args.task/'frozen_selection.json'
        f=json.loads(path.read_text());f.update(native_estimators_per_branch=8,context_cap_per_branch=256,
            native_estimator_count_matched_across_all_methods=True,
            native1_primary_results_preserved=True,not_backbone_weight_finetuning=True)
        path.write_text(json.dumps(f,indent=2))
