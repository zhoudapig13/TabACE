"""Run the approved case-control development stages without final-test feedback."""
import argparse
import torch
import tune_medical_development as main
import tune_medical_local as local
import tune_medical_gate_floor as floor


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['initial','local','floor'],required=True)
    p.add_argument('--gpu',type=int,default=0);args=p.parse_args()
    root=main.ROOT;data=root/'data/processed/target_development_readmission_cc_v5'
    initial=root/'results/target_development_readmission_cc_v5'
    local_out=root/'results/target_development_readmission_cc_local_v6'
    floor_out=root/'results/target_development_readmission_cc_floor_v7'
    main.DATA=data;main.OUT=initial;main.PLAN_PATH=root/'configs/target_development_readmission_cc_v5.json'
    main.EVALUATE_TEST=False
    local.DATA=data;local.BASE_RESULTS=initial;local.LOCAL_OUT=local_out
    floor.DATA=data;floor.BASE_RESULTS=initial;floor.LOCAL_RESULTS=local_out;floor.FLOOR_OUT=floor_out
    torch.set_num_threads(1);torch.manual_seed(0)
    task='readmission_other_to_ER'
    if args.stage=='initial':main.Worker(task,args.gpu).run()
    elif args.stage=='local':local.LocalWorker(task,args.gpu).run_local()
    else:floor.FloorWorker(task,args.gpu).run_floor()
