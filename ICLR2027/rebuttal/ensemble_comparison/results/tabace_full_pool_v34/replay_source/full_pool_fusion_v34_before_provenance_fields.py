"""Apply the registered train-only reliability adapter to restored pool banks."""
import argparse
import reliability_fusion_v33 as reliability
import full_pool_contexts_v34 as full


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);p.add_argument('--task',choices=['classification','regression'],required=True);p.add_argument('--block',choices=['clean','five','seven'],required=True);a=p.parse_args()
    reliability.OUT=full.OUT
    reliability.banks.OUT=full.OUT
    reliability.configure()
    reliability.fusion.run(a.budget,a.task,a.block,shared_members=10)
