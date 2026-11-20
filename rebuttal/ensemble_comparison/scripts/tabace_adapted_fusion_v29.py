"""Train-only fusion for explicitly marked v29 candidate adaptations."""
import argparse
import tabace_fusion_v28 as fusion
import tabace_context_adaptation_v29 as adapted


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--budget',type=int,required=True)
    parser.add_argument('--task',choices=['classification','regression'],required=True)
    parser.add_argument('--block',choices=['clean','five','seven'],required=True)
    args=parser.parse_args()
    fusion.main.OUT=adapted.OUT
    fusion.GAINS=[1.,2.,4.,8.]
    fusion.GATE_VARIANT='entropy_or_disagreement'
    fusion.run(args.budget,args.task,args.block,shared_members=10)
