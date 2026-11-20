"""Resume failed shards with the documented constant-subcontext fix."""
import argparse
import tabpfn_official_methods_v2 as engine
import tabpfn_constant_context_compat_v2 as compat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['classification', 'regression'], required=True)
    parser.add_argument('--shard', type=int, required=True); parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    compat.install()
    engine.raw_prediction = compat.scope_prediction(engine.raw_prediction)
    engine.run(args.shard, args.shards, selected_task=args.task)
