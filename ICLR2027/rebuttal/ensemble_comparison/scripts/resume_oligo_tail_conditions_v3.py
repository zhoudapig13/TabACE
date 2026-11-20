"""Rebalance unfinished conditions without changing scientific configuration."""
import argparse
import json
import logging
import warnings

import torch

from tabpfn_official_benchmark_v2 import OUT
from tabpfn_oligoicp_v2 import run_cell
from tabpfn_official_methods_v2 import adapter
from run_tabpfn import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1); args = parser.parse_args()
    torch.set_num_threads(1); logging.disable(logging.INFO); warnings.filterwarnings('ignore')
    work = json.loads((OUT / 'oligo400_tail_queue.json').read_text())['conditions'][args.shard::args.shards]
    estimator = model('regression', 'cuda:0'); cache = {}
    for entry in work:
        track, meta, family, rho = entry
        key = (track, meta['task_id'])
        if key not in cache:
            cache[key] = adapter('regression', track, meta)
        auxiliary, policy = cache[key]
        run_cell(meta, track, family, rho, estimator, auxiliary, policy, 400)
