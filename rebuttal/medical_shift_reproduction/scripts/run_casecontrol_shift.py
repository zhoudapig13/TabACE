"""Reuse unchanged comparison methods under a separately frozen sampling protocol."""
import argparse
import json
import time
import run_public_shift as experiment

experiment.DATA = experiment.ROOT/'data/processed/public_casecontrol_v1'
experiment.OUT = experiment.ROOT/'results/public_casecontrol_v1'
experiment.PLAN = json.loads((experiment.ROOT/'configs/public_casecontrol_protocol.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['sepsis_A_to_B', 'sepsis_B_to_A'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    deadline = time.monotonic()+1800
    while not (experiment.DATA/args.task/'metadata.json').exists():
        if time.monotonic() > deadline:
            raise SystemExit('Input preparation timeout.')
        time.sleep(2)
    experiment.run(args.task, args.gpu)
