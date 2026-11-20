"""One-command reproduction, with cached feature inference and all trial logs.

Usage: <ensemble_comparison/.venv/bin/python> scripts/reproduce.py
Defaults to six independent GPU workers on GPUs 0--5. No agents are spawned.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from prepare_acs import RUN


def stage(script, extra=()):
    command = [sys.executable, "-u", str(RUN / "scripts" / script), *extra]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def gpu_stage(script, gpus, extra=()):
    jobs = [(s, t) for s in ["geographic", "temporal"] for t in ["Employment", "Income", "TravelTime"]]
    for offset in range(0, len(jobs), len(gpus)):
        children = []
        for gpu, (scenario, task) in zip(gpus, jobs[offset:offset+len(gpus)]):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [sys.executable, "-u", str(RUN / "scripts" / script), "--scenario", scenario, "--task", task, *extra]
            log = (RUN / "logs" / f"reproduce_{Path(script).stem}_{scenario}_{task}.log").open("a")
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log, command))
        failed = []
        for child, log, command in children:
            code = child.wait(); log.close()
            if code:
                failed.append((command, code))
        if failed:
            raise RuntimeError(f"GPU stage failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--all-history", action="store_true", help="Also reproduce exploratory earlier budgets up to 2048; default uses only the active 512-row protocol")
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    stage("prepare_acs.py")
    gpu_stage("run_natural.py", args.gpus.split(","), [] if args.all_history else ["--base-only"])
    if args.all_history:
        stage("fit_and_report.py")
        gpu_stage("run_expanded_views.py", args.gpus.split(","))
        stage("fit_expanded.py")
        gpu_stage("run_expanded_views.py", args.gpus.split(","), ["--coverage"])
        stage("fit_expanded.py", ["--coverage"])
    budgets = "256,512,1024,2048" if args.all_history else "512"
    gpu_stage("run_row_budgets.py", args.gpus.split(","), ["--budgets", budgets])
    if args.all_history:
        stage("fit_row_budgets.py")
        stage("fit_row_budgets.py", ["--matched"])
    if args.all_history:
        stage("fit_row_budgets.py", ["--fixed-rows", "512"])
        stage("nested_source_adapter.py")
        stage("correct_supervision_metadata.py")
        stage("audit_natural.py")
        stage("audit_nested_source_adapter.py")
    stage("strict_nested_source.py")
    stage("audit_nested_source_adapter.py", ["--strict-source"])
