"""Reproduce the supervised router with the existing frozen Shared baseline.

This entry point never refits or replaces Shared. For a fresh installation,
first run reproduce.py to reconstruct the 512-row TFM caches and v8 reference.
"""
import os
from pathlib import Path
import subprocess
import sys

from prepare_acs import RUN


def stage(name, *args):
    command = [sys.executable, "-u", str(RUN / "scripts" / name), *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    reference = RUN / "results/strict_source_nested_512_v8"
    if not (reference / "audit.json").exists():
        raise RuntimeError("Missing frozen v8 baseline: run scripts/reproduce.py first")
    processes = []
    for scenario in ["geographic", "temporal"]:
        for task in ["Employment", "Income", "TravelTime"]:
            command = [sys.executable, "-u", str(RUN / "scripts/specific_query_adapter_v9.py"),
                       "--scenario", scenario, "--task", task]
            stream = (RUN / "logs" / f"reproduce_query_{scenario}_{task}.log").open("a")
            processes.append((subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT), stream, command))
    failures = []
    for process, stream, command in processes:
        status = process.wait(); stream.close()
        if status:
            failures.append((command, status))
    if failures:
        raise RuntimeError(f"Training failed: {failures}")
    stage("specific_query_adapter_v9.py", "--evaluate")
    stage("audit_query_adapters.py")
    stage("specific_convex_router_v10.py")
    stage("specific_convex_router_v10.py", "--evaluate")
    stage("audit_query_adapters.py", "--convex")
