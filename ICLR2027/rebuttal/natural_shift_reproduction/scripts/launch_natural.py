"""Launch six isolated GPU workers; record every PID and command."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from prepare_acs import RUN, save_json

parser = argparse.ArgumentParser()
parser.add_argument("--base-only", action="store_true")
args = parser.parse_args()
workers = []
for gpu, (scenario, task) in enumerate((s, t) for s in ["geographic", "temporal"]
                                      for t in ["Employment", "Income", "TravelTime"]):
    command = [sys.executable, "-u", str(Path(__file__).with_name("run_natural.py")),
               "--scenario", scenario, "--task", task]
    if args.base_only:
        command.append("--base-only")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    log = RUN / "logs" / f"{scenario}_{task}{'_base' if args.base_only else ''}.log"
    with log.open("a") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    workers.append({"gpu": gpu, "pid": process.pid, "scenario": scenario, "task": task,
                    "command": command, "log": str(log)})
save_json(RUN / "configs/workers.json", {"workers": workers, "seed": 0})
for item in workers:
    print(item["gpu"], item["pid"], item["scenario"], item["task"], flush=True)
