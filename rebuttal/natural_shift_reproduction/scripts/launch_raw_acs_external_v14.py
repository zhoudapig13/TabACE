"""Run each seed/case once, on one GPU per worker, retaining failures and logs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import time

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "results/raw_acs_external_v14"
MAIN = RUN.parent / "ensemble_comparison/.venv/bin/python"
DRIFT = RUN / ".drift-venv/bin/python"
SCRIPT = RUN / "scripts/raw_acs_external_v14.py"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = Queue()
    for seed in range(5):
        for scenario in ["geographic", "temporal"]:
            for task in ["Employment", "Income", "TravelTime"]:
                for backend in ["distpfn", "drift"]:
                    jobs.put((backend, scenario, task, seed))
    statuses = []

    def worker(gpu):
        while True:
            try:
                backend, scenario, task, seed = jobs.get_nowait()
            except Empty:
                return
            name = f"{backend}__{scenario}__{task}__seed{seed}"
            python = DRIFT if backend == "drift" else MAIN
            env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            t0 = time.time()
            log = RUN / "logs" / f"v14_{name}.log"
            with log.open("w") as f:
                result = subprocess.run([str(python), "-u", str(SCRIPT), "--backend", backend,
                                         "--scenario", scenario, "--task", task,
                                         "--seed", str(seed)], env=env, stdout=f, stderr=subprocess.STDOUT)
            row = {"job": name, "gpu": gpu, "exit_code": result.returncode,
                   "seconds": round(time.time() - t0, 2), "log": str(log)}
            statuses.append(row)
            print(json.dumps(row), flush=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker, range(8)))
    (OUT / "execution_status.json").write_text(json.dumps(statuses, indent=2) + "\n")
    failed = [row for row in statuses if row["exit_code"]]
    print(json.dumps({"completed": len(statuses) - len(failed), "failed": failed}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
