"""Schedule 30 full source-fit/inference repetitions, one worker per GPU."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import queue
import subprocess
import sys
import threading
import time

from multi_seed_natural_v13 import DEST, PLAN, RUN, SCENARIOS, TASKS
from prepare_acs import save_json, sha


if __name__ == "__main__":
    if not PLAN.exists():
        raise RuntimeError("Make the frozen plan first")
    gate = DEST / "seed_0/geographic/Income/completion.json"
    check = json.loads(gate.read_text())
    assert check["plan_sha256"] == sha(PLAN)
    assert all(row["Shared_max_probability_difference"] == 0. and row["Specific_max_probability_difference"] == 0.
               for row in check["seed0_reference_checks"])
    gpu_queue = queue.Queue()
    for gpu in range(8):
        gpu_queue.put(gpu)
    lock = threading.Lock()
    jobs = {(seed, scenario, task): {"seed": seed, "scenario": scenario, "task": task, "status": "PENDING"}
            for seed in range(5) for scenario in SCENARIOS for task in TASKS}

    def persist():
        save_json(RUN / "configs/multi_seed_v13_workers.json", {"plan_sha256": sha(PLAN), "jobs": list(jobs.values())})

    def worker(job):
        seed, scenario, task = job
        marker = DEST / f"seed_{seed}" / scenario / task / "completion.json"
        if marker.exists():
            old = json.loads(marker.read_text())
            assert old["plan_sha256"] == sha(PLAN)
            with lock:
                jobs[job]["status"] = "COMPLETE_CACHED"; persist()
            return
        gpu = gpu_queue.get()
        try:
            command = [sys.executable, "-u", str(RUN / "scripts/multi_seed_natural_v13.py"),
                       "--seed", str(seed), "--scenario", scenario, "--task", task]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED=str(seed))
            path = RUN / "logs" / f"multi_v13_seed{seed}_{scenario}_{task}.log"
            with path.open("a") as stream:
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                with lock:
                    jobs[job].update(status="RUNNING", gpu=gpu, pid=process.pid, log=str(path), command=command)
                    persist()
                    print("LAUNCHED", seed, scenario, task, "GPU", gpu, "PID", process.pid, flush=True)
                status = process.wait()
            with lock:
                jobs[job].update(status="COMPLETE" if status == 0 else "FAILED", exit_code=status)
                persist()
                complete = sum(item["status"].startswith("COMPLETE") for item in jobs.values())
                print("FINISHED", seed, scenario, task, "code", status, "completed", complete, "/30", flush=True)
            if status:
                raise RuntimeError(f"Worker failed: {job}; inspect {path}")
        finally:
            gpu_queue.put(gpu)

    failures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(worker, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append({"job": futures[future], "error": repr(error)})
    if failures:
        raise RuntimeError(f"Failed repetitions: {failures}")
    print("ALL 30 REPETITIONS COMPLETE", flush=True)
