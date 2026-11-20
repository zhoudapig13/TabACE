"""Run the ten disjoint dataset-fold jobs with a bounded CPU worker pool."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

SCRIPT = Path(__file__).with_name("paper_rule_reconstruction_v77.py")
ROOT = SCRIPT.parents[1] / "results/tabace_paper_rule_reconstruction_v77"


def run_job(task, fold):
    path = ROOT / "logs" / f"{task}_{fold}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        process = subprocess.Popen([sys.executable, str(SCRIPT), "worker", "--task", task, "--fold", str(fold)],
                                   stdout=handle, stderr=subprocess.STDOUT)
        code = process.wait()
    return {"task": task, "fold": fold, "returncode": code, "log": str(path)}


if __name__ == "__main__":
    assert (ROOT / "plan.json").exists()
    records = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_job, task, fold)
                   for fold in range(5) for task in ["classification", "regression"]]
        for future in as_completed(futures):
            records.append(future.result())
            state = {"started_unix": started, "status": "RUNNING", "workers": records}
            (ROOT / "execution.json").write_text(json.dumps(state, indent=2) + "\n")
            subprocess.run([sys.executable, str(SCRIPT), "report"], stdout=subprocess.DEVNULL, check=True)
            print("FINISHED", records[-1], flush=True)
    state = {"started_unix": started, "elapsed_seconds": time.time() - started,
             "status": "COMPLETE" if all(r["returncode"] == 0 for r in records) else "FAILED",
             "workers": records}
    (ROOT / "execution.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)
