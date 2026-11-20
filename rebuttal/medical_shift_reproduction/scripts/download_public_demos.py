"""Fetch official open demo inputs; validate each file against publisher SHA256.

Only public demo endpoints are used. This does not download credentialed cohorts.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "eicu_demo": {
        "url": "https://physionet.org/files/eicu-crd-demo/2.0.1/",
        "files": ["LICENSE.txt", "patient.csv.gz", "hospital.csv.gz",
                  "lab.csv.gz", "vitalPeriodic.csv.gz", "vitalAperiodic.csv.gz"],
    },
    "mimiciv_demo": {
        "url": "https://physionet.org/files/mimic-iv-demo/2.2/",
        "files": ["LICENSE.txt", "hosp/patients.csv.gz", "hosp/admissions.csv.gz",
                  "icu/icustays.csv.gz", "hosp/labevents.csv.gz",
                  "hosp/d_labitems.csv.gz", "icu/chartevents.csv.gz", "icu/d_items.csv.gz"],
    },
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        with urlopen(url, timeout=60) as src, tmp.open("wb") as dst:
            for block in iter(lambda: src.read(1 << 20), b""):
                dst.write(block)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def run_job(job):
    dataset, base_url, name, expected = job
    path = ROOT / "data/raw" / dataset / name
    if not path.exists() or sha256(path) != expected:
        fetch(base_url + name, path)
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"Checksum mismatch: {dataset}/{name}")
    row = {"dataset": dataset, "url": base_url + name,
           "file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
           "sha256": observed, "publisher_checksum_verified": True}
    print(json.dumps(row), flush=True)
    return row


def main():
    jobs = []
    for dataset, spec in SOURCES.items():
        manifest = ROOT / "data/raw" / dataset / "SHA256SUMS.txt"
        if not manifest.exists():
            fetch(spec["url"] + "SHA256SUMS.txt", manifest)
        expected = {}
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            expected[name.lstrip("*").removeprefix("./")] = digest
        for name in spec["files"]:
            jobs.append((dataset, spec["url"], name, expected[name]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run_job, jobs))
    output = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "PUBLIC_DEMO_ONLY", "files": rows}
    (ROOT / "results/public_demo_downloads.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
