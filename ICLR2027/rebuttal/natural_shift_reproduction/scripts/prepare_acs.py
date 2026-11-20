"""Official ACS tasks, frozen household-disjoint reconstruction (seed 0).

The original Table 2 splits are missing. These are new, explicitly recorded
splits, not recovered originals. Test labels are stored separately from features.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile

import folktables
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit

RUN = Path(__file__).resolve().parents[1]
TASKS = {"Employment": folktables.ACSEmployment,
         "Income": folktables.ACSIncome,
         "TravelTime": folktables.ACSTravelTime}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(job):
    year, state = job
    directory = RUN / "data/raw" / str(year)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"csv_p{state.lower()}.zip"
    url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/{archive.name}"
    if not archive.exists():
        temp = archive.with_suffix(".part")
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temp.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        candidates = [name for name in z.namelist() if name.lower().endswith(".csv")]
        names = [name for name in candidates if "psam_p" in name.lower() or f"p{state.lower()}" in name.lower()]
        if len(names) != 1:
            raise ValueError(f"Ambiguous person CSV in {archive}: {candidates}")
        member = names[0]
        wanted = set().union(*(set(problem.features) for problem in TASKS.values()))
        wanted.update(["ESR", "PINCP", "JWMNP", "PWGTP", "SERIALNO", "WKHP"])
        with z.open(member) as handle:
            frame = pd.read_csv(handle, usecols=lambda name: name.strip() in wanted,
                                dtype={"SERIALNO": str}, skipinitialspace=True)
    save_json(directory / f"{state}_provenance.json", {
        "url": url, "archive_sha256": sha(archive), "member": member,
        "raw_rows": len(frame), "year": year, "state": state,
        "folktables_version": folktables.__version__,
    })
    print(f"DATA {year} {state}: {len(frame)} rows", flush=True)
    for task_index, (name, problem) in enumerate(TASKS.items()):
        out = RUN / "data/splits" / f"{year}_{state}_{name}"
        if out.with_suffix(".json").exists():
            continue
        selected = problem._preprocess(frame)
        X, y, _ = problem.df_to_numpy(frame)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        row_ids = selected.index.to_numpy(dtype=np.int64)
        households = selected["SERIALNO"].astype(str).to_numpy(dtype=str)
        assert len(X) == len(row_ids) == len(y)
        # Source-only fit/selection and test roles are consistent across uses of
        # CA2018, so global Shared calibration cannot overlap temporal test rows.
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
        development, test = next(splitter.split(X, y, households))
        inner = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=0)
        train_ids, cal_ids = next(inner.split(X[development], y[development], households[development]))
        train_ids, cal_ids = development[train_ids], development[cal_ids]
        rng = np.random.default_rng(0)
        pool = rng.permutation(train_ids)[:2048]
        cal = rng.permutation(cal_ids)[:4096]
        test = rng.permutation(test)[:10000]
        canonical = np.random.default_rng(0).permutation(len(pool))[:256]
        roles = [set(households[pool]), set(households[cal]), set(households[test])]
        assert all(roles[i].isdisjoint(roles[j]) for i in range(3) for j in range(i))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out) + "_source.npz", X=X[pool], y=y[pool],
                            row_ids=row_ids[pool], households=households[pool], canonical=canonical)
        np.savez_compressed(str(out) + "_calibration.npz", X=X[cal], y=y[cal],
                            row_ids=row_ids[cal], households=households[cal])
        np.savez_compressed(str(out) + "_test_features.npz", X=X[test],
                            row_ids=row_ids[test], households=households[test])
        np.savez_compressed(str(out) + "_test_labels.npz", y=y[test], row_ids=row_ids[test])
        metadata = {
            "task": name, "official_definition": f"folktables.ACS{name}",
            "year": year, "state": state, "seed": 0, "features": list(problem.features),
            "filtered_population_rows": len(y), "source_pool_rows": len(pool),
            "canonical_context_rows": len(canonical), "calibration_rows": len(cal),
            "test_rows": len(test), "source_positive_fraction": float(y[pool].mean()),
            "household_disjoint": True, "original_split_recovered": False,
            "test_labels_used_for_selection": False,
        }
        save_json(out.with_suffix(".json"), metadata)
        print(f"SPLIT {year} {state} {name}: pool={len(pool)} calibration={len(cal)} test={len(test)}", flush=True)
    return {"year": year, "state": state}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = list(executor.map(fetch, [(2018, "CA"), (2018, "MI"), (2014, "CA")]))
    save_json(RUN / "configs/data_manifest.json", {
        "status": "COMPLETE", "seed": 0, "domains": jobs,
        "protocol": "reconstructed_official_acs_household_disjoint_v1",
        "notes": ["Original Table 2 scripts and splits are unavailable.",
                  "No corruption, label alteration, or test-score matching is applied.",
                  "2048 accessible labeled source rows; Base randomly uses 256.",
                  "4096 disjoint source calibration rows provide additional supervision for fusion adaptation."]})
