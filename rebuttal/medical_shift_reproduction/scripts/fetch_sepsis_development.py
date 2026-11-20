"""Uniformly acquire additional, previously unselected target-development patients."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sqlite3
import numpy as np
from fetch_sepsis_public import ROOT, inventory, get_record


if __name__ == '__main__':
    base = ROOT/'data/raw/sepsis_public'
    original = json.loads((base/'selection.json').read_text())
    db = sqlite3.connect(base/'records.sqlite')
    selected = {}
    for hospital in ['A', 'B']:
        old = set(original[hospital]['keys'])
        remaining = sorted(set(inventory(hospital))-old)
        chosen = sorted(np.random.default_rng(0).choice(remaining, 4096, replace=False).tolist())
        assert not set(chosen).intersection(old)
        selected[hospital] = dict(keys=chosen, n=len(chosen), selection='uniform unused inventory; seed0')
        cached = {r[0] for r in db.execute('SELECT key FROM records')}
        pending = [k for k in chosen if k not in cached]
        print('DEVELOPMENT_DOWNLOAD_START', hospital, len(pending), flush=True)
        failures = 0
        with ThreadPoolExecutor(32) as pool:
            jobs = [pool.submit(get_record, key) for key in pending]
            for done, future in enumerate(as_completed(jobs), 1):
                try:
                    key, raw, digest = future.result()
                    db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)', (key, raw, digest))
                except Exception as exc:
                    failures += 1
                    print('DOWNLOAD_FAILURE_TYPE', type(exc).__name__, flush=True)
                if done % 100 == 0:
                    db.commit()
                if done % 500 == 0 or done == len(pending):
                    print('DEVELOPMENT_DOWNLOAD_PROGRESS', hospital, done, len(pending), flush=True)
        db.commit()
        if failures:
            raise SystemExit('Incomplete download; resume cached files.')
    db.close()
    (base/'development_selection.json').write_text(json.dumps(selected, indent=2))
    print('DEVELOPMENT_DOWNLOAD_COMPLETE', flush=True)
