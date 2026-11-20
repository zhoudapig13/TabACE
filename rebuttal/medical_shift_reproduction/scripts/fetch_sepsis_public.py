"""Download a predeclared uniform patient sample from the official public S3 mirror.

Raw records stay in a resumable local SQLite cache; no patient values are logged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
LOCAL = threading.local()
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
ORIGIN = 'https://physionet-open.s3.amazonaws.com/'


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session


def inventory(hospital):
    out = ROOT / 'data/raw/sepsis_public' / f'inventory_{hospital}.json'
    if out.exists():
        return json.loads(out.read_text())
    prefix = f'challenge-2019/1.0.0/training/training_set{hospital}/'
    keys, token = [], None
    while True:
        args = {'list-type': 2, 'prefix': prefix, 'max-keys': 1000}
        if token:
            args['continuation-token'] = token
        response = session().get(ORIGIN, params=args, timeout=60)
        response.raise_for_status()
        xml = ET.fromstring(response.content)
        keys.extend(n.find('s:Key', NS).text for n in xml.findall('s:Contents', NS)
                    if n.find('s:Key', NS).text.endswith('.psv'))
        if xml.find('s:IsTruncated', NS).text != 'true':
            break
        token = xml.find('s:NextContinuationToken', NS).text
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(keys)))
    print('INVENTORY', hospital, len(keys), flush=True)
    return sorted(keys)


def get_record(key):
    for attempt in range(8):
        try:
            response = session().get(ORIGIN + key, timeout=60)
            response.raise_for_status()
            content = response.content
            if not content.startswith(b'HR|O2Sat|Temp|') or b'SepsisLabel' not in content.split(b'\n', 1)[0]:
                raise ValueError('Unexpected clinical file schema')
            return key, content, hashlib.sha256(content).hexdigest()
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(2**attempt, 20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patients-per-hospital', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    base = ROOT / 'data/raw/sepsis_public'
    base.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(base / 'records.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, raw BLOB, sha256 TEXT)')
    selection = {}
    failures = []
    for hospital in ['A', 'B']:
        keys = inventory(hospital)
        rng = np.random.default_rng(0)
        indices = rng.choice(len(keys), min(args.patients_per_hospital, len(keys)), replace=False)
        selected = sorted(keys[i] for i in indices)
        selection[hospital] = {'available': len(keys), 'selected': len(selected),
                               'selection_seed': 0, 'keys': selected}
        existing = {row[0] for row in db.execute('SELECT key FROM records')}
        missing = [key for key in selected if key not in existing]
        print('DOWNLOAD_START', hospital, 'missing', len(missing), flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(get_record, key) for key in missing]
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    key, content, digest = future.result()
                except Exception as exc:
                    failures.append({'hospital': hospital, 'error_type': type(exc).__name__})
                    print('RECORD_RETRY_EXHAUSTED', hospital, type(exc).__name__, flush=True)
                    continue
                db.execute('INSERT OR REPLACE INTO records VALUES (?, ?, ?)', (key, content, digest))
                if done % 100 == 0:
                    db.commit()
                if done % 500 == 0 or done == len(missing):
                    print('DOWNLOADED', hospital, done, '/', len(missing), flush=True)
        db.commit()
    db.close()
    (base / 'download_status.json').write_text(json.dumps({'failed': failures}, indent=2))
    if failures:
        raise SystemExit('Incomplete public download; rerun to resume cached files.')
    (base / 'selection.json').write_text(json.dumps(selection, indent=2))
    print('PUBLIC_SEPSIS_DOWNLOAD_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
