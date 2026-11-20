"""Verify the selected raw files locally without printing patient records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def verify(destination: Path, urls_path: Path):
    expected = {}
    for line in (destination / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.lstrip('*').removeprefix('./')] = digest
    records = []
    for url in urls_path.read_text().splitlines():
        if not url.strip():
            continue
        relative = '/'.join(urlsplit(url).path.strip('/').split('/')[3:])
        if relative == 'SHA256SUMS.txt':
            continue
        path = destination / relative
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        ok = h.hexdigest() == expected.get(relative)
        records.append({'file': relative, 'bytes': path.stat().st_size,
                        'sha256': h.hexdigest(), 'verified': ok})
        print(('PASS ' if ok else 'FAIL ') + relative, flush=True)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_urls': urls_path.read_text().splitlines(), 'files': records}
    (destination / 'download_verification.json').write_text(json.dumps(report, indent=2))
    if not records or not all(row['verified'] for row in records):
        raise SystemExit('Some files failed verification; do not use this download.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('urls', type=Path)
    args = parser.parse_args()
    verify(args.destination, args.urls)
