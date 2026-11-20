from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URLS = {
    "ihdp_npci_1-100.train.npz": "http://www.fredjo.com/files/ihdp_npci_1-100.train.npz",
    "ihdp_npci_1-100.test.npz": "http://www.fredjo.com/files/ihdp_npci_1-100.test.npz",
}


def download(url: str, output: Path, timeout: int = 60) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    output.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IHDP data for experiment3.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "ihdp"))
    parser.add_argument("--no-download", action="store_true", help="Only check whether expected files exist.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for filename, url in DEFAULT_URLS.items():
        path = data_dir / filename
        if path.exists():
            print(f"[experiment3] found {path}")
            continue
        if args.no_download:
            missing.append(path)
            continue
        print(f"[experiment3] downloading {url} -> {path}")
        try:
            download(url, path)
        except Exception as exc:
            missing.append(path)
            print(f"[experiment3] download failed for {filename}: {exc!r}")
    if missing:
        print("[experiment3] IHDP is not ready. Place these files manually:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)
    print(f"[experiment3] IHDP ready under {data_dir}")


if __name__ == "__main__":
    main()
