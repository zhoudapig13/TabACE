#!/usr/bin/env bash
# Run manually after your PhysioNet credentialing, training, and project DUA approval.
# Credentials are prompted by wget, not stored in this script or passed as arguments.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"
DATA_DIR="${MEDICAL_DATA_ROOT:-$PROJECT_DIR/data/raw}"
SELECTION="${1:-both}"
case "$SELECTION" in
  eicu) DATASETS=(eicu_full) ;;
  mimiciv) DATASETS=(mimiciv_full) ;;
  both) DATASETS=(eicu_full mimiciv_full) ;;
  *) printf 'Usage: bash %s [eicu|mimiciv|both]\n' "$0" >&2; exit 2 ;;
esac

command -v wget >/dev/null
command -v python3 >/dev/null
mkdir -p -- "$DATA_DIR"
printf 'Destination: %s\n' "$DATA_DIR"
df -h -- "$DATA_DIR"
printf 'Downloads retain .csv.gz compression. Each database asks for your password.\n'
read -r -p 'PhysioNet username: ' PN_USER
if [[ -z "$PN_USER" ]]; then
  printf 'Username must not be empty.\n' >&2
  exit 2
fi

for dataset in "${DATASETS[@]}"; do
  destination="$DATA_DIR/$dataset"
  mkdir -p -- "$destination"
  chmod 700 -- "$destination"
  printf 'Downloading selected tables: %s\n' "$dataset"
  wget --continue --tries=3 --timeout=60 --https-only \
    --user="$PN_USER" --ask-password \
    --force-directories --no-host-directories --cut-dirs=3 \
    --directory-prefix="$destination" \
    --input-file="$PROJECT_DIR/configs/${dataset}_urls.txt"
  python3 "$SCRIPT_DIR/verify_authorized_downloads.py" \
    "$destination" "$PROJECT_DIR/configs/${dataset}_urls.txt"
done
