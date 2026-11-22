# Project backup and restoration

The owner requested omitting unnecessary runtime files from the server backup. The recommended snapshot is the [curated-project-20261122 release](https://github.com/zhoudapig13/TabACE/releases/tag/curated-project-20261122).

It preserves source code, paper files and figures, configurations, aggregate experiment results, provenance, and audit records, including records from multiple development attempts. The selection contains 14,274 files totaling 1,013,130,084 bytes before compression. The original project-relative layout remains under `ICLR2027/`. The archive contains `ICLR2027/BACKUP_SELECTION.json`, which lists every selected file and omitted directory category.

The snapshot excludes re-downloadable raw datasets, prediction banks and per-query caches, virtual environments, training progress logs, retrainable model checkpoints, and dense runtime directories. The local originals were not deleted. Credentials, private keys, and Git metadata are excluded from the archive.

## Completion check

A release is complete only when it contains `BACKUP_COMPLETE.json`, `SHA256SUMS`, and `RESTORE.md`. The numbered archive volumes are checked against GitHub SHA-256 digests.

## Restore

Install GitHub CLI and Zstandard, then run:

```bash
gh release download curated-project-20261122 --repo zhoudapig13/TabACE --dir curated-backup
cd curated-backup
sha256sum -c SHA256SUMS
mkdir -p restored
cat TabACE-curated.tar.zst.part-* | zstd -d | tar -xf - -C restored
```

The result is `restored/ICLR2027/`. The ordinary Git checkout remains the fastest way to obtain the project code, README, and paper.

## Earlier raw-directory uploads

The `server-backup-20261120` and `essential-backup-20261120` uploads were stopped when the backup scope changed. They contain incomplete raw archive volumes and are superseded by the curated snapshot above.
