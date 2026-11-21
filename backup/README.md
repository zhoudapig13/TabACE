# Working-directory backups

Source directory: `ICLR2027/`.

The Git repository preserves source, documents, and selected reports under `ICLR2027/`. Large working files are uploaded as checked archive volumes in GitHub Releases:

- [`essential-backup-20261120`](https://github.com/zhoudapig13/TabACE/releases/tag/essential-backup-20261120): project snapshot excluding prediction-bank directories named `banks`, plus the exclusions below.
- [`server-backup-20261120`](https://github.com/zhoudapig13/TabACE/releases/tag/server-backup-20261120): full project snapshot, including prediction banks, with the exclusions below.

An archive is complete only when its release contains `BACKUP_COMPLETE.json`, `SHA256SUMS`, and `RESTORE.md`. Upload workers resume matching volumes by SHA-256 and retry temporary network failures. A partial release must not be mistaken for a complete backup.

The archive preserves the original project directory, regular files, and symbolic links, including datasets, model weights, and historical outputs. Disposable Git metadata, environment/cache directories, and credential files are excluded. Consult the manifest for exact exclusions.

## Restore

Install GitHub CLI and Zstandard, then download all assets into an empty directory:

```bash
gh release download server-backup-20261120 --repo zhoudapig13/TabACE --dir backup-assets
cd backup-assets
sha256sum -c SHA256SUMS
mkdir -p restored
cat *.tar.zst.part-* | zstd -d | tar -xf - -C restored
```

The restored `ICLR2027/` directory contains the archived working files. `BACKUP_COMPLETE.json` records the uploaded chunk names, byte counts, and SHA-256 checksums. Keep all chunks together and in their original numbered order.
