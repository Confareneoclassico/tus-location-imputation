#!/usr/bin/env python3
"""Copy the aggregate outputs of location_imputation.py into results/v<version>/.

The pipeline writes aggregate tables and microdata (the PRIVATE_ files) into the same
output folder, which should live under data/restricted/. This script is the only route
from there into the repository: it copies the aggregate files, never a PRIVATE_ file or
any file type other than .csv, .csv.gz, .json and .txt, and removes the local folder names from
the paths recorded in run_settings.json, so that the copy can be committed. Files that an
earlier run left in the folder, which run_settings.json lists, are not copied either.

Usage:
    python src/export_results.py data/restricted/imputation_out
    python src/export_results.py data/restricted/imputation_out results/v4.1
Without a second argument the copy goes to results/v<version>/, with the version read
from run_settings.json.
"""
import csv
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

SHAREABLE = {".csv", ".json", ".txt", ".gz"}


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1]).expanduser().resolve()
    settings_file = source / "run_settings.json"
    if not settings_file.is_file():
        raise SystemExit(f"{settings_file} not found: is {source} an output folder of location_imputation.py?")
    settings = json.loads(settings_file.read_text())
    version = settings["version"].split()[0]
    repo = Path(__file__).resolve().parent.parent
    dest = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else repo / "results" / f"v{version}"
    if dest == source:
        raise SystemExit("source and destination are the same folder")
    dest.mkdir(parents=True, exist_ok=True)

    # Local folder that held the country files, to be removed from anything written to dest
    files = settings.get("files", {})
    prefix = os.path.commonpath([os.path.dirname(p) for p in files.values()]) if files else None
    if files:
        settings["files"] = {code: os.path.relpath(path, prefix) for code, path in files.items()}
        settings["files_note"] = ("paths relative to the folder that held the country files "
                                  "(local folder names removed by export_results.py)")

    # Files that an earlier run left in the output folder (recorded by the pipeline from version 4.0)
    stale = set(settings.get("files_of_earlier_runs", []))
    copied, skipped, left = [], [], []
    for f in sorted(source.iterdir()):
        if not f.is_file():
            continue
        if f.name in stale:
            left.append(f.name)
            continue
        if (f.name.startswith("PRIVATE_") or f.suffix.lower() not in SHAREABLE
                or (f.suffix.lower() == ".gz" and not f.name.lower().endswith(".csv.gz"))):
            skipped.append(f.name)
            continue
        if f.name == "run_settings.json":
            (dest / f.name).write_text(json.dumps(settings, indent=2) + "\n")
        elif f.name.lower().endswith(".csv.gz"):
            text = gzip.decompress(f.read_bytes()).decode("utf-8")
            if prefix and prefix in text:
                text = text.replace(prefix + os.sep, "").replace(prefix, ".")
            (dest / f.name).write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
            shutil.copystat(f, dest / f.name)
        else:
            text = f.read_text(encoding="utf-8")
            if prefix and prefix in text:
                text = text.replace(prefix + os.sep, "").replace(prefix, ".")
            (dest / f.name).write_text(text, encoding="utf-8")
            shutil.copystat(f, dest / f.name)
        copied.append(f.name)

    status = {}
    checks = dest / "checks.csv"
    if checks.is_file():
        with checks.open(newline="") as fh:
            for row in csv.DictReader(fh):
                status[row["status"]] = status.get(row["status"], 0) + 1
    print(f"version {version}: {len(copied)} aggregate files copied to {dest}")
    print(f"not copied (microdata or other file types): {', '.join(skipped) or 'none'}")
    if left:
        print(f"not copied (left by an earlier run, not outputs of this one): {', '.join(left)}")
    print("checks: " + ", ".join(f"{n} {s}" for s, n in sorted(status.items())))
    if status.get("fail"):
        print("WARNING: some checks failed; see checks.csv before using these results.")


if __name__ == "__main__":
    main()
